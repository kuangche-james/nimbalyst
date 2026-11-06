/**
 * Antigravity CLI (`agy`) protocol adapter (`AgentProtocol`).
 *
 * Drives the official `agy` CLI (`agy --print --output-format stream-json`) as a
 * headless per-turn NDJSON process via `runHeadlessNdjson`.
 *
 * Session lifecycle:
 *   - First turn: `createSession()` returns a `ProtocolSession` with `id: ''`.
 *     `sendMessage()` invokes `agy --print --output-format stream-json ... -- <prompt>`
 *     without `--conversation`. The CLI's first NDJSON frame (`event: 'init'`)
 *     carries `conversation_id` and `init.model`, which `sendMessage()` stores
 *     onto `session.id` and `session.appliedModel`.
 *   - Subsequent turns: `resumeSession(sessionId)` restores `session.id`, and
 *     `sendMessage()` passes `--conversation <session.id>` so the CLI resumes
 *     the multi-turn conversation in place.
 */

import { existsSync, readFileSync } from 'node:fs';
import * as path from 'node:path';
import {
  AgentProtocol,
  ProtocolEvent,
  ProtocolMessage,
  ProtocolSession,
  SessionOptions,
} from './ProtocolInterface';
import {
  HeadlessNdjsonItem,
  HeadlessNdjsonRunOptions,
  runHeadlessNdjson,
} from './headless/HeadlessNdjsonProcess';
import { type AgyEditChange, mapAgyRecord } from './headless/AgyRecordMapper';

export {
  type AgyEditChange,
  mapAgyRecord,
  normalizeAgyToolArguments,
  normalizeAgyUsage,
} from './headless/AgyRecordMapper';

const AGY_EDIT_TOOL_NAMES = new Set([
  'write_file',
  'edit_file',
  'write_to_file',
  'replace_file_content',
  'multi_replace_file_content',
]);

export const AGY_INLINE_DOCUMENT_BUDGET = 16_000;

/**
 * Windows `CreateProcessW` caps the entire command line (executable + flags +
 * positional prompt) at 32,767 UTF-16 code units. Bound the assembled command
 * line conservatively at 28,000 characters so quoting/escaping and long paths
 * never prevent a turn from starting on Windows.
 */
export const AGY_MAX_COMMAND_LINE_CHARS = 28_000;

interface AgySessionRaw extends Record<string, unknown> {
  workspacePath: string;
  model?: string;
  systemPrompt?: string;
}

export interface AgyCliProtocolDeps {
  runNdjson?: (options: HeadlessNdjsonRunOptions) => AsyncGenerator<HeadlessNdjsonItem>;
}

export class AgyCliProtocol implements AgentProtocol {
  readonly platform = 'antigravity-gemini-agent';

  private executablePath = process.platform === 'win32' ? 'agy.exe' : 'agy';
  private processEnv: Record<string, string> | undefined;
  private readonly sessions = new Map<string, SessionOptions>();
  private readonly runNdjsonImpl: NonNullable<AgyCliProtocolDeps['runNdjson']>;

  constructor(deps?: AgyCliProtocolDeps) {
    this.runNdjsonImpl = deps?.runNdjson ?? runHeadlessNdjson;
  }

  setAgyPath(executablePath: string): void {
    this.executablePath = executablePath;
  }

  setProcessEnv(env: Record<string, string>): void {
    this.processEnv = env;
  }

  async createSession(options: SessionOptions): Promise<ProtocolSession> {
    const raw: AgySessionRaw = {
      workspacePath: options.workspacePath,
      model: options.model,
      systemPrompt: options.systemPrompt,
    };
    return {
      id: '',
      platform: this.platform,
      deliveredMcpServerCount: 0,
      raw,
    };
  }

  async resumeSession(sessionId: string, options?: SessionOptions): Promise<ProtocolSession> {
    const previous = this.sessions.get(sessionId);
    const merged: SessionOptions = {
      workspacePath: options?.workspacePath ?? previous?.workspacePath ?? process.cwd(),
      model: options?.model ?? previous?.model,
      systemPrompt: options?.systemPrompt ?? previous?.systemPrompt,
      mcpServers: options?.mcpServers ?? previous?.mcpServers,
      permissionMode: options?.permissionMode ?? previous?.permissionMode,
      raw: options?.raw ?? previous?.raw,
    };
    if (sessionId) {
      this.sessions.set(sessionId, merged);
    }
    const raw: AgySessionRaw = {
      workspacePath: merged.workspacePath,
      model: merged.model,
      systemPrompt: merged.systemPrompt,
    };
    return {
      id: sessionId,
      platform: this.platform,
      deliveredMcpServerCount: 0,
      raw,
    };
  }

  async forkSession(_sessionId: string, options: SessionOptions): Promise<ProtocolSession> {
    return this.createSession(options);
  }

  async *sendMessage(
    session: ProtocolSession,
    message: ProtocolMessage,
  ): AsyncIterableIterator<ProtocolEvent> {
    const raw = (session.raw as AgySessionRaw | undefined) ??
      (session.id ? this.sessions.get(session.id) : undefined) ?? { workspacePath: process.cwd() };
    const bareModel = stripProviderPrefix(raw.model);

    const args: string[] = [];
    if (session.id) {
      args.push('--conversation', session.id);
    }
    args.push('--print', '--output-format', 'stream-json');
    if (bareModel) {
      args.push('--model', bareModel);
    }
    if (message.mode === 'planning') {
      args.push('--mode', 'plan');
    }
    // HeadlessCliAgentProvider gates the turn up front on workspace trust.
    args.push('--dangerously-skip-permissions');

    const commandLineOverhead =
      this.executablePath.length +
      args.reduce((sum, arg) => sum + arg.length + 3, 6);
    args.push(
      '--',
      formatAgyPromptWithAttachments(message, raw.systemPrompt, commandLineOverhead),
    );

    const pendingEdits = new Map<
      string,
      {
        absPath: string;
        toolName: string;
        params: Record<string, unknown>;
        beforeContent: string | null;
        fallbackAfterContent?: string;
      }
    >();

    for await (const item of this.runNdjsonImpl({
      command: this.executablePath,
      args,
      cwd: raw.workspacePath,
      env: this.processEnv,
      abortSignal: message.abortSignal,
    })) {
      if (item.kind === 'garbage') continue;

      const record = item.value;
      const eventKind = typeof record.event === 'string' ? record.event : record.type;

      if (eventKind === 'init') {
        const convId = typeof record.conversation_id === 'string' ? record.conversation_id.trim() : '';
        if (convId) {
          session.id = convId;
          this.sessions.set(convId, {
            workspacePath: raw.workspacePath,
            model: raw.model,
            systemPrompt: raw.systemPrompt,
          });
        }
        const initObj =
          record.init && typeof record.init === 'object' && !Array.isArray(record.init)
            ? (record.init as Record<string, unknown>)
            : record;
        if (typeof initObj.model === 'string' && initObj.model.trim()) {
          session.appliedModel = initObj.model.trim();
        }
      }

      if (eventKind === 'result') {
        const resObj =
          record.result && typeof record.result === 'object' && !Array.isArray(record.result)
            ? (record.result as Record<string, unknown>)
            : record;
        const convId =
          typeof resObj.conversation_id === 'string'
            ? resObj.conversation_id.trim()
            : typeof record.conversation_id === 'string'
              ? record.conversation_id.trim()
              : '';
        if (convId && !session.id) {
          session.id = convId;
          this.sessions.set(convId, {
            workspacePath: raw.workspacePath,
            model: raw.model,
            systemPrompt: raw.systemPrompt,
          });
        }
      }

      let editChanges: AgyEditChange[] | undefined;
      if (eventKind === 'step_update') {
        const step =
          record.step_update && typeof record.step_update === 'object' && !Array.isArray(record.step_update)
            ? (record.step_update as Record<string, unknown>)
            : record;
        if (step.step_type === 'tool') {
          const toolInfo =
            step.tool_info && typeof step.tool_info === 'object' && !Array.isArray(step.tool_info)
              ? (step.tool_info as Record<string, unknown>)
              : undefined;
          const toolName =
            typeof step.tool_name === 'string'
              ? step.tool_name
              : typeof toolInfo?.name === 'string'
                ? toolInfo.name
                : '';
          if (AGY_EDIT_TOOL_NAMES.has(toolName)) {
            const convId =
              typeof step.conversation_id === 'string'
                ? step.conversation_id
                : typeof record.conversation_id === 'string'
                  ? record.conversation_id
                  : 'agy';
            const stepIndex = typeof step.step_index === 'number' ? step.step_index : 0;
            const callId =
              typeof step.tool_call_id === 'string' && step.tool_call_id
                ? step.tool_call_id
                : `${convId}:step-${stepIndex}`;
            const state = typeof step.state === 'string' ? step.state.toUpperCase() : '';
            const params =
              toolInfo?.parameters && typeof toolInfo.parameters === 'object' && !Array.isArray(toolInfo.parameters)
                ? (toolInfo.parameters as Record<string, unknown>)
                : {};
            const rawPath =
              typeof params.Path === 'string'
                ? params.Path
                : typeof params.TargetFile === 'string'
                  ? params.TargetFile
                  : typeof params.file_path === 'string'
                    ? params.file_path
                    : typeof params.path === 'string'
                      ? params.path
                      : '';
            if (rawPath && (state === 'ACTIVE' || state === 'RUNNING' || state === 'PENDING')) {
              const absPath = path.isAbsolute(rawPath)
                ? rawPath
                : path.resolve(raw.workspacePath, rawPath);
              let beforeContent: string | null = null;
              try {
                if (existsSync(absPath)) {
                  beforeContent = readFileSync(absPath, 'utf8');
                }
              } catch {
                beforeContent = null;
              }
              const fallbackAfterContent =
                typeof params.Content === 'string'
                  ? params.Content
                  : typeof params.CodeContent === 'string'
                    ? params.CodeContent
                    : typeof params.content === 'string'
                      ? params.content
                      : undefined;
              pendingEdits.set(callId, {
                absPath,
                toolName,
                params,
                beforeContent,
                fallbackAfterContent,
              });
            } else if (state === 'DONE') {
              const pending = pendingEdits.get(callId);
              if (pending) {
                pendingEdits.delete(callId);
                let afterContent: string | undefined;
                try {
                  if (existsSync(pending.absPath)) {
                    afterContent = readFileSync(pending.absPath, 'utf8');
                  }
                } catch {
                  afterContent = undefined;
                }
                if (afterContent === undefined) {
                  afterContent = pending.fallbackAfterContent;
                }
                if (typeof afterContent === 'string') {
                  const mergedParams =
                    Object.keys(params).length > 0 ? { ...pending.params, ...params } : pending.params;
                  editChanges = [
                    buildAgyEditChange(
                      pending.absPath,
                      pending.toolName || toolName,
                      mergedParams,
                      pending.beforeContent,
                      afterContent,
                    ),
                  ];
                }
              }
            }
          }
        }
      }

      for (const ev of mapAgyRecord(record, editChanges)) {
        yield ev;
      }
    }
  }

  async abortSession(_session: ProtocolSession): Promise<void> {
    // AbortSignal passed to runHeadlessNdjson kills the child process.
  }

  async cleanupSession(session: ProtocolSession): Promise<void> {
    if (session.id) {
      this.sessions.delete(session.id);
    }
  }
}

function buildAgyEditChange(
  absPath: string,
  toolName: string,
  params: Record<string, unknown>,
  observedBefore: string | null,
  afterContent: string,
): AgyEditChange {
  if (observedBefore === null) {
    return {
      path: absPath,
      kind: 'add',
      beforeContent: '',
      afterContent,
      authoritative: false,
    };
  }

  if (observedBefore !== afterContent) {
    return {
      path: absPath,
      kind: 'update',
      beforeContent: observedBefore,
      afterContent,
      authoritative: false,
    };
  }

  // `observedBefore === afterContent`: the CLI already wrote the file before
  // the parent process consumed the `ACTIVE` stdout line. For in-place edit
  // tools (`edit_file`, `replace_file_content`, `multi_replace_file_content`),
  // reverse-apply the replacement hunk(s) against `afterContent` to recover
  // the exact pre-edit baseline deterministically.
  const recovered = tryReverseApplyAgyEdit(toolName, params, afterContent);
  if (recovered !== null && recovered !== afterContent) {
    return {
      path: absPath,
      kind: 'update',
      beforeContent: recovered,
      afterContent,
      authoritative: true,
    };
  }

  // Full-file write (`write_file`) where the `ACTIVE` disk read lost the race:
  // do NOT report `afterContent` as `beforeContent`. Leave `beforeContent`
  // empty and `authoritative: false` so `MessageStreamingHandler` consults
  // `FileSnapshotCache` for existing files and falls back to `''` for newly
  // created files.
  return {
    path: absPath,
    kind: 'update',
    beforeContent: '',
    afterContent,
    authoritative: false,
  };
}

function tryReverseApplyAgyEdit(
  toolName: string,
  params: Record<string, unknown>,
  afterContent: string,
): string | null {
  if (toolName === 'edit_file' || toolName === 'replace_file_content') {
    const target =
      typeof params.Target === 'string'
        ? params.Target
        : typeof params.TargetContent === 'string'
          ? params.TargetContent
          : undefined;
    const replacement =
      typeof params.Replacement === 'string'
        ? params.Replacement
        : typeof params.ReplacementContent === 'string'
          ? params.ReplacementContent
          : undefined;
    if (
      target !== undefined &&
      replacement !== undefined &&
      replacement.length > 0 &&
      target !== replacement &&
      afterContent.includes(replacement)
    ) {
      return afterContent.split(replacement).join(target);
    }
  }

  if (toolName === 'multi_replace_file_content' && Array.isArray(params.ReplacementChunks)) {
    let current = afterContent;
    let changed = false;
    for (let i = params.ReplacementChunks.length - 1; i >= 0; i--) {
      const chunk = params.ReplacementChunks[i];
      if (!chunk || typeof chunk !== 'object') continue;
      const rec = chunk as Record<string, unknown>;
      const target = typeof rec.TargetContent === 'string' ? rec.TargetContent : undefined;
      const replacement =
        typeof rec.ReplacementContent === 'string' ? rec.ReplacementContent : undefined;
      if (
        target !== undefined &&
        replacement !== undefined &&
        replacement.length > 0 &&
        target !== replacement &&
        current.includes(replacement)
      ) {
        current = current.replace(replacement, () => target);
        changed = true;
      }
    }
    if (changed) return current;
  }

  return null;
}

export function formatAgyPromptWithAttachments(
  message: ProtocolMessage,
  systemPrompt?: string,
  commandLineOverhead = 256,
): string {
  const maxPromptChars = Math.max(4_096, AGY_MAX_COMMAND_LINE_CHARS - commandLineOverhead);
  const trimmedSystem = systemPrompt?.trim() ?? '';
  const systemSection = trimmedSystem
    ? `<system_instructions>\n${trimmedSystem}\n</system_instructions>`
    : '';

  const attachments = message.attachments?.filter((a) => a.filepath) ?? [];
  const reservedChars =
    message.content.length + (systemSection ? systemSection.length + 2 : 0) + 256;
  let remainingInlineBudget = Math.min(
    AGY_INLINE_DOCUMENT_BUDGET,
    Math.max(0, maxPromptChars - reservedChars),
  );

  const inlinedDocuments: string[] = [];
  const referencedPaths: string[] = [];

  for (const attachment of attachments) {
    const filepath = attachment.filepath!;
    if (attachment.type === 'document') {
      try {
        const text = readFileSync(filepath, 'utf8');
        const wrapped = `<file name="${attachment.filename}">\n${text}\n</file>`;
        const byteLen = Buffer.byteLength(text, 'utf8');
        if (byteLen <= remainingInlineBudget && wrapped.length <= remainingInlineBudget) {
          remainingInlineBudget -= Math.max(byteLen, wrapped.length);
          inlinedDocuments.push(wrapped);
        } else {
          referencedPaths.push(`- ${attachment.filename}: ${filepath}`);
        }
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        inlinedDocuments.push(
          `<file name="${attachment.filename}" error="failed to read attachment: ${detail}" />`,
        );
      }
      continue;
    }

    try {
      readFileSync(filepath, { flag: 'r' });
      referencedPaths.push(`- ${attachment.filename}: ${filepath}`);
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      inlinedDocuments.push(
        `<file name="${attachment.filename}" error="failed to read attachment: ${detail}" />`,
      );
    }
  }

  const userSections = [message.content];
  if (inlinedDocuments.length > 0) userSections.push(inlinedDocuments.join('\n\n'));
  if (referencedPaths.length > 0) userSections.push(`Attached files:\n${referencedPaths.join('\n')}`);
  const userBody = userSections.join('\n\n');

  if (!systemSection) {
    return userBody.length <= maxPromptChars ? userBody : userBody.slice(0, maxPromptChars);
  }

  const combined = `${systemSection}\n\n${userBody}`;
  if (combined.length <= maxPromptChars) {
    return combined;
  }

  // Keep the user's message and attachment references intact first; fit as much
  // of the system instructions as the Windows command-line budget allows.
  const availableForSystem = maxPromptChars - userBody.length - 40;
  if (availableForSystem > 64) {
    const boundedSystem = `<system_instructions>\n${trimmedSystem.slice(0, availableForSystem)}\n</system_instructions>`;
    return `${boundedSystem}\n\n${userBody}`;
  }
  return userBody.length <= maxPromptChars ? userBody : userBody.slice(0, maxPromptChars);
}

function stripProviderPrefix(model: string | undefined): string | undefined {
  if (!model) return undefined;
  const idx = model.indexOf(':');
  return idx >= 0 ? model.slice(idx + 1) : model;
}
