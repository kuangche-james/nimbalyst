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

export const AGY_INLINE_DOCUMENT_BUDGET = 64 * 1024;

interface AgySessionRaw extends Record<string, unknown> {
  workspacePath: string;
  model?: string;
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
    };
    return {
      id: '',
      platform: this.platform,
      deliveredMcpServerCount: 0,
      raw,
    };
  }

  async resumeSession(sessionId: string, options?: SessionOptions): Promise<ProtocolSession> {
    const merged = options ?? this.sessions.get(sessionId) ?? { workspacePath: process.cwd() };
    if (sessionId) {
      this.sessions.set(sessionId, merged);
    }
    const raw: AgySessionRaw = {
      workspacePath: merged.workspacePath,
      model: merged.model,
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
    args.push('--', formatAgyPromptWithAttachments(message));

    let sawResult = false;
    let stderrTail = '';
    const pendingEdits = new Map<
      string,
      { absPath: string; beforeContent: string | null; fallbackAfterContent?: string }
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
          this.sessions.set(convId, { workspacePath: raw.workspacePath, model: raw.model });
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
        sawResult = true;
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
          this.sessions.set(convId, { workspacePath: raw.workspacePath, model: raw.model });
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
              pendingEdits.set(callId, { absPath, beforeContent, fallbackAfterContent });
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
                  editChanges = [
                    {
                      path: pending.absPath,
                      kind: pending.beforeContent === null ? 'add' : 'update',
                      beforeContent: pending.beforeContent ?? '',
                      afterContent,
                    },
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

function formatAgyPromptWithAttachments(message: ProtocolMessage): string {
  const attachments = message.attachments?.filter((a) => a.filepath);
  if (!attachments?.length) return message.content;

  let remainingInlineBudget = AGY_INLINE_DOCUMENT_BUDGET;
  const inlinedDocuments: string[] = [];
  const referencedPaths: string[] = [];

  for (const attachment of attachments) {
    const filepath = attachment.filepath!;
    if (attachment.type === 'document') {
      try {
        const text = readFileSync(filepath, 'utf8');
        if (Buffer.byteLength(text, 'utf8') <= remainingInlineBudget) {
          remainingInlineBudget -= Buffer.byteLength(text, 'utf8');
          inlinedDocuments.push(`<file name="${attachment.filename}">\n${text}\n</file>`);
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

  const sections = [message.content];
  if (inlinedDocuments.length > 0) sections.push(inlinedDocuments.join('\n\n'));
  if (referencedPaths.length > 0) sections.push(`Attached files:\n${referencedPaths.join('\n')}`);
  return sections.join('\n\n');
}

function stripProviderPrefix(model: string | undefined): string | undefined {
  if (!model) return undefined;
  const idx = model.indexOf(':');
  return idx >= 0 ? model.slice(idx + 1) : model;
}
