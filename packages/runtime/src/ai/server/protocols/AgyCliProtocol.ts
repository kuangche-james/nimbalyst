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

import { randomBytes } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import * as fsPromises from 'node:fs/promises';
import * as os from 'node:os';
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
 * line conservatively at 28,000 characters on Windows; POSIX platforms permit
 * much larger single-argument lengths (128 KiB+).
 */
export const AGY_MAX_COMMAND_LINE_CHARS = 28_000;
export const AGY_POSIX_MAX_COMMAND_LINE_CHARS = 128_000;

export const AGY_NIMBALYST_MCP_KEY_PREFIX = 'nimbalyst:';

/**
 * Upper bound on how long one turn may hold `agyStartupGate` while waiting for
 * the CLI's first stdout record. A CLI that hangs before printing anything must
 * not block every other Gemini session in the process.
 */
export const AGY_STARTUP_GATE_TIMEOUT_MS = 15_000;

/**
 * Serialises the window between writing agy's global MCP config and the CLI
 * reading it. Only one turn may be inside that window at a time; the next
 * turn's write waits until the previous process has produced output.
 */
let agyStartupGate: Promise<void> = Promise.resolve();

function agyAbortError(): Error {
  const err = new Error('Antigravity turn was cancelled before the CLI started');
  err.name = 'AbortError';
  return err;
}

/**
 * Resolves with the release function once every earlier holder has released.
 * Rejects with an `AbortError` if `signal` is already aborted or aborts while
 * waiting; the slot this call reserved is released either way, so a cancelled
 * turn never blocks the turns queued behind it.
 */
function acquireAgyStartupGate(signal?: AbortSignal): Promise<() => void> {
  if (signal?.aborted) return Promise.reject(agyAbortError());
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  const previous = agyStartupGate;
  agyStartupGate = previous.then(() => held);
  return new Promise<() => void>((resolve, reject) => {
    const onAbort = () => {
      release();
      reject(agyAbortError());
    };
    signal?.addEventListener('abort', onAbort, { once: true });
    void previous.then(() => {
      signal?.removeEventListener('abort', onAbort);
      if (signal?.aborted) {
        onAbort();
        return;
      }
      resolve(release);
    });
  });
}

export interface AgyMcpServerEntry {
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
  headers?: Record<string, string>;
  [key: string]: unknown;
}

interface AgySessionRaw extends Record<string, unknown> {
  workspacePath: string;
  model?: string;
  systemPrompt?: string;
  mcpServers?: SessionOptions['mcpServers'];
}

export interface AgyCliProtocolDeps {
  runNdjson?: (options: HeadlessNdjsonRunOptions) => AsyncGenerator<HeadlessNdjsonItem>;
  mcpConfigPath?: string;
  syncMcpServers?: (servers: Record<string, AgyMcpServerEntry>) => Promise<void>;
}

export class AgyCliProtocol implements AgentProtocol {
  readonly platform = 'antigravity-gemini-agent';

  private executablePath = process.platform === 'win32' ? 'agy.exe' : 'agy';
  private processEnv: Record<string, string> | undefined;
  private readonly sessions = new Map<string, SessionOptions>();
  private readonly runNdjsonImpl: NonNullable<AgyCliProtocolDeps['runNdjson']>;
  private readonly mcpConfigPath?: string;
  private readonly syncMcpServersImpl: (servers: Record<string, AgyMcpServerEntry>) => Promise<void>;

  constructor(deps?: AgyCliProtocolDeps) {
    this.runNdjsonImpl = deps?.runNdjson ?? runHeadlessNdjson;
    this.mcpConfigPath = deps?.mcpConfigPath;
    this.syncMcpServersImpl =
      deps?.syncMcpServers ??
      ((servers) => syncAgyMcpConfigFile(servers, this.mcpConfigPath));
  }

  setAgyPath(executablePath: string): void {
    this.executablePath = executablePath;
  }

  setProcessEnv(env: Record<string, string>): void {
    this.processEnv = env;
  }

  async createSession(options: SessionOptions): Promise<ProtocolSession> {
    // MCP servers are written to agy's config file in `sendMessage`, right
    // before the process spawns, under `agyStartupGate`. Writing them here
    // would let a second session's `createSession` overwrite the file before
    // this session's CLI ever reads it.
    const normalizedMcp = normalizeAgyMcpServers(options.mcpServers);
    const raw: AgySessionRaw = {
      workspacePath: options.workspacePath,
      model: options.model,
      systemPrompt: options.systemPrompt,
      mcpServers: options.mcpServers,
    };
    return {
      id: '',
      platform: this.platform,
      deliveredMcpServerCount: Object.keys(normalizedMcp).length,
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
    const normalizedMcp = normalizeAgyMcpServers(merged.mcpServers);
    const raw: AgySessionRaw = {
      workspacePath: merged.workspacePath,
      model: merged.model,
      systemPrompt: merged.systemPrompt,
      mcpServers: merged.mcpServers,
    };
    return {
      id: sessionId,
      platform: this.platform,
      deliveredMcpServerCount: Object.keys(normalizedMcp).length,
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

    // `--print` takes an optional prompt value, so it must come last with the
    // prompt attached (`--print=<prompt>`). Written as a bare `--print` ahead
    // of the other flags, agy consumed `--output-format` as the prompt and
    // ignored the real one after `--`, exiting with code 2 on a real Windows
    // install (#1653). The attached form also keeps a prompt that starts with
    // `-` from being read as a flag.
    const args: string[] = [];
    if (session.id) {
      args.push('--conversation', session.id);
    }
    args.push('--output-format', 'stream-json');
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
      args.reduce((sum, arg) => sum + arg.length + 3, 6) +
      '--print='.length;
    args.push(
      `--print=${formatAgyPromptWithAttachments(message, raw.systemPrompt, commandLineOverhead)}`,
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

    // agy reads `~/.gemini/config/mcp_config.json` once, at process startup,
    // and has no per-invocation config flag. The file is global, and
    // `McpConfigService` embeds the workspace and session id in the server
    // URLs, so two sessions starting together would otherwise race: both
    // writes land before either CLI reads, and both processes get the second
    // session's endpoints. Hold the gate from the write until this process
    // proves it has started (its first stdout record) or the timeout lapses.
    //
    // A turn cancelled while it waits here must not write its config or
    // spawn: `runHeadlessNdjson` only listens for a future abort, so a child
    // started with an already-aborted signal would run the cancelled prompt
    // until the first event is consumed.
    const releaseGate = await acquireAgyStartupGate(message.abortSignal);
    let gateHeld = true;
    const releaseStartupGate = () => {
      if (!gateHeld) return;
      gateHeld = false;
      releaseGate();
    };
    const gateTimer = setTimeout(releaseStartupGate, AGY_STARTUP_GATE_TIMEOUT_MS);
    gateTimer.unref?.();

    try {
      await this.syncMcpServersImpl(normalizeAgyMcpServers(raw.mcpServers));
      if (message.abortSignal?.aborted) {
        // Cancelled during the config write: do not start the process.
        throw agyAbortError();
      }
    } catch (error) {
      releaseStartupGate();
      clearTimeout(gateTimer);
      throw error;
    }

    try {
      for await (const item of this.runNdjsonImpl({
        command: this.executablePath,
        args,
        cwd: raw.workspacePath,
        env: this.processEnv,
        abortSignal: message.abortSignal,
      })) {
        // Any output at all means the CLI is up and has loaded its config.
        releaseStartupGate();
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
              mcpServers: raw.mcpServers,
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
              mcpServers: raw.mcpServers,
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
    } finally {
      releaseStartupGate();
      clearTimeout(gateTimer);
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
  // recover the pre-edit baseline ONLY when the replacement hunk is
  // unambiguous (single match or bounded by an unchanged-line-count range).
  // Always keep `authoritative: false` so `MessageStreamingHandler` consults
  // `FileSnapshotCache` first and only falls back to `beforeContent` on a
  // cache miss.
  const recovered = tryReverseApplyAgyEdit(toolName, params, afterContent);
  if (recovered !== null && recovered !== afterContent) {
    return {
      path: absPath,
      kind: 'update',
      beforeContent: recovered,
      afterContent,
      authoritative: false,
    };
  }

  // Full-file write (`write_file`) or ambiguous replacement where the `ACTIVE`
  // disk read lost the race: the pre-edit content is unknown. Do NOT report
  // `afterContent` as `beforeContent`, and do NOT report `''` either: the
  // host writes whatever it is given as the pre-edit tag with
  // `replaceSpeculative`, and `FileSnapshotCache` cannot always fill the gap
  // (it skips gitignored and never-snapshotted paths), so an empty string
  // would render the whole file as added and make Reject empty it. Leaving
  // `beforeContent` undefined means no `pre_edit_snapshot` entry is emitted
  // for this path, and the filesystem watcher's own tag stands.
  return {
    path: absPath,
    kind: 'update',
    afterContent,
    authoritative: false,
  };
}

function countOccurrences(haystack: string, needle: string): number {
  if (!needle) return 0;
  let count = 0;
  let idx = 0;
  while (true) {
    const found = haystack.indexOf(needle, idx);
    if (found < 0) break;
    count++;
    idx = found + needle.length;
  }
  return count;
}

function tryReverseApplySingleAgyHunk(
  text: string,
  target: string,
  replacement: string,
  startLine?: unknown,
  endLine?: unknown,
): string | null {
  if (!replacement || target === replacement || target.includes(replacement)) {
    return null;
  }

  if (
    typeof startLine === 'number' &&
    typeof endLine === 'number' &&
    Number.isInteger(startLine) &&
    Number.isInteger(endLine) &&
    startLine >= 1 &&
    endLine >= startLine
  ) {
    const targetLines = target.split('\n').length;
    const replacementLines = replacement.split('\n').length;
    if (targetLines === replacementLines) {
      const lines = text.split('\n');
      if (endLine <= lines.length) {
        const region = lines.slice(startLine - 1, endLine).join('\n');
        if (countOccurrences(region, replacement) === 1) {
          const regionIdx = region.indexOf(replacement);
          const restoredRegion =
            region.slice(0, regionIdx) + target + region.slice(regionIdx + replacement.length);
          return [
            ...lines.slice(0, startLine - 1),
            ...restoredRegion.split('\n'),
            ...lines.slice(endLine),
          ].join('\n');
        }
      }
    }
    if (countOccurrences(text, replacement) !== 1) {
      return null;
    }
  }

  if (countOccurrences(text, replacement) !== 1) {
    return null;
  }

  const idx = text.indexOf(replacement);
  return text.slice(0, idx) + target + text.slice(idx + replacement.length);
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
    if (target !== undefined && replacement !== undefined) {
      return tryReverseApplySingleAgyHunk(
        afterContent,
        target,
        replacement,
        params.StartLine,
        params.EndLine,
      );
    }
  }

  if (toolName === 'multi_replace_file_content' && Array.isArray(params.ReplacementChunks)) {
    let current = afterContent;
    let changed = false;
    for (let i = params.ReplacementChunks.length - 1; i >= 0; i--) {
      const chunk = params.ReplacementChunks[i];
      if (!chunk || typeof chunk !== 'object') return null;
      const rec = chunk as Record<string, unknown>;
      const target = typeof rec.TargetContent === 'string' ? rec.TargetContent : undefined;
      const replacement =
        typeof rec.ReplacementContent === 'string' ? rec.ReplacementContent : undefined;
      if (target === undefined || replacement === undefined) {
        return null;
      }
      const next = tryReverseApplySingleAgyHunk(
        current,
        target,
        replacement,
        rec.StartLine,
        rec.EndLine,
      );
      if (next === null) {
        return null;
      }
      current = next;
      changed = true;
    }
    if (changed) return current;
  }

  return null;
}

export function formatAgyPromptWithAttachments(
  message: ProtocolMessage,
  systemPrompt?: string,
  commandLineOverhead = 256,
  platform: NodeJS.Platform = process.platform,
): string {
  const maxCommandLineChars =
    platform === 'win32' ? AGY_MAX_COMMAND_LINE_CHARS : AGY_POSIX_MAX_COMMAND_LINE_CHARS;
  const maxPromptChars = Math.max(4_096, maxCommandLineChars - commandLineOverhead);
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

  const fullPrompt = systemSection ? `${systemSection}\n\n${userBody}` : userBody;
  if (fullPrompt.length > maxPromptChars) {
    const err = new Error(
      `spawn E2BIG: prompt length (${fullPrompt.length}) exceeds command-line limit (${maxPromptChars})`,
    ) as NodeJS.ErrnoException;
    err.code = 'E2BIG';
    throw err;
  }

  return fullPrompt;
}

export function normalizeAgyMcpServers(
  mcpServers?: Record<string, unknown>,
): Record<string, AgyMcpServerEntry> {
  const normalized: Record<string, AgyMcpServerEntry> = {};
  if (!mcpServers || typeof mcpServers !== 'object') return normalized;

  for (const [name, raw] of Object.entries(mcpServers)) {
    if (!name || !raw || typeof raw !== 'object') continue;
    const rec = raw as Record<string, unknown>;
    const url =
      typeof rec.url === 'string' && rec.url.trim()
        ? rec.url.trim()
        : typeof rec.serverUrl === 'string' && rec.serverUrl.trim()
          ? rec.serverUrl.trim()
          : undefined;
    if (url) {
      const entry: AgyMcpServerEntry = { url };
      if (rec.headers && typeof rec.headers === 'object' && !Array.isArray(rec.headers)) {
        const headers: Record<string, string> = {};
        for (const [k, v] of Object.entries(rec.headers as Record<string, unknown>)) {
          if (typeof v === 'string') headers[k] = v;
        }
        if (Object.keys(headers).length > 0) entry.headers = headers;
      }
      normalized[name] = entry;
      continue;
    }

    if (typeof rec.command === 'string' && rec.command.trim()) {
      const entry: AgyMcpServerEntry = { command: rec.command.trim() };
      if (Array.isArray(rec.args)) {
        entry.args = rec.args.filter((a): a is string => typeof a === 'string');
      }
      if (rec.env && typeof rec.env === 'object' && !Array.isArray(rec.env)) {
        const env: Record<string, string> = {};
        for (const [k, v] of Object.entries(rec.env as Record<string, unknown>)) {
          if (typeof v === 'string') env[k] = v;
        }
        if (Object.keys(env).length > 0) entry.env = env;
      }
      normalized[name] = entry;
    }
  }

  return normalized;
}

export function resolveDefaultAgyMcpConfigPath(homedir = os.homedir()): string {
  return path.join(homedir, '.gemini', 'config', 'mcp_config.json');
}

let agyMcpSyncQueue: Promise<unknown> = Promise.resolve();

export function syncAgyMcpConfigFile(
  servers: Record<string, AgyMcpServerEntry>,
  configPath = resolveDefaultAgyMcpConfigPath(),
): Promise<void> {
  const task = () => syncAgyMcpConfigFileExclusive(servers, configPath);
  const next = agyMcpSyncQueue.then(task, task);
  agyMcpSyncQueue = next.catch(() => undefined);
  return next;
}

async function syncAgyMcpConfigFileExclusive(
  servers: Record<string, AgyMcpServerEntry>,
  configPath: string,
): Promise<void> {
  let existing: { mcpServers?: Record<string, AgyMcpServerEntry>; [key: string]: unknown } | null =
    null;
  let existingMode: number | null = null;

  try {
    const raw = await fsPromises.readFile(configPath, 'utf8');
    existing = JSON.parse(raw) as {
      mcpServers?: Record<string, AgyMcpServerEntry>;
      [key: string]: unknown;
    };
    existingMode = (await fsPromises.stat(configPath)).mode & 0o777;
  } catch (error) {
    const code = (error as NodeJS.ErrnoException)?.code;
    if (code !== 'ENOENT') {
      return;
    }
  }

  const previousServers = existing?.mcpServers ?? {};
  const hasNimbalystEntries = Object.keys(previousServers).some((key) =>
    key.startsWith(AGY_NIMBALYST_MCP_KEY_PREFIX),
  );

  if (Object.keys(servers).length === 0 && !hasNimbalystEntries) {
    return;
  }

  const nextServers: Record<string, AgyMcpServerEntry> = {};
  for (const [name, entry] of Object.entries(previousServers)) {
    if (!name.startsWith(AGY_NIMBALYST_MCP_KEY_PREFIX)) {
      nextServers[name] = entry;
    }
  }
  for (const [name, entry] of Object.entries(servers)) {
    const prefixedName = name.startsWith(AGY_NIMBALYST_MCP_KEY_PREFIX)
      ? name
      : `${AGY_NIMBALYST_MCP_KEY_PREFIX}${name}`;
    nextServers[prefixedName] = entry;
  }

  const merged = { ...(existing ?? {}), mcpServers: nextServers };
  const serialized = `${JSON.stringify(merged, null, 2)}\n`;

  try {
    await fsPromises.mkdir(path.dirname(configPath), { recursive: true });
    const tempPath = path.join(
      path.dirname(configPath),
      `${path.basename(configPath)}.${randomBytes(6).toString('hex')}.tmp`,
    );
    try {
      await fsPromises.writeFile(tempPath, serialized, { encoding: 'utf8', mode: 0o600 });
      if (existingMode !== null) {
        await fsPromises.chmod(tempPath, existingMode).catch(() => undefined);
      }
      await fsPromises.rename(tempPath, configPath);
    } catch (writeError) {
      await fsPromises.rm(tempPath, { force: true }).catch(() => undefined);
      throw writeError;
    }
  } catch {
    // Non-fatal: a turn without MCP servers is better than a turn that fails to start.
  }
}

function stripProviderPrefix(model: string | undefined): string | undefined {
  if (!model) return undefined;
  const idx = model.indexOf(':');
  return idx >= 0 ? model.slice(idx + 1) : model;
}
