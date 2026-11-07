/**
 * Gemini agent provider, backed by the official Antigravity CLI (`agy`).
 *
 * Subclasses `HeadlessCliAgentProvider` and drives `agy --print --output-format stream-json`
 * via `AgyCliProtocol` — the same headless per-turn NDJSON architecture used by
 * `CursorAgentProvider`.
 *
 * Also retains the legacy `AntigravityServerManager` / `AntigravityToolLoopProtocol`
 * fallback when a host tool executor is explicitly injected for unit tests.
 */

import { existsSync } from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  HeadlessCliAgentProvider,
  type HeadlessCliAgentDescriptor,
  type HeadlessCliEnvironmentLoaders,
  buildHeadlessChildEnvironment,
} from './HeadlessCliAgentProvider';
import { AgyCliProtocol } from '../protocols/AgyCliProtocol';
import type { ProtocolEvent, ToolResult } from '../protocols/ProtocolInterface';
import { appendSessionDirective, type MetaAgentWorkflowPreset } from '../../prompt';
import {
  AntigravityServerManager,
  AntigravityVersionGateError,
} from './geminiAntigravity/AntigravityServerManager';
import { AntigravityToolLoopProtocol } from './geminiAntigravity/AntigravityToolLoopProtocol';
import {
  DEFAULT_GEMINI_MODEL_KEY,
  bareGeminiModelKey,
  discoverGeminiModels,
  parseAgyModelList,
} from './geminiAntigravity/geminiAntigravityModels';
import type { MCPServerConfig } from '../../../types/MCPServerConfig';
import type {
  AgentToolDefinition,
  AIModel,
  AIProviderType,
  ChatAttachment,
  DocumentContext,
  Message,
  ProviderConfig,
  StreamChunk,
} from '../types';

export const GEMINI_ANTIGRAVITY_PROVIDER: AIProviderType = 'antigravity-gemini-agent';

export interface GeminiToolExecutionResult {
  text: string;
  fileWrite?: {
    absPath: string;
    beforeContent: string | null;
    afterContent: string;
  };
}

export interface GeminiToolExecutorArgs {
  sessionId: string;
  workspacePath?: string;
  name: string;
  args: Record<string, unknown>;
}

export type GeminiToolExecutor = (args: GeminiToolExecutorArgs) => Promise<GeminiToolExecutionResult>;

export interface GeminiServerConfig {
  overrideIdeVersion?: string;
  spawnPortCandidates?: number[];
}

interface SessionState {
  sessionId: string;
  workspacePath?: string;
  modelKey: string;
  toolLoop: AntigravityToolLoopProtocol;
  abortController: AbortController | null;
}

const RUN_COMMAND_TIMEOUT_MS = 120_000;
const RUN_COMMAND_MAX_BUFFER = 4 * 1024 * 1024;
const RUN_COMMAND_MAX_OUTPUT = 48_000;

function clampCommandOutput(text: string): string {
  return text.length <= RUN_COMMAND_MAX_OUTPUT
    ? text
    : `${text.slice(0, RUN_COMMAND_MAX_OUTPUT)}\n\n[output truncated at ${RUN_COMMAND_MAX_OUTPUT} characters]`;
}

export function formatPriorMessagesForAgySeed(
  messages: Message[] | undefined,
  currentMessage: string,
): string | undefined {
  if (!messages || messages.length === 0) return undefined;
  const prior = [...messages];
  const last = prior[prior.length - 1];
  if (
    last?.role === 'user' &&
    typeof last.content === 'string' &&
    last.content.trim() === currentMessage.trim()
  ) {
    prior.pop();
  }

  const formatted = prior
    .filter(
      (m) =>
        (m.role === 'user' || m.role === 'assistant') &&
        typeof m.content === 'string' &&
        m.content.trim().length > 0,
    )
    .map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}:\n${m.content.trim()}`);

  if (formatted.length === 0) return undefined;
  return `<conversation_history>\nThe following is the prior conversation history from this session before continuing:\n\n${formatted.join('\n\n')}\n</conversation_history>`;
}

const DESCRIPTOR: HeadlessCliAgentDescriptor = {
  providerName: GEMINI_ANTIGRAVITY_PROVIDER,
  displayName: 'Antigravity',
  description: 'Gemini via Google Antigravity CLI (agy), driven headlessly over stream JSON',
  executableName: process.platform === 'win32' ? 'agy.exe' : 'agy',
  homeRelativeInstallPaths: [
    'AppData/Local/agy/bin/agy.exe',
    'AppData/Local/agy/agy.exe',
    '.local/bin/agy',
    '.gemini/antigravity-cli/bin/agy',
  ],
  notInstalledMessage:
    'Antigravity CLI (agy) is not installed or not in PATH. Install Antigravity from https://antigravity.google/download, ensure `agy` is on your PATH, and try again.',
  notLoggedInMessage:
    'Antigravity is not logged in. Run `agy` or open Antigravity to sign in to your Google account, then try again.',
  supportsToolPermissions: false,
  permissionModeMessage:
    'Gemini runs through the Antigravity CLI in headless mode, which carries out file edits '
    + 'and shell commands without asking first, so there is no per-tool approval for '
    + 'Nimbalyst to intercept. To use Gemini in this project, set its permission mode to '
    + '"Allow edits only", "Agent-verified", or "Allow everything" (click the shield icon in '
    + 'the session header, or open Project Settings > Permissions), then send the message again.',
};

export class GeminiAntigravityProvider extends HeadlessCliAgentProvider {
  static readonly DEFAULT_MODEL = `${GEMINI_ANTIGRAVITY_PROVIDER}:${DEFAULT_GEMINI_MODEL_KEY}`;

  static readonly NOT_INSTALLED_MESSAGE = DESCRIPTOR.notInstalledMessage;
  static readonly NOT_SIGNED_IN_MESSAGE = DESCRIPTOR.notLoggedInMessage;
  static readonly VERSION_GATE_MESSAGE =
    'Antigravity rejected this build as out of date. Update Antigravity to its '
    + 'latest version and try again. (If you need to keep the current version, '
    + 'set a newer `overrideIdeVersion` on the Gemini provider in ai-settings.)';

  protected readonly descriptor = DESCRIPTOR;
  protected readonly protocol: AgyCliProtocol;

  private static mcpConfigLoader:
    | ((workspacePath?: string) => Promise<Record<string, MCPServerConfig>>)
    | null = null;
  private static shellEnvironmentLoader: (() => Record<string, string> | null) | null = null;
  private static enhancedPathLoader: (() => string) | null = null;
  private static agyPathLoader: (() => string | null) | null = null;

  private static toolExecutor: GeminiToolExecutor | null = null;
  private static serverConfigLoader: (() => GeminiServerConfig) | null = null;

  private readonly sessionStates = new Map<string, SessionState>();
  private readonly server: AntigravityServerManager;
  private readonly explicitServer: boolean;

  constructor(deps?: { protocol?: AgyCliProtocol; server?: AntigravityServerManager }) {
    super(GeminiAntigravityProvider.loaders());
    this.protocol = deps?.protocol ?? new AgyCliProtocol();
    this.server = deps?.server ?? AntigravityServerManager.shared();
    this.explicitServer = Boolean(deps?.server);
  }

  private static loaders(): HeadlessCliEnvironmentLoaders {
    return {
      mcpConfigLoader: GeminiAntigravityProvider.mcpConfigLoader,
      shellEnvironmentLoader: GeminiAntigravityProvider.shellEnvironmentLoader,
      enhancedPathLoader: GeminiAntigravityProvider.enhancedPathLoader,
      executablePathLoader: GeminiAntigravityProvider.agyPathLoader,
    };
  }

  protected getLoaders(): HeadlessCliEnvironmentLoaders {
    return GeminiAntigravityProvider.loaders();
  }

  protected configureProtocol(executablePath: string, env: Record<string, string>): void {
    this.protocol.setAgyPath(executablePath);
    this.protocol.setProcessEnv(env);
  }

  protected getDefaultModelId(): string {
    return GeminiAntigravityProvider.DEFAULT_MODEL;
  }

  static getDefaultModel(): string {
    return GeminiAntigravityProvider.DEFAULT_MODEL;
  }

  override async initialize(config: ProviderConfig): Promise<void> {
    await super.initialize(config);
    const serverConfig = GeminiAntigravityProvider.serverConfigLoader?.();
    if (serverConfig) this.server.configure(serverConfig);
  }

  override abort(): void {
    for (const state of this.sessionStates.values()) {
      state.abortController?.abort();
      state.toolLoop.abort();
    }
    super.abort();
  }

  override destroy(): void {
    this.abort();
    this.sessionStates.clear();
    super.destroy();
  }

  private turnSystemPromptOverride: string | undefined;
  private turnHistorySeed: string | undefined;

  /**
   * Emit `pre_edit_snapshot` / `post_edit_snapshot` chunks when `AgyCliProtocol`
   * attaches full-file `changes` onto an edit tool result, while keeping
   * `getFileChangeFidelity() === 'tool-args'` so the filesystem watcher remains
   * active for shell-driven deletes and moves.
   *
   * `authoritative` is left `false` for `AgyCliProtocol` snapshots so
   * `MessageStreamingHandler` always consults `FileSnapshotCache` first and
   * falls back to `beforeContent` (disk read or unambiguous reverse-applied
   * hunk) on cache misses.
   */
  protected override *buildEditSnapshots(event: ProtocolEvent): Generator<StreamChunk> {
    if (event.type !== 'tool_result') return;

    const result = event.toolResult?.result;
    if (!result || typeof result === 'string') return;
    const changes = (result as ToolResult).changes;
    if (!Array.isArray(changes) || changes.length === 0) return;

    const toolUseId = event.toolResult?.id ?? `${this.getProviderName()}-edit-${changes.length}`;
    const preEntries: Array<{ path: string; content: string; kind?: string }> = [];
    const postEntries: Array<{ path: string; content: string; kind?: string }> = [];
    let allAuthoritative = true;

    for (const raw of changes) {
      const change = raw as {
        path?: unknown;
        kind?: unknown;
        beforeContent?: unknown;
        afterContent?: unknown;
        authoritative?: unknown;
      };
      if (typeof change.path !== 'string' || !change.path) continue;
      const kind = typeof change.kind === 'string' ? change.kind : 'update';

      if (typeof change.beforeContent === 'string') {
        preEntries.push({ path: change.path, content: change.beforeContent, kind });
        if (change.authoritative !== true) {
          allAuthoritative = false;
        }
      }
      if (kind !== 'delete' && typeof change.afterContent === 'string') {
        postEntries.push({ path: change.path, content: change.afterContent, kind });
      }
    }

    if (preEntries.length > 0) {
      yield {
        type: 'pre_edit_snapshot',
        preEditSnapshot: {
          toolUseId,
          entries: preEntries,
          authoritative: allAuthoritative,
        },
      };
    }
    if (postEntries.length > 0) {
      yield {
        type: 'post_edit_snapshot',
        postEditSnapshot: {
          toolUseId,
          entries: postEntries,
        },
      };
    }
  }

  protected override buildSystemPrompt(
    documentContext?: DocumentContext,
    sessionDirective?: string,
    hasOutOfBandNaming: boolean = false,
    isMetaAgent: boolean = false,
    workflowPreset: MetaAgentWorkflowPreset = 'default',
  ): string {
    const base = super.buildSystemPrompt(
      documentContext,
      sessionDirective,
      hasOutOfBandNaming,
      isMetaAgent,
      workflowPreset,
    );
    const parts = [base, this.turnSystemPromptOverride?.trim(), this.turnHistorySeed].filter(
      (part): part is string => Boolean(part && part.length > 0),
    );
    return parts.join('\n\n');
  }

  override async *sendMessage(
    message: string,
    documentContext?: DocumentContext,
    sessionId?: string,
    messages?: Message[],
    workspacePath?: string,
    attachments?: ChatAttachment[],
    tools?: AgentToolDefinition[],
    systemPrompt?: string,
  ): AsyncIterableIterator<StreamChunk> {
    if (GeminiAntigravityProvider.toolExecutor !== null || this.explicitServer) {
      yield* this.sendMessageViaServerManager(
        message,
        documentContext,
        sessionId,
        messages,
        workspacePath,
        attachments,
        tools,
        systemPrompt,
      );
      return;
    }

    const hasNativeSession = Boolean(sessionId && this.sessions.getSessionId(sessionId));
    this.turnSystemPromptOverride = systemPrompt;
    this.turnHistorySeed =
      !hasNativeSession && messages && messages.length > 0
        ? formatPriorMessagesForAgySeed(messages, message)
        : undefined;
    try {
      yield* super.sendMessage(
        message,
        documentContext,
        sessionId,
        messages,
        workspacePath,
        attachments,
      );
    } finally {
      this.turnSystemPromptOverride = undefined;
      this.turnHistorySeed = undefined;
    }
  }

  private async *sendMessageViaServerManager(
    message: string,
    documentContext?: DocumentContext,
    sessionId?: string,
    messages?: Message[],
    workspacePath?: string,
    _attachments?: ChatAttachment[],
    tools?: AgentToolDefinition[],
    systemPrompt?: string,
  ): AsyncIterableIterator<StreamChunk> {
    if (!sessionId) {
      yield { type: 'error', error: 'Gemini requires a session id.' };
      return;
    }

    const turnSystemPrompt = appendSessionDirective(
      systemPrompt ?? '',
      await this.getSessionDirective(sessionId),
    );

    const state = this.resolveSession(sessionId, workspacePath, documentContext);
    const abortController = new AbortController();
    state.abortController = abortController;
    this.abortController = abortController;

    if (messages && messages.length > 0) {
      const prior = [...messages];
      const last = prior[prior.length - 1];
      if (
        last?.role === 'user' &&
        typeof last.content === 'string' &&
        last.content.trim() === message.trim()
      ) {
        prior.pop();
      }
      state.toolLoop.seedHistory(prior);
    }

    await this.logAgentMessageBestEffort(sessionId, 'input', message, {
      metadata: {
        role: 'user',
        timestamp: Date.now(),
        model: state.modelKey,
        documentContext,
      },
    });

    let finalText = '';
    let sawText = false;
    let toolCallSeq = 0;
    const pendingCalls = new Map<
      string,
      { id: string; name: string; args: Record<string, unknown> }
    >();

    try {
      for await (const step of state.toolLoop.run(
        message,
        turnSystemPrompt,
        (tools ?? []) as Array<{ type: 'function'; function: { name: string } }>,
        (name, args) => this.executeTool(state, name, args),
        undefined,
        abortController.signal,
      )) {
        if (abortController.signal.aborted) break;

        if (step.type === 'tool_call') {
          const id = `agy-${Date.now()}-${toolCallSeq++}`;
          pendingCalls.set(step.name, { id, name: step.name, args: step.args });
          yield { type: 'tool_call', toolCall: { id, name: step.name, arguments: step.args } };
        } else if (step.type === 'tool_result') {
          const pending = pendingCalls.get(step.name);
          if (!pending) {
            this.pendingEditSnapshots.length = 0;
            continue;
          }
          for (const snapshot of this.drainEditSnapshots(pending.id)) {
            yield snapshot;
          }
          await this.logAgentMessageBestEffort(
            sessionId,
            'output',
            JSON.stringify({ name: step.name, result: step.result, args: pending.args }),
            { metadata: { role: 'tool', timestamp: Date.now(), toolUseId: pending.id } },
          );
          yield {
            type: 'tool_call',
            toolCall: {
              id: pending.id,
              name: pending.name,
              arguments: pending.args,
              result: step.result,
            },
          };
          pendingCalls.delete(step.name);
        } else if (step.type === 'text') {
          finalText = step.content;
          sawText = true;
          yield {
            type: 'text',
            content: finalText.trim().length === 0 ? '(model returned no text)' : finalText,
          };
        } else if (step.type === 'complete') {
          const persisted =
            finalText.trim().length === 0
              ? sawText
                ? '(model returned no text)'
                : '(no model response)'
              : finalText;
          await this.logAgentMessageBestEffort(sessionId, 'output', persisted, {
            metadata: { role: 'assistant', timestamp: Date.now(), model: state.modelKey },
          });
          yield { type: 'complete', content: persisted, isComplete: true };
        }
      }
    } catch (error) {
      if (abortController.signal.aborted) return;
      const errorMessage = error instanceof Error ? error.message : String(error);
      if (/not found at |Install Antigravity/i.test(errorMessage)) {
        yield { type: 'error', error: GeminiAntigravityProvider.NOT_INSTALLED_MESSAGE };
      } else if (error instanceof AntigravityVersionGateError) {
        yield { type: 'error', error: GeminiAntigravityProvider.VERSION_GATE_MESSAGE };
      } else if (/not logged in|unauthorized|forbidden|HTTP 401|HTTP 403/i.test(errorMessage)) {
        yield {
          type: 'error',
          error: GeminiAntigravityProvider.NOT_SIGNED_IN_MESSAGE,
          isAuthError: true,
        };
      } else {
        yield { type: 'error', error: errorMessage };
      }
    } finally {
      state.abortController = null;
      if (this.abortController === abortController) this.abortController = null;
      this.pendingEditSnapshots.length = 0;
    }
  }

  private readonly pendingEditSnapshots: Array<{
    beforeContent: string | null;
    afterContent: string;
    absPath: string;
  }> = [];

  private async executeTool(
    state: SessionState,
    name: string,
    args: Record<string, unknown>,
  ): Promise<string> {
    if (name === 'run_command') {
      return this.runCommand(state.workspacePath, args);
    }
    const executor = GeminiAntigravityProvider.toolExecutor;
    if (!executor) {
      return `Error: "${name}" cannot run -- no tool executor is installed for this provider.`;
    }
    const result = await executor({
      sessionId: state.sessionId,
      workspacePath: state.workspacePath,
      name,
      args,
    });
    if (result.fileWrite) this.pendingEditSnapshots.push(result.fileWrite);
    return result.text;
  }

  private *drainEditSnapshots(toolUseId: string): Generator<StreamChunk> {
    if (this.pendingEditSnapshots.length === 0) return;
    const writes = this.pendingEditSnapshots.splice(0, this.pendingEditSnapshots.length);
    yield {
      type: 'pre_edit_snapshot',
      preEditSnapshot: {
        toolUseId,
        entries: writes.map((w) => ({
          path: w.absPath,
          content: w.beforeContent ?? '',
          kind: w.beforeContent === null ? 'add' : 'update',
        })),
        authoritative: true,
      },
    };
    yield {
      type: 'post_edit_snapshot',
      postEditSnapshot: {
        toolUseId,
        entries: writes.map((w) => ({
          path: w.absPath,
          content: w.afterContent,
          kind: w.beforeContent === null ? 'add' : 'update',
        })),
      },
    };
  }

  private async runCommand(
    workspacePath: string | undefined,
    args: Record<string, unknown>,
  ): Promise<string> {
    const command = typeof args.command === 'string' ? args.command.trim() : '';
    if (!command) return 'Error: run_command requires a non-empty "command" string.';
    if (!workspacePath) {
      return 'Error: run_command needs an open workspace; none is bound to this session.';
    }
    const { exec } = await import('child_process');
    const { promisify } = await import('util');
    const execAsync = promisify(exec);
    try {
      const { stdout, stderr } = await execAsync(command, {
        cwd: workspacePath,
        timeout: RUN_COMMAND_TIMEOUT_MS,
        maxBuffer: RUN_COMMAND_MAX_BUFFER,
        windowsHide: true,
      });
      const body =
        [stdout ? `stdout:\n${stdout}` : '', stderr ? `stderr:\n${stderr}` : '']
          .filter(Boolean)
          .join('\n\n') || '(no output)';
      return clampCommandOutput(`$ ${command}\nexit code: 0\n\n${body}`);
    } catch (err: unknown) {
      const e = err as {
        code?: number;
        killed?: boolean;
        signal?: string;
        stdout?: string;
        stderr?: string;
        message?: string;
      };
      if (e.killed && e.signal === 'SIGTERM') {
        return clampCommandOutput(
          `$ ${command}\n[command timed out after ${RUN_COMMAND_TIMEOUT_MS / 1000}s]`,
        );
      }
      const parts = [
        `$ ${command}`,
        `exit code: ${typeof e.code === 'number' ? e.code : 'unknown'}`,
        e.stdout ? `stdout:\n${e.stdout}` : '',
        e.stderr ? `stderr:\n${e.stderr}` : e.message ? `error: ${e.message}` : '',
      ].filter(Boolean);
      return clampCommandOutput(parts.join('\n\n'));
    }
  }

  private resolveSession(
    sessionId: string,
    workspacePath: string | undefined,
    documentContext: DocumentContext | undefined,
  ): SessionState {
    const modelKey = bareGeminiModelKey(
      this.config?.model ?? (documentContext as { model?: string } | undefined)?.model,
    );
    const existing = this.sessionStates.get(sessionId);
    if (existing) {
      if (workspacePath) existing.workspacePath = workspacePath;
      if (existing.modelKey !== modelKey) {
        existing.modelKey = modelKey;
        existing.toolLoop.setModelKey(modelKey);
      }
      return existing;
    }
    const state: SessionState = {
      sessionId,
      workspacePath,
      modelKey,
      toolLoop: new AntigravityToolLoopProtocol({ modelKey, server: this.server }),
      abortController: null,
    };
    this.sessionStates.set(sessionId, state);
    return state;
  }

  // --- Static injection (Electron main process, at startup) ----------------

  static setMCPConfigLoader(
    loader: ((workspacePath?: string) => Promise<Record<string, MCPServerConfig>>) | null,
  ): void {
    GeminiAntigravityProvider.mcpConfigLoader = loader;
  }

  static setShellEnvironmentLoader(loader: (() => Record<string, string> | null) | null): void {
    GeminiAntigravityProvider.shellEnvironmentLoader = loader;
  }

  static setEnhancedPathLoader(loader: (() => string) | null): void {
    GeminiAntigravityProvider.enhancedPathLoader = loader;
  }

  static setAgyPathLoader(loader: (() => string | null) | null): void {
    GeminiAntigravityProvider.agyPathLoader = loader;
  }

  static setToolExecutor(executor: GeminiToolExecutor | null): void {
    GeminiAntigravityProvider.toolExecutor = executor;
  }

  static setServerConfigLoader(loader: (() => GeminiServerConfig) | null): void {
    GeminiAntigravityProvider.serverConfigLoader = loader;
  }

  // --- Detection and catalog ----------------------------------------------

  /** Resolve the `agy` CLI executable path from configured loader, PATH, or standard install paths. */
  static resolveAgyExecutablePath(enhancedPath?: string): string | undefined {
    const configured = GeminiAntigravityProvider.agyPathLoader?.();
    if (configured && existsSync(configured)) return configured;

    const searchPath =
      enhancedPath ?? GeminiAntigravityProvider.enhancedPathLoader?.() ?? process.env.PATH ?? '';
    for (const dir of searchPath.split(path.delimiter)) {
      if (!dir) continue;
      const candidate = path.join(dir, DESCRIPTOR.executableName);
      if (existsSync(candidate)) return candidate;
    }

    const home = os.homedir();
    for (const rel of DESCRIPTOR.homeRelativeInstallPaths) {
      const candidate = path.join(home, rel);
      if (existsSync(candidate)) return candidate;
    }
    return undefined;
  }

  /**
   * True when the Antigravity CLI (`agy`) is present. The IDE's language
   * server is deliberately not enough: production turns only run through
   * `agy` (the legacy server transport needs an injected tool executor), so
   * reporting it installed would advertise a provider that fails every turn.
   */
  static isInstalled(): boolean {
    return GeminiAntigravityProvider.resolveAgyExecutablePath() !== undefined;
  }

  /** Path the install is expected at, for the settings panel to show. */
  static installPath(): string {
    return (
      GeminiAntigravityProvider.resolveAgyExecutablePath() ??
      path.join(
        os.homedir(),
        process.platform === 'win32' ? 'AppData/Local/agy/bin/agy.exe' : '.local/bin/agy',
      )
    );
  }

  /**
   * Discover the Gemini model catalog via `agy models` (with scrubbed API keys
   * so only the user's Antigravity CLI login is consulted), falling back to
   * `discoverGeminiModels` (seed list / running hub) if `agy` is unavailable.
   */
  static async getModels(): Promise<AIModel[]> {
    const command =
      GeminiAntigravityProvider.agyPathLoader?.() ??
      GeminiAntigravityProvider.resolveAgyExecutablePath() ??
      DESCRIPTOR.executableName;
    try {
      const { execFile } = await import('child_process');
      const output = await new Promise<string>((resolve, reject) => {
        execFile(
          command,
          ['models'],
          {
            encoding: 'utf8',
            timeout: 10_000,
            windowsHide: true,
            env: buildHeadlessChildEnvironment(GeminiAntigravityProvider.loaders()),
          },
          (error, stdout) => (error ? reject(error) : resolve(stdout)),
        );
      });
      const parsed = parseAgyModelList(output);
      if (parsed.length > 0) {
        console.log(`[Gemini] agy models at ${command} returned ${parsed.length} Gemini model(s)`);
        return parsed.map((m) => ({
          id: `${GEMINI_ANTIGRAVITY_PROVIDER}:${m.key}`,
          name: m.displayName,
          provider: GEMINI_ANTIGRAVITY_PROVIDER,
        }));
      }
      console.warn(
        `[Gemini] agy models at ${command} listed no gemini-* models; using the seed catalog. Output head: ${JSON.stringify(output.slice(0, 300))}`,
      );
    } catch (error) {
      // Fall through to seed/hub catalog, but say so: a silent fallback here
      // is indistinguishable from the CLI working (#1653).
      const detail = error instanceof Error ? error.message : String(error);
      console.warn(`[Gemini] agy models at ${command} failed (${detail}); using the seed catalog`);
    }

    const models = await discoverGeminiModels(AntigravityServerManager.shared());
    return models.map((m) => ({
      id: `${GEMINI_ANTIGRAVITY_PROVIDER}:${m.key}`,
      name: m.displayName,
      provider: GEMINI_ANTIGRAVITY_PROVIDER,
    }));
  }
}
