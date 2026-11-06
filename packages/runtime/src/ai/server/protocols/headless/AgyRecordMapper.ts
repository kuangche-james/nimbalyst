/**
 * Pure NDJSON-record -> `ProtocolEvent[]` mapper for Antigravity CLI (`agy`).
 *
 * Kept dependency-free (no `child_process`, no `fs`) because it is called from
 * two places that cannot share a process boundary:
 *   - `AgyCliProtocol`, on live stdout lines during a turn;
 *   - `GeminiAntigravityRawParser`, when replaying stored `raw_messages` rows to
 *     rebuild a historical transcript.
 *
 * Verified against `agy` 1.2.16 (`agy --print <prompt> --output-format stream-json`):
 *   - `{"event":"init","conversation_id":"<uuid>","init":{"model":"...","cwd":"...","tools":[...],"permission_mode":"..."}}`
 *   - `{"event":"step_update","step_update":{"conversation_id":"<uuid>","step_index":2,"state":"ACTIVE","step_type":"tool","tool_name":"read_file","tool_info":{"name":"read_file","parameters":{"Path":"..."}}}}`
 *   - `{"event":"step_update","step_update":{"conversation_id":"<uuid>","step_index":2,"state":"DONE","step_type":"tool","tool_name":"read_file","duration_seconds":0.32,"tool_info":{"name":"read_file","parameters":{"Path":"..."},"output":"..."}}}`
 *   - `{"event":"step_update","step_update":{"conversation_id":"<uuid>","step_index":3,"state":"ACTIVE","step_type":"agent_response","text_delta":"..."}}`
 *   - `{"event":"result","result":{"conversation_id":"<uuid>","status":"SUCCESS","response":"...","duration_seconds":...,"num_turns":1,"usage":{"input_tokens":...,"output_tokens":...,"thinking_tokens":...,"cache_read_tokens":...,"total_tokens":...}}}`
 */

import type { ProtocolEvent, ToolResult } from '../ProtocolInterface';

export type AgyTokenUsage = NonNullable<ProtocolEvent['usage']>;

export interface AgyEditChange {
  path: string;
  kind: 'add' | 'update';
  beforeContent: string;
  afterContent: string;
}

/**
 * Map one parsed `agy --print --output-format stream-json` NDJSON record onto
 * `ProtocolEvent`s. Always begins with `raw_event` so `HeadlessCliAgentProvider`
 * persists the line to `raw_messages`.
 */
export function mapAgyRecord(
  record: Record<string, unknown>,
  editChanges?: AgyEditChange[],
): ProtocolEvent[] {
  const out: ProtocolEvent[] = [{ type: 'raw_event', metadata: { rawEvent: record } }];
  const eventKind = str(record.event) ?? str(record.type);

  switch (eventKind) {
    case 'init':
      return out;

    case 'step_update': {
      const step = asRecord(record.step_update) ?? record;
      const stepType = str(step.step_type);
      const state = (str(step.state) ?? '').toUpperCase();
      const conversationId = str(step.conversation_id) ?? str(record.conversation_id) ?? 'agy';
      const stepIndex = typeof step.step_index === 'number' ? step.step_index : 0;

      if (stepType === 'agent_response') {
        const thinkingDelta = str(step.thinking_delta) ?? str(step.thought_delta);
        if (thinkingDelta) {
          out.push({ type: 'reasoning', content: thinkingDelta });
        }
        const textDelta = str(step.text_delta);
        if (textDelta) {
          out.push({ type: 'text', content: textDelta });
        }
        return out;
      }

      if (stepType === 'thinking' || stepType === 'thought') {
        const delta = str(step.thinking_delta) ?? str(step.thought_delta) ?? str(step.text_delta);
        if (delta) {
          out.push({ type: 'reasoning', content: delta });
        }
        return out;
      }

      if (stepType === 'tool') {
        const toolInfo = asRecord(step.tool_info);
        const toolName = str(step.tool_name) ?? str(toolInfo?.name) ?? 'tool';
        const callId = str(step.tool_call_id) ?? `${conversationId}:step-${stepIndex}`;
        const args = normalizeAgyToolArguments(asRecord(toolInfo?.parameters) ?? {});

        if (state === 'ACTIVE' || state === 'RUNNING' || state === 'PENDING') {
          out.push({
            type: 'tool_call',
            toolCall: {
              id: callId,
              name: toolName,
              arguments: args,
            },
          });
          return out;
        }

        if (state === 'DONE' || state === 'ERROR' || state === 'FAILED' || state === 'CANCELLED') {
          const isError =
            state === 'ERROR' ||
            state === 'FAILED' ||
            state === 'CANCELLED' ||
            Boolean(toolInfo?.error) ||
            Boolean(step.error);
          const output = str(toolInfo?.output) ?? str(step.output);
          const errorText =
            str(toolInfo?.error) ??
            str(step.error) ??
            (isError ? (output ?? `Tool ${toolName} failed (${state})`) : undefined);
          const command = str(args.command);

          const resultPayload: ToolResult = {
            success: !isError,
            ...(output !== undefined ? { output } : {}),
            ...(errorText !== undefined ? { error: errorText } : {}),
            ...(command ? { command } : {}),
            ...(!isError && editChanges && editChanges.length > 0 ? { changes: editChanges } : {}),
          };

          out.push({
            type: 'tool_result',
            toolResult: {
              id: callId,
              name: toolName,
              result: resultPayload,
            },
          });
        }
      }
      return out;
    }

    case 'result': {
      const res = asRecord(record.result) ?? record;
      const status = str(res.status) ?? str(record.status) ?? '';
      const isError =
        Boolean(res.is_error) ||
        Boolean(record.is_error) ||
        (status !== '' && status.toUpperCase() !== 'SUCCESS');

      if (isError) {
        const errText =
          str(res.error) ??
          str(res.message) ??
          str(res.response) ??
          str(record.error) ??
          `Antigravity CLI turn failed (${status || 'error'})`;
        out.push({ type: 'error', error: errText });
        return out;
      }

      const usage = normalizeAgyUsage(res.usage ?? record.usage);
      const sessionId = str(res.conversation_id) ?? str(record.conversation_id);
      out.push({
        type: 'complete',
        content: str(res.response) ?? str(res.result),
        usage,
        ...(sessionId ? { metadata: { sessionId } } : {}),
      });
      return out;
    }

    case 'error': {
      const errText =
        str(record.error) ??
        str(record.message) ??
        str(asRecord(record.error)?.message) ??
        'Antigravity CLI error';
      out.push({ type: 'error', error: errText });
      return out;
    }

    default:
      return out;
  }
}

/**
 * Normalize `agy` tool parameters so Nimbalyst's file-path and shell-command
 * extractors (`extractFilePath`, `SessionFileTracker`, `pathResolver`) see
 * canonical `file_path`, `path`, and `command` keys alongside the original
 * PascalCase parameters (`Path`, `TargetFile`, `CommandLine`).
 */
export function normalizeAgyToolArguments(
  rawParams: Record<string, unknown>,
): Record<string, unknown> {
  const args: Record<string, unknown> = { ...rawParams };
  const filePath =
    str(rawParams.Path) ??
    str(rawParams.TargetFile) ??
    str(rawParams.file_path) ??
    str(rawParams.filePath) ??
    str(rawParams.path);
  if (filePath) {
    if (typeof args.file_path !== 'string') args.file_path = filePath;
    if (typeof args.path !== 'string') args.path = filePath;
  }
  const command = str(rawParams.CommandLine) ?? str(rawParams.command);
  if (command && typeof args.command !== 'string') {
    args.command = command;
  }
  return args;
}

/**
 * Normalize `agy`'s `result.usage` object into Nimbalyst's `TokenUsage`.
 *
 * In `agy` 1.2.16:
 *   `{"input_tokens":21594,"output_tokens":46,"thinking_tokens":0,"cache_read_tokens":12211,"total_tokens":21640}`
 *   `{"input_tokens":13264,"output_tokens":5,"thinking_tokens":27,"cache_read_tokens":0,"total_tokens":13296}`
 * `input_tokens` already includes `cache_read_tokens`, whereas `thinking_tokens`
 * is reported separately from `output_tokens` and included in `total_tokens`.
 */
export function normalizeAgyUsage(raw: unknown): AgyTokenUsage | undefined {
  const u = asRecord(raw);
  if (!u) return undefined;
  const rawInput = num(u.input_tokens) ?? num(u.inputTokens);
  const rawOutput = num(u.output_tokens) ?? num(u.outputTokens);
  if (rawInput === undefined && rawOutput === undefined) return undefined;

  const cacheRead = num(u.cache_read_tokens) ?? num(u.cacheReadTokens) ?? 0;
  const cacheWrite = num(u.cache_write_tokens) ?? num(u.cacheWriteTokens) ?? 0;
  const thinking = num(u.thinking_tokens) ?? num(u.thinkingTokens) ?? 0;
  const reportedTotal = num(u.total_tokens) ?? num(u.totalTokens);

  const baseInput = rawInput ?? 0;
  const baseOutput = rawOutput ?? 0;
  const outputTokens = baseOutput + thinking;

  // If reportedTotal equals baseInput + cacheRead + outputTokens (and cacheRead > 0),
  // then baseInput was uncached-only; otherwise baseInput already includes cacheRead.
  const inputTokens =
    cacheRead > 0 && reportedTotal === baseInput + cacheRead + outputTokens
      ? baseInput + cacheRead + cacheWrite
      : baseInput;
  const totalTokens = reportedTotal ?? inputTokens + outputTokens;

  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: totalTokens,
    ...(cacheRead > 0 ? { cache_read_input_tokens: cacheRead } : {}),
    ...(cacheWrite > 0 ? { cache_creation_input_tokens: cacheWrite } : {}),
    ...(thinking > 0 ? { reasoning_tokens: thinking } : {}),
  };
}

function asRecord(v: unknown): Record<string, unknown> | undefined {
  return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : undefined;
}

function str(v: unknown): string | undefined {
  return typeof v === 'string' && v.length > 0 ? v : undefined;
}

function num(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}
