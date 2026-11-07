/**
 * Canonical-event parser for `antigravity-gemini-agent`.
 *
 * Its raw log is not a vendor wire format. The language server streams nothing
 * and reports no events; the host's own tool loop decides what happened, so the
 * rows are Nimbalyst's own minimal record of a turn:
 *
 *   direction 'input',  metadata.role 'user'      -> plain prompt text
 *   direction 'output', metadata.role 'assistant' -> plain answer text
 *   direction 'output', metadata.role 'tool'      -> {"name","args","result"}
 *
 * Until this parser existed, Gemini sessions fell through to
 * `ClaudeCodeRawParser` — where the text rows happened to work via its
 * not-JSON fallbacks, but a tool row parsed as JSON, matched none of the
 * Claude SDK shapes, and produced *nothing*. Tool calls were simply absent from
 * a reloaded Gemini transcript. That was invisible while Gemini could not track
 * files; it is not acceptable now that it can.
 *
 * A tool row is self-contained: one row carries the call and its result
 * together, so this parser emits `tool_call_started` and `tool_call_completed`
 * as a pair from a single message. That differs from every streaming agent,
 * where the two arrive separately.
 *
 * The shape is deliberately unchanged from the years this provider shipped as
 * an extension, so rows written back then parse identically to ones written
 * now. The one addition is `metadata.toolUseId`; rows predating it fall back to
 * a synthetic id derived from the row, which is stable across reloads because
 * the row id is.
 */

import type { RawMessage } from '../TranscriptTransformer';
import type {
  IRawMessageParser,
  ParseContext,
  CanonicalEventDescriptor,
} from './IRawMessageParser';
import { mapAgyRecord } from '../../protocols/headless/AgyRecordMapper';
import type { ProtocolEvent } from '../../protocols/ProtocolInterface';

/** Tool argument keys that name a file, in the Gemini toolset's spelling. */
const FILE_PATH_KEYS = ['path', 'file_path', 'Path', 'TargetFile'] as const;

export class GeminiAntigravityRawParser implements IRawMessageParser {
  async parseMessage(
    msg: RawMessage,
    context: ParseContext,
  ): Promise<CanonicalEventDescriptor[]> {
    if (msg.hidden) return [];
    return msg.direction === 'input' ? this.parseInput(msg) : this.parseOutput(msg, context);
  }

  private parseInput(msg: RawMessage): CanonicalEventDescriptor[] {
    const raw = String(msg.content ?? '').trim();
    if (!raw) return [];
    let text = raw;
    try {
      const parsed = JSON.parse(raw) as { text?: unknown };
      if (parsed && typeof parsed === 'object' && typeof parsed.text === 'string') {
        text = parsed.text;
      }
    } catch {
      // Plain text input
    }
    if (!text) return [];
    return [{
      type: 'user_message',
      text,
      mode: (msg.metadata?.mode as 'agent' | 'planning' | undefined) ?? 'agent',
      createdAt: msg.createdAt,
    }];
  }

  private async parseOutput(
    msg: RawMessage,
    context: ParseContext,
  ): Promise<CanonicalEventDescriptor[]> {
    if (msg.metadata?.role === 'tool') {
      return this.parseToolRow(msg);
    }

    // Legacy assistant text is stored verbatim with metadata.role === 'assistant'.
    // Trust it as text even if the model's answer happens to be valid JSON.
    if (msg.metadata?.role === 'assistant') {
      const text = String(msg.content ?? '').trim();
      return text ? [{ type: 'assistant_message', text, createdAt: msg.createdAt }] : [];
    }

    const rawText = String(msg.content ?? '').trim();
    if (!rawText) return [];

    let record: Record<string, unknown> | undefined;
    try {
      const parsed = JSON.parse(rawText);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        record = parsed as Record<string, unknown>;
      }
    } catch {
      // Not JSON — fall back to plain text assistant message.
    }

    if (!record) {
      return [{ type: 'assistant_message', text: rawText, createdAt: msg.createdAt }];
    }

    // Turn-final assistant message written by HeadlessCliAgentProvider.storeAssistantResponse.
    if (record.type === 'item.completed') {
      const item = record.item as {
        type?: unknown;
        role?: unknown;
        content?: Array<{ type?: unknown; text?: unknown }>;
      } | undefined;
      if (item?.type === 'message' && item.role === 'assistant' && Array.isArray(item.content)) {
        const text = item.content
          .filter((c) => c?.type === 'output_text' && typeof c.text === 'string')
          .map((c) => c.text as string)
          .join('');
        return text ? [{ type: 'assistant_message', text, createdAt: msg.createdAt }] : [];
      }
      return [];
    }

    if (record.type === 'nimbalyst_tool_use') {
      const id = typeof record.id === 'string' ? record.id : null;
      if (id && context.hasToolCall && (context.hasToolCall(id) || await context.findByProviderToolCallId?.(id))) {
        return [];
      }
      const args = record.input;
      return [{
        type: 'tool_call_started',
        toolName: typeof record.name === 'string' ? record.name : 'unknown',
        toolDisplayName: typeof record.name === 'string' ? record.name : 'unknown',
        arguments: (args && typeof args === 'object' ? args : {}) as Record<string, unknown>,
        providerToolCallId: id,
        createdAt: msg.createdAt,
      }];
    }
    if (record.type === 'nimbalyst_tool_result') {
      const id = typeof record.tool_use_id === 'string'
        ? record.tool_use_id
        : typeof record.id === 'string' ? record.id : null;
      if (!id) return [];
      const isError = record.is_error === true;
      return [{
        type: 'tool_call_completed',
        providerToolCallId: id,
        status: isError ? 'error' : 'completed',
        isError,
        result: typeof record.result === 'string' ? record.result : safeStringify(record.result),
      }];
    }

    // Antigravity CLI (`agy --print --output-format stream-json`) NDJSON records.
    if (typeof record.event === 'string') {
      const out: CanonicalEventDescriptor[] = [];
      for (const ev of mapAgyRecord(record)) {
        const projected = projectAgyProtocolEvent(ev, msg.createdAt);
        if (projected) out.push(projected);
      }
      return out;
    }

    return [{ type: 'assistant_message', text: rawText, createdAt: msg.createdAt }];
  }

  private parseToolRow(msg: RawMessage): CanonicalEventDescriptor[] {
    let row: { name?: unknown; args?: unknown; result?: unknown };
    try {
      row = JSON.parse(msg.content) as typeof row;
    } catch {
      // A tool row that is not JSON is corrupt rather than textual. Emitting it
      // as an assistant message would put raw tool output in the conversation.
      return [];
    }
    const toolName = typeof row.name === 'string' && row.name ? row.name : 'unknown';
    const args = (row.args && typeof row.args === 'object' ? row.args : {}) as Record<string, unknown>;
    const providerToolCallId = typeof msg.metadata?.toolUseId === 'string'
      ? msg.metadata.toolUseId
      : `gemini-tool-${msg.id}`;

    const resultText = typeof row.result === 'string' ? row.result : safeStringify(row.result);
    // The tool loop reports failures as text, not as a status: an unavailable
    // tool and a failed write both come back as a JSON body carrying `isError`.
    // Anything else is a success, including an empty result.
    const isError = looksLikeErrorResult(row.result);

    return [
      {
        type: 'tool_call_started',
        toolName,
        toolDisplayName: toolName,
        arguments: args,
        targetFilePath: readTargetFilePath(args),
        providerToolCallId,
        createdAt: msg.createdAt,
      },
      {
        type: 'tool_call_completed',
        providerToolCallId,
        status: isError ? 'error' : 'completed',
        isError,
        result: resultText,
      },
    ];
  }
}

function projectAgyProtocolEvent(
  ev: ProtocolEvent,
  createdAt?: Date,
): CanonicalEventDescriptor | null {
  switch (ev.type) {
    case 'tool_call': {
      const tc = ev.toolCall;
      if (!tc) return null;
      const args = (tc.arguments ?? {}) as Record<string, unknown>;
      return {
        type: 'tool_call_started',
        toolName: tc.name,
        toolDisplayName: tc.name,
        arguments: args,
        targetFilePath: readTargetFilePath(args),
        providerToolCallId: tc.id ?? null,
        createdAt,
      };
    }
    case 'tool_result': {
      const tr = ev.toolResult;
      if (!tr || !tr.id) return null;
      const isObj = typeof tr.result === 'object' && tr.result !== null;
      const isError = isObj && (tr.result as { success?: boolean }).success === false;
      const resultStr =
        typeof tr.result === 'string' ? tr.result : safeStringify(tr.result);
      return {
        type: 'tool_call_completed',
        providerToolCallId: tr.id,
        status: isError ? 'error' : 'completed',
        isError,
        result: resultStr,
      };
    }
    default:
      return null;
  }
}

function looksLikeErrorResult(result: unknown): boolean {
  if (typeof result === 'string') return false;
  if (!result || typeof result !== 'object') return false;
  return (result as { isError?: unknown }).isError === true;
}

function readTargetFilePath(args: Record<string, unknown>): string | null {
  for (const key of FILE_PATH_KEYS) {
    const value = args[key];
    if (typeof value === 'string' && value) return value;
  }
  return null;
}

function safeStringify(value: unknown): string {
  if (value === undefined || value === null) return '';
  try {
    return JSON.stringify(value) ?? '';
  } catch {
    return String(value);
  }
}
