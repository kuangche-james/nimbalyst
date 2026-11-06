// @vitest-environment node
/**
 * Unit and integration tests for the Antigravity CLI (`agy`) provider,
 * protocol (`AgyCliProtocol`), record mapper (`mapAgyRecord`), model discovery
 * (`parseAgyModelList`), and transcript parser (`GeminiAntigravityRawParser`).
 */
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { BaseAgentProvider } from '../../BaseAgentProvider';
import { GeminiAntigravityProvider } from '../../GeminiAntigravityProvider';
import {
  AGY_INLINE_DOCUMENT_BUDGET,
  AGY_MAX_COMMAND_LINE_CHARS,
  AgyCliProtocol,
  formatAgyPromptWithAttachments,
  mapAgyRecord,
  normalizeAgyToolArguments,
  normalizeAgyUsage,
} from '../../../protocols/AgyCliProtocol';
import {
  formatModelDisplayName,
  isOfferedFlashModel,
  parseAgyModelList,
  selectGeminiModels,
} from '../geminiAntigravityModels';
import { AntigravityServerManager } from '../AntigravityServerManager';
import { GeminiAntigravityRawParser } from '../../../transcript/parsers/GeminiAntigravityRawParser';
import type { ParseContext } from '../../../transcript/parsers/IRawMessageParser';
import type { RawMessage } from '../../../transcript/TranscriptTransformer';
import type { StreamChunk } from '../../../types';

async function collect(stream: AsyncIterableIterator<StreamChunk>): Promise<StreamChunk[]> {
  const out: StreamChunk[] = [];
  for await (const chunk of stream) out.push(chunk);
  return out;
}

afterEach(() => {
  BaseAgentProvider.setTrustChecker(null);
  GeminiAntigravityProvider.setToolExecutor(null);
  GeminiAntigravityProvider.setAgyPathLoader(null);
  GeminiAntigravityProvider.setMCPConfigLoader(null);
  GeminiAntigravityProvider.setShellEnvironmentLoader(null);
  GeminiAntigravityProvider.setEnhancedPathLoader(null);
  vi.restoreAllMocks();
});

describe('mapAgyRecord & normalizeAgyUsage', () => {
  it('maps live agy 1.2.16 init, step_update, and result NDJSON frames', () => {
    const records = [
      {
        event: 'init',
        conversation_id: '02c1b0b1-bd3f-4fa1-91f2-d063e0bbef44',
        init: {
          model: 'gemini-3.8-flash-low',
          cwd: 'c:\\Work\\nimbalyst-study',
          tools: ['read_file', 'write_file', 'run_command'],
          permission_mode: 'always-proceed',
        },
      },
      {
        event: 'step_update',
        step_update: {
          conversation_id: '02c1b0b1-bd3f-4fa1-91f2-d063e0bbef44',
          step_index: 2,
          state: 'ACTIVE',
          step_type: 'tool',
          tool_name: 'read_file',
          tool_info: {
            name: 'read_file',
            parameters: { EndLine: 1, Path: 'C:\\Work\\nimbalyst-study\\CLAUDE.md', StartLine: 1 },
          },
        },
      },
      {
        event: 'step_update',
        step_update: {
          conversation_id: '02c1b0b1-bd3f-4fa1-91f2-d063e0bbef44',
          step_index: 2,
          state: 'DONE',
          step_type: 'tool',
          tool_name: 'read_file',
          duration_seconds: 0.321,
          tool_info: {
            name: 'read_file',
            parameters: { EndLine: 1, Path: 'C:\\Work\\nimbalyst-study\\CLAUDE.md', StartLine: 1 },
            output: '118 lines, 12352 bytes',
          },
        },
      },
      {
        event: 'step_update',
        step_update: {
          conversation_id: '02c1b0b1-bd3f-4fa1-91f2-d063e0bbef44',
          step_index: 3,
          state: 'ACTIVE',
          step_type: 'agent_response',
          text_delta: '# CLAUDE',
        },
      },
      {
        event: 'step_update',
        step_update: {
          conversation_id: '02c1b0b1-bd3f-4fa1-91f2-d063e0bbef44',
          step_index: 3,
          state: 'ACTIVE',
          step_type: 'agent_response',
          text_delta: '.md\n',
        },
      },
      {
        event: 'result',
        result: {
          conversation_id: '02c1b0b1-bd3f-4fa1-91f2-d063e0bbef44',
          status: 'SUCCESS',
          response: '# CLAUDE.md\n',
          duration_seconds: 40.68,
          num_turns: 1,
          usage: {
            input_tokens: 21594,
            output_tokens: 46,
            thinking_tokens: 0,
            cache_read_tokens: 12211,
            total_tokens: 21640,
          },
        },
      },
    ];

    const events = records.flatMap((r) => mapAgyRecord(r));

    const toolCall = events.find((e) => e.type === 'tool_call');
    expect(toolCall?.toolCall).toMatchObject({
      id: '02c1b0b1-bd3f-4fa1-91f2-d063e0bbef44:step-2',
      name: 'read_file',
      arguments: {
        Path: 'C:\\Work\\nimbalyst-study\\CLAUDE.md',
        file_path: 'C:\\Work\\nimbalyst-study\\CLAUDE.md',
        path: 'C:\\Work\\nimbalyst-study\\CLAUDE.md',
      },
    });

    const toolResult = events.find((e) => e.type === 'tool_result');
    expect(toolResult?.toolResult).toMatchObject({
      id: '02c1b0b1-bd3f-4fa1-91f2-d063e0bbef44:step-2',
      name: 'read_file',
      result: {
        success: true,
        output: '118 lines, 12352 bytes',
      },
    });

    const textJoined = events
      .filter((e) => e.type === 'text')
      .map((e) => e.content)
      .join('');
    expect(textJoined).toBe('# CLAUDE.md\n');

    const complete = events.find((e) => e.type === 'complete');
    expect(complete?.usage).toEqual({
      input_tokens: 21594,
      output_tokens: 46,
      total_tokens: 21640,
      cache_read_input_tokens: 12211,
    });
  });

  it('includes thinking_tokens in output_tokens when normalizing agy usage', () => {
    expect(
      normalizeAgyUsage({
        input_tokens: 13264,
        output_tokens: 5,
        thinking_tokens: 27,
        cache_read_tokens: 0,
        total_tokens: 13296,
      }),
    ).toEqual({
      input_tokens: 13264,
      output_tokens: 32,
      total_tokens: 13296,
      reasoning_tokens: 27,
    });
  });

  it('normalizes CommandLine onto command for run_command tool calls', () => {
    expect(normalizeAgyToolArguments({ CommandLine: 'git status' })).toEqual({
      CommandLine: 'git status',
      command: 'git status',
    });
  });
});

describe('GeminiAntigravityProvider via AgyCliProtocol (HeadlessCliAgentProvider)', () => {
  it('streams text, tool calls, edit snapshots, and token usage from agy NDJSON', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'agy-provider-'));
    try {
      const targetFile = join(dir, 'hello.txt');
      writeFileSync(targetFile, 'before\n', 'utf8');

      BaseAgentProvider.setTrustChecker(() => ({ trusted: true, mode: 'allow-all' }));
      GeminiAntigravityProvider.setAgyPathLoader(() => process.execPath);

      let capturedArgs: string[] = [];
      const protocol = new AgyCliProtocol({
        runNdjson: async function* (options) {
          capturedArgs = options.args;
          yield {
            kind: 'record',
            value: {
              event: 'init',
              conversation_id: 'conv-uuid-1',
              init: { model: 'gemini-4-argon-medium', cwd: dir },
            },
          };
          yield {
            kind: 'record',
            value: {
              event: 'step_update',
              step_update: {
                conversation_id: 'conv-uuid-1',
                step_index: 1,
                state: 'ACTIVE',
                step_type: 'tool',
                tool_name: 'write_file',
                tool_info: {
                  name: 'write_file',
                  parameters: { Path: targetFile, Content: 'after\n' },
                },
              },
            },
          };
          // Simulate agy writing the file before emitting DONE
          writeFileSync(targetFile, 'after\n', 'utf8');
          yield {
            kind: 'record',
            value: {
              event: 'step_update',
              step_update: {
                conversation_id: 'conv-uuid-1',
                step_index: 1,
                state: 'DONE',
                step_type: 'tool',
                tool_name: 'write_file',
                tool_info: {
                  name: 'write_file',
                  parameters: { Path: targetFile, Content: 'after\n' },
                  output: 'Updated hello.txt',
                },
              },
            },
          };
          yield {
            kind: 'record',
            value: {
              event: 'step_update',
              step_update: {
                conversation_id: 'conv-uuid-1',
                step_index: 2,
                state: 'ACTIVE',
                step_type: 'agent_response',
                text_delta: 'Updated hello.txt.',
              },
            },
          };
          yield {
            kind: 'record',
            value: {
              event: 'result',
              result: {
                conversation_id: 'conv-uuid-1',
                status: 'SUCCESS',
                response: 'Updated hello.txt.',
                usage: {
                  input_tokens: 1000,
                  output_tokens: 25,
                  thinking_tokens: 10,
                  cache_read_tokens: 500,
                  total_tokens: 1035,
                },
              },
            },
          };
        },
      });

      const provider = new GeminiAntigravityProvider({ protocol });
      await provider.initialize({ model: 'antigravity-gemini-agent:gemini-4-argon-medium' });

      const chunks = await collect(
        provider.sendMessage('update hello.txt', undefined, 'nim-session-1', undefined, dir),
      );

      expect(capturedArgs.slice(0, -1)).toEqual([
        '--print',
        '--output-format',
        'stream-json',
        '--model',
        'gemini-4-argon-medium',
        '--dangerously-skip-permissions',
        '--',
      ]);
      const sentPrompt = capturedArgs[capturedArgs.length - 1];
      expect(sentPrompt).toContain('<system_instructions>');
      expect(sentPrompt.endsWith('\n\nupdate hello.txt')).toBe(true);

      // Verify conversation_id was persisted for multi-turn resumption
      expect(provider.getProviderSessionData('nim-session-1')).toEqual({
        providerSessionId: 'conv-uuid-1',
      });
      expect(provider.getInitData()?.model).toBe('gemini-4-argon-medium');

      // Verify pre/post edit snapshots and terminal tool_call result were emitted in order
      const announceIdx = chunks.findIndex(
        (c) => c.type === 'tool_call' && c.toolCall?.result === undefined,
      );
      const preIdx = chunks.findIndex((c) => c.type === 'pre_edit_snapshot');
      const postIdx = chunks.findIndex((c) => c.type === 'post_edit_snapshot');
      const resultIdx = chunks.findIndex(
        (c) => c.type === 'tool_call' && c.toolCall?.result !== undefined,
      );
      const completeIdx = chunks.findIndex((c) => c.type === 'complete');

      expect(announceIdx).toBeGreaterThanOrEqual(0);
      expect(announceIdx).toBeLessThan(preIdx);
      expect(preIdx).toBeLessThan(postIdx);
      expect(postIdx).toBeLessThan(resultIdx);
      expect(resultIdx).toBeLessThan(completeIdx);

      expect(chunks[resultIdx].toolCall).toMatchObject({
        id: 'conv-uuid-1:step-1',
        name: 'write_file',
        arguments: {
          Path: targetFile,
          Content: 'after\n',
          file_path: targetFile,
          path: targetFile,
        },
        result: {
          success: true,
          output: 'Updated hello.txt',
        },
      });

      expect(chunks[preIdx].preEditSnapshot).toEqual({
        toolUseId: 'conv-uuid-1:step-1',
        entries: [{ path: targetFile, content: 'before\n', kind: 'update' }],
        authoritative: false,
      });
      expect(chunks[postIdx].postEditSnapshot).toEqual({
        toolUseId: 'conv-uuid-1:step-1',
        entries: [{ path: targetFile, content: 'after\n', kind: 'update' }],
      });

      const complete = chunks[chunks.length - 1];
      expect(complete.type).toBe('complete');
      expect(complete.usage).toEqual({
        input_tokens: 1000,
        output_tokens: 35,
        total_tokens: 1035,
        cache_read_input_tokens: 500,
        reasoning_tokens: 10,
      });

      // Second turn in the same session passes --conversation conv-uuid-1
      await collect(
        provider.sendMessage('follow up', undefined, 'nim-session-1', undefined, dir),
      );
      expect(capturedArgs.slice(0, 2)).toEqual(['--conversation', 'conv-uuid-1']);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('reconstructs pre-edit baseline when edit_file writes before ACTIVE is consumed, and avoids recording post-edit content for write_file', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'agy-race-'));
    try {
      const editPath = join(dir, 'edited.ts');
      const ambiguousPath = join(dir, 'ambiguous.ts');
      const writePath = join(dir, 'overwritten.ts');
      // All three files have ALREADY been modified on disk before ACTIVE arrives
      writeFileSync(editPath, 'export const count = 2;\n', 'utf8');
      writeFileSync(ambiguousPath, 'new\nnew\n', 'utf8');
      writeFileSync(writePath, 'brand new content\n', 'utf8');

      BaseAgentProvider.setTrustChecker(() => ({ trusted: true, mode: 'allow-all' }));
      GeminiAntigravityProvider.setAgyPathLoader(() => process.execPath);

      const protocol = new AgyCliProtocol({
        runNdjson: async function* () {
          yield {
            kind: 'record',
            value: {
              event: 'init',
              conversation_id: 'conv-race',
              init: { model: 'gemini-3.8-flash-medium', cwd: dir },
            },
          };
          yield {
            kind: 'record',
            value: {
              event: 'step_update',
              step_update: {
                conversation_id: 'conv-race',
                step_index: 1,
                state: 'ACTIVE',
                step_type: 'tool',
                tool_name: 'edit_file',
                tool_info: {
                  name: 'edit_file',
                  parameters: {
                    Path: editPath,
                    Target: 'export const count = 1;',
                    Replacement: 'export const count = 2;',
                  },
                },
              },
            },
          };
          yield {
            kind: 'record',
            value: {
              event: 'step_update',
              step_update: {
                conversation_id: 'conv-race',
                step_index: 1,
                state: 'DONE',
                step_type: 'tool',
                tool_name: 'edit_file',
                tool_info: {
                  name: 'edit_file',
                  parameters: {
                    Path: editPath,
                    Target: 'export const count = 1;',
                    Replacement: 'export const count = 2;',
                  },
                  output: 'Made 1 replacement',
                },
              },
            },
          };
          yield {
            kind: 'record',
            value: {
              event: 'step_update',
              step_update: {
                conversation_id: 'conv-race',
                step_index: 2,
                state: 'ACTIVE',
                step_type: 'tool',
                tool_name: 'edit_file',
                tool_info: {
                  name: 'edit_file',
                  parameters: {
                    Path: ambiguousPath,
                    Target: 'old',
                    Replacement: 'new',
                  },
                },
              },
            },
          };
          yield {
            kind: 'record',
            value: {
              event: 'step_update',
              step_update: {
                conversation_id: 'conv-race',
                step_index: 2,
                state: 'DONE',
                step_type: 'tool',
                tool_name: 'edit_file',
                tool_info: {
                  name: 'edit_file',
                  parameters: {
                    Path: ambiguousPath,
                    Target: 'old',
                    Replacement: 'new',
                  },
                  output: 'Made 1 replacement',
                },
              },
            },
          };
          yield {
            kind: 'record',
            value: {
              event: 'step_update',
              step_update: {
                conversation_id: 'conv-race',
                step_index: 3,
                state: 'ACTIVE',
                step_type: 'tool',
                tool_name: 'write_file',
                tool_info: {
                  name: 'write_file',
                  parameters: {
                    Path: writePath,
                    Content: 'brand new content\n',
                  },
                },
              },
            },
          };
          yield {
            kind: 'record',
            value: {
              event: 'step_update',
              step_update: {
                conversation_id: 'conv-race',
                step_index: 3,
                state: 'DONE',
                step_type: 'tool',
                tool_name: 'write_file',
                tool_info: {
                  name: 'write_file',
                  parameters: {
                    Path: writePath,
                    Content: 'brand new content\n',
                  },
                  output: 'Created file',
                },
              },
            },
          };
          yield {
            kind: 'record',
            value: {
              event: 'result',
              result: { conversation_id: 'conv-race', status: 'SUCCESS', response: 'Done' },
            },
          };
        },
      });

      const provider = new GeminiAntigravityProvider({ protocol });
      await provider.initialize({});
      const chunks = await collect(
        provider.sendMessage('apply edits', undefined, 'nim-race', undefined, dir),
      );

      const preSnapshots = chunks.filter((c) => c.type === 'pre_edit_snapshot');
      const postSnapshots = chunks.filter((c) => c.type === 'post_edit_snapshot');

      // Unambiguous single replacement recovers beforeContent while keeping authoritative: false
      expect(preSnapshots.map((c) => c.preEditSnapshot)).toEqual([
        {
          toolUseId: 'conv-race:step-1',
          entries: [{ path: editPath, content: 'export const count = 1;\n', kind: 'update' }],
          authoritative: false,
        },
      ]);

      // Ambiguous replacement (`old\nnew\n` -> `new\nnew\n`) and write_file both lost the
      // ACTIVE race with no way to recover the baseline. Neither emits a pre_edit_snapshot:
      // an empty baseline would be written as the pre-edit tag (replaceSpeculative) and
      // FileSnapshotCache cannot always fill it (gitignored, never-snapshotted paths), so
      // the whole file would show as added and Reject would empty it. Post-edit content
      // is still recorded for all three.
      expect(postSnapshots.flatMap((c) => c.postEditSnapshot?.entries.map((e) => e.path))).toEqual([
        editPath,
        ambiguousPath,
        writePath,
      ]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('reverse-applies lost-race edits only when the hunk is position-bound or unique, and never partially for multi-chunk edits', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'agy-reverse-'));
    try {
      const rangedPath = join(dir, 'ranged.ts');
      const multiPath = join(dir, 'multi.ts');
      const nestedPath = join(dir, 'nested.ts');
      // Every file is already in its post-edit state before ACTIVE is consumed.
      // `x = 2;` appears twice; only StartLine/EndLine can say which one was edited.
      writeFileSync(rangedPath, 'x = 2;\ny = 1;\nx = 2;\n', 'utf8');
      // First chunk is unique, second chunk is ambiguous (`new` twice).
      writeFileSync(multiPath, 'beta\nnew\nnew\n', 'utf8');
      // Replacement is a substring of the target, so reversal cannot be trusted.
      writeFileSync(nestedPath, 'foo\n', 'utf8');

      const toolStep = (
        stepIndex: number,
        state: 'ACTIVE' | 'DONE',
        name: string,
        parameters: Record<string, unknown>,
      ) => ({
        kind: 'record' as const,
        value: {
          event: 'step_update',
          step_update: {
            conversation_id: 'conv-reverse',
            step_index: stepIndex,
            state,
            step_type: 'tool',
            tool_name: name,
            tool_info: { name, parameters, ...(state === 'DONE' ? { output: 'ok' } : {}) },
          },
        },
      });
      const rangedParams = {
        Path: rangedPath,
        Target: 'x = 1;',
        Replacement: 'x = 2;',
        StartLine: 3,
        EndLine: 3,
      };
      const multiParams = {
        TargetFile: multiPath,
        ReplacementChunks: [
          { TargetContent: 'alpha', ReplacementContent: 'beta' },
          { TargetContent: 'old', ReplacementContent: 'new' },
        ],
      };
      const nestedParams = { Path: nestedPath, Target: 'foo bar', Replacement: 'foo' };

      BaseAgentProvider.setTrustChecker(() => ({ trusted: true, mode: 'allow-all' }));
      GeminiAntigravityProvider.setAgyPathLoader(() => process.execPath);

      const protocol = new AgyCliProtocol({
        runNdjson: async function* () {
          yield {
            kind: 'record',
            value: {
              event: 'init',
              conversation_id: 'conv-reverse',
              init: { model: 'gemini-3.8-flash-medium', cwd: dir },
            },
          };
          yield toolStep(1, 'ACTIVE', 'edit_file', rangedParams);
          yield toolStep(1, 'DONE', 'edit_file', rangedParams);
          yield toolStep(2, 'ACTIVE', 'multi_replace_file_content', multiParams);
          yield toolStep(2, 'DONE', 'multi_replace_file_content', multiParams);
          yield toolStep(3, 'ACTIVE', 'edit_file', nestedParams);
          yield toolStep(3, 'DONE', 'edit_file', nestedParams);
          yield {
            kind: 'record',
            value: {
              event: 'result',
              result: { conversation_id: 'conv-reverse', status: 'SUCCESS', response: 'Done' },
            },
          };
        },
      });

      const provider = new GeminiAntigravityProvider({ protocol });
      await provider.initialize({});
      const chunks = await collect(
        provider.sendMessage('apply edits', undefined, 'nim-reverse', undefined, dir),
      );

      const preSnapshots = chunks.filter((c) => c.type === 'pre_edit_snapshot');
      const postSnapshots = chunks.filter((c) => c.type === 'post_edit_snapshot');

      // Duplicate replacement text, but StartLine/EndLine pins it to line 3:
      // only that occurrence is reversed, the first `x = 2;` is left alone.
      //
      // The other two have no trustworthy baseline and emit no pre_edit_snapshot:
      // - one ambiguous chunk aborts the whole multi-chunk reversal rather than
      //   producing a baseline where only `beta` -> `alpha` was undone;
      // - `foo bar` -> `foo`: the replacement is inside the target, so reversal
      //   would loop on itself.
      expect(preSnapshots.map((c) => c.preEditSnapshot)).toEqual([
        {
          toolUseId: 'conv-reverse:step-1',
          entries: [{ path: rangedPath, content: 'x = 2;\ny = 1;\nx = 1;\n', kind: 'update' }],
          authoritative: false,
        },
      ]);
      expect(postSnapshots.flatMap((c) => c.postEditSnapshot?.entries.map((e) => e.path))).toEqual([
        rangedPath,
        multiPath,
        nestedPath,
      ]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('seeds prior conversation history when continuing an existing Gemini session without a native conversation id', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'agy-seed-'));
    try {
      BaseAgentProvider.setTrustChecker(() => ({ trusted: true, mode: 'allow-all' }));
      GeminiAntigravityProvider.setAgyPathLoader(() => process.execPath);

      const promptsSent: string[] = [];
      const protocol = new AgyCliProtocol({
        runNdjson: async function* (opts) {
          promptsSent.push(opts.args[opts.args.length - 1]);
          yield {
            kind: 'record',
            value: {
              event: 'init',
              conversation_id: 'conv-seeded-1',
              init: { model: 'gemini-4-argon-medium', cwd: dir },
            },
          };
          yield {
            kind: 'record',
            value: {
              event: 'result',
              result: { conversation_id: 'conv-seeded-1', status: 'SUCCESS', response: 'Reply' },
            },
          };
        },
      });

      const provider = new GeminiAntigravityProvider({ protocol });
      await provider.initialize({});

      const priorMessages = [
        { id: '1', role: 'user' as const, content: 'What is our API port?', timestamp: 1 },
        { id: '2', role: 'assistant' as const, content: 'The API port is 4100.', timestamp: 2 },
        { id: '3', role: 'user' as const, content: 'Change it to 4200.', timestamp: 3 },
      ];

      await collect(
        provider.sendMessage(
          'Change it to 4200.',
          undefined,
          'legacy-gemini-session',
          priorMessages,
          dir,
        ),
      );

      expect(promptsSent).toHaveLength(1);
      expect(promptsSent[0]).toContain('<conversation_history>');
      expect(promptsSent[0]).toContain('User:\nWhat is our API port?');
      expect(promptsSent[0]).toContain('Assistant:\nThe API port is 4100.');
      // Trailing duplicate of the current user message is not duplicated inside <conversation_history>
      const historyMatch = promptsSent[0].match(/<conversation_history>[\s\S]*?<\/conversation_history>/);
      expect(historyMatch?.[0]).not.toContain('Change it to 4200.');
      expect(promptsSent[0].endsWith('\n\nChange it to 4200.')).toBe(true);

      // Second turn uses the native conversation_id and does NOT re-seed <conversation_history>
      await collect(
        provider.sendMessage(
          'Now verify it.',
          undefined,
          'legacy-gemini-session',
          [...priorMessages, { id: '4', role: 'assistant' as const, content: 'Reply', timestamp: 4 }],
          dir,
        ),
      );

      expect(promptsSent).toHaveLength(2);
      expect(promptsSent[1]).not.toContain('<conversation_history>');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('delivers SessionOptions.mcpServers to agy mcp_config.json and reports deliveredMcpServerCount', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'agy-mcp-'));
    try {
      const mcpConfigPath = join(dir, 'mcp_config.json');
      writeFileSync(
        mcpConfigPath,
        JSON.stringify({
          mcpServers: {
            'user-custom': { command: 'node', args: ['custom.js'] },
            'nimbalyst:stale': { url: 'http://127.0.0.1:9999/mcp' },
          },
        }),
        'utf8',
      );

      BaseAgentProvider.setTrustChecker(() => ({ trusted: true, mode: 'allow-all' }));
      GeminiAntigravityProvider.setAgyPathLoader(() => process.execPath);
      GeminiAntigravityProvider.setMCPConfigLoader(async () => ({
        'nimbalyst-host': {
          type: 'sse',
          url: 'http://127.0.0.1:41001/mcp/host?sessionId=s1',
          headers: { Authorization: 'Bearer tok-123' },
        },
        'local-stdio': {
          type: 'stdio',
          command: 'npx',
          args: ['-y', '@modelcontextprotocol/server-memory'],
        },
      }));

      const protocol = new AgyCliProtocol({
        mcpConfigPath,
        runNdjson: async function* () {
          yield {
            kind: 'record',
            value: {
              event: 'init',
              conversation_id: 'conv-mcp-1',
              init: { model: 'gemini-4-argon-medium', cwd: dir },
            },
          };
          yield {
            kind: 'record',
            value: {
              event: 'result',
              result: { conversation_id: 'conv-mcp-1', status: 'SUCCESS', response: 'OK' },
            },
          };
        },
      });

      const provider = new GeminiAntigravityProvider({ protocol });
      await provider.initialize({});
      await collect(provider.sendMessage('spawn child', undefined, 'mcp-session', undefined, dir));

      expect(provider.getInitData()?.mcpServerCount).toBe(2);
      const written = JSON.parse(readFileSync(mcpConfigPath, 'utf8'));
      expect(written.mcpServers).toEqual({
        'user-custom': { command: 'node', args: ['custom.js'] },
        'nimbalyst:nimbalyst-host': {
          url: 'http://127.0.0.1:41001/mcp/host?sessionId=s1',
          headers: { Authorization: 'Bearer tok-123' },
        },
        'nimbalyst:local-stdio': {
          command: 'npx',
          args: ['-y', '@modelcontextprotocol/server-memory'],
        },
      });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('holds the next session\'s MCP config write until the previous agy process has produced output', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'agy-mcp-gate-'));
    try {
      BaseAgentProvider.setTrustChecker(() => ({ trusted: true, mode: 'allow-all' }));
      GeminiAntigravityProvider.setAgyPathLoader(() => process.execPath);

      // Each "process" starts only when the test says so, standing in for the
      // real CLI's startup window between spawn and its first stdout line.
      const timeline: string[] = [];
      let startFirst!: () => void;
      const firstMayStart = new Promise<void>((resolve) => {
        startFirst = resolve;
      });
      const makeProtocol = (label: string, gateOnStart?: Promise<void>) =>
        new AgyCliProtocol({
          syncMcpServers: async () => {
            timeline.push(`${label}:write`);
          },
          runNdjson: async function* () {
            if (gateOnStart) await gateOnStart;
            timeline.push(`${label}:init`);
            yield {
              kind: 'record',
              value: {
                event: 'init',
                conversation_id: `conv-${label}`,
                init: { model: 'gemini-4-argon-medium', cwd: dir },
              },
            };
            yield {
              kind: 'record',
              value: {
                event: 'result',
                result: { conversation_id: `conv-${label}`, status: 'SUCCESS', response: 'OK' },
              },
            };
          },
        });

      const first = new GeminiAntigravityProvider({ protocol: makeProtocol('a', firstMayStart) });
      const second = new GeminiAntigravityProvider({ protocol: makeProtocol('b') });
      await first.initialize({});
      await second.initialize({});

      const firstTurn = collect(first.sendMessage('one', undefined, 'gate-a', undefined, dir));
      // Give the first turn time to write its config and block on "process startup".
      await vi.waitFor(() => expect(timeline).toContain('a:write'));
      const secondTurn = collect(second.sendMessage('two', undefined, 'gate-b', undefined, dir));
      await new Promise((resolve) => setTimeout(resolve, 50));

      // The second session has not been allowed to overwrite the file yet.
      expect(timeline).toEqual(['a:write']);

      startFirst();
      await Promise.all([firstTurn, secondTurn]);
      expect(timeline).toEqual(['a:write', 'a:init', 'b:write', 'b:init']);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('bounds inlined attachments on Windows, preserves long prompts on POSIX, and fails explicitly with E2BIG instead of truncating', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'agy-cmdline-'));
    try {
      const smallDoc = join(dir, 'small.md');
      const largeDoc = join(dir, 'large.md');
      writeFileSync(smallDoc, 'short attachment text', 'utf8');
      writeFileSync(largeDoc, 'x'.repeat(AGY_INLINE_DOCUMENT_BUDGET + 2048), 'utf8');

      const prompt = formatAgyPromptWithAttachments(
        {
          content: 'Summarize these files',
          attachments: [
            {
              id: '1',
              type: 'document',
              filename: 'small.md',
              filepath: smallDoc,
              mimeType: 'text/markdown',
              size: 21,
              addedAt: 0,
            },
            {
              id: '2',
              type: 'document',
              filename: 'large.md',
              filepath: largeDoc,
              mimeType: 'text/markdown',
              size: AGY_INLINE_DOCUMENT_BUDGET + 2048,
              addedAt: 0,
            },
          ],
        },
        'Follow project rules carefully.',
        200,
        'win32',
      );

      expect(prompt).toContain('<system_instructions>\nFollow project rules carefully.\n</system_instructions>');
      expect(prompt).toContain('<file name="small.md">\nshort attachment text\n</file>');
      expect(prompt).toContain(`- large.md: ${largeDoc}`);
      expect(prompt.length).toBeLessThanOrEqual(AGY_MAX_COMMAND_LINE_CHARS);

      // Long prompt (35,000 chars) on POSIX is preserved in full without truncation
      const longSpec = `START-${'a'.repeat(35_000)}-END_REQUIREMENT`;
      const posixPrompt = formatAgyPromptWithAttachments(
        { content: longSpec },
        'System rule.',
        200,
        'linux',
      );
      expect(posixPrompt.endsWith('-END_REQUIREMENT')).toBe(true);
      expect(posixPrompt).toContain(longSpec);

      // On Windows, a prompt that itself exceeds the command-line limit throws E2BIG
      // rather than silently slicing off the final requirements
      expect(() =>
        formatAgyPromptWithAttachments({ content: longSpec }, 'System rule.', 200, 'win32'),
      ).toThrow(/E2BIG/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('GeminiAntigravityProvider.isInstalled', () => {
  it('ignores the IDE language server: only the agy CLI can serve a production turn', () => {
    vi.spyOn(GeminiAntigravityProvider, 'resolveAgyExecutablePath').mockReturnValue(undefined);
    vi.spyOn(AntigravityServerManager, 'isInstalled').mockReturnValue(true);
    expect(GeminiAntigravityProvider.isInstalled()).toBe(false);

    vi.spyOn(GeminiAntigravityProvider, 'resolveAgyExecutablePath').mockReturnValue('/usr/local/bin/agy');
    expect(GeminiAntigravityProvider.isInstalled()).toBe(true);
  });
});

describe('parseAgyModelList and Gemini 4 Argon / tieredModelIds support', () => {
  it('parses agy models output and filters to Google-served gemini-* models', () => {
    const output = [
      'gemini-4-argon-medium     Gemini 4 Argon (Medium)',
      'gemini-4-argon-high       Gemini 4 Argon (High)',
      'gemini-4-argon-low        Gemini 4 Argon (Low)',
      'gemini-3.8-flash-high     Gemini 3.8 Flash (High)',
      'gemini-3.8-flash-medium   Gemini 3.8 Flash (Medium)',
      'gemini-3.8-flash-low      Gemini 3.8 Flash (Low)',
      'claude-opus-5-5-low       Claude Opus 5.5 (Low)',
      'gpt-oss-120b-medium       GPT-OSS 120B (Medium)',
    ].join('\n');

    expect(parseAgyModelList(output)).toEqual([
      { key: 'gemini-4-argon-medium', displayName: 'Gemini 4 Argon (Medium)' },
      { key: 'gemini-4-argon-high', displayName: 'Gemini 4 Argon (High)' },
      { key: 'gemini-4-argon-low', displayName: 'Gemini 4 Argon (Low)' },
      { key: 'gemini-3.8-flash-high', displayName: 'Gemini 3.8 Flash (High)' },
      { key: 'gemini-3.8-flash-medium', displayName: 'Gemini 3.8 Flash (Medium)' },
      { key: 'gemini-3.8-flash-low', displayName: 'Gemini 3.8 Flash (Low)' },
    ]);
  });

  it('formats bare tieredModelIds keys and accepts Gemini 4 Argon tiers', () => {
    expect(formatModelDisplayName('gemini-4-argon-high')).toBe('Gemini 4 Argon (High)');
    expect(formatModelDisplayName('gemini-3.8-flash-medium')).toBe('Gemini 3.8 Flash (Medium)');
    expect(formatModelDisplayName('gemini-3.7-flash-tiered')).toBe('');

    expect(isOfferedFlashModel('Gemini 4 Argon (High)')).toBe(true);
    expect(isOfferedFlashModel('Gemini 4 Argon (Medium)')).toBe(true);

    const catalog = new Map([
      ['gemini-4-argon-high', { key: 'gemini-4-argon-high', enum: 'gemini-4-argon-high', displayName: '' }],
      ['gemini-3.8-flash-medium', { key: 'gemini-3.8-flash-medium', enum: 'gemini-3.8-flash-medium', displayName: '' }],
      ['claude-opus-5-5-low', { key: 'claude-opus-5-5-low', enum: 'claude-opus-5-5-low', displayName: '' }],
    ]);
    expect(selectGeminiModels(catalog, new Set())).toEqual([
      { key: 'gemini-3.8-flash-medium', displayName: 'Gemini 3.8 Flash (Medium)' },
      { key: 'gemini-4-argon-high', displayName: 'Gemini 4 Argon (High)' },
    ]);
  });

  it('AntigravityServerManager.getAvailableModels reads tieredModelIds array and map shapes', async () => {
    const manager = new AntigravityServerManager();
    vi.spyOn(manager as unknown as { rpc: () => Promise<unknown> }, 'rpc').mockResolvedValue({
      response: {
        models: {},
        tieredModelIds: ['gemini-4-argon-medium', 'gemini-3.8-flash-high'],
      },
    });
    const catalog = await manager.getAvailableModels({ httpsPort: 42100, csrf: 't', owned: false });
    expect([...catalog.keys()]).toEqual(['gemini-4-argon-medium', 'gemini-3.8-flash-high']);
    await expect(
      manager.resolveModelEnum('gemini-4-argon-medium', { httpsPort: 42100, csrf: 't', owned: false }),
    ).resolves.toBe('gemini-4-argon-medium');
  });
});

describe('GeminiAntigravityRawParser agy NDJSON and item.completed support', () => {
  const parser = new GeminiAntigravityRawParser();
  const context = {} as ParseContext;

  it('projects agy step_update tool calls into started/completed events and renders item.completed', async () => {
    const started = await parser.parseMessage(
      {
        id: 1,
        sessionId: 's1',
        source: 'antigravity-gemini-agent',
        direction: 'output',
        content: JSON.stringify({
          event: 'step_update',
          step_update: {
            conversation_id: 'c1',
            step_index: 2,
            state: 'ACTIVE',
            step_type: 'tool',
            tool_name: 'write_file',
            tool_info: {
              name: 'write_file',
              parameters: { Path: '/ws/src/index.ts', Content: 'hello' },
            },
          },
        }),
      } as RawMessage,
      context,
    );
    expect(started).toEqual([
      {
        type: 'tool_call_started',
        toolName: 'write_file',
        toolDisplayName: 'write_file',
        arguments: {
          Path: '/ws/src/index.ts',
          Content: 'hello',
          file_path: '/ws/src/index.ts',
          path: '/ws/src/index.ts',
        },
        targetFilePath: '/ws/src/index.ts',
        providerToolCallId: 'c1:step-2',
        createdAt: undefined,
      },
    ]);

    const completed = await parser.parseMessage(
      {
        id: 2,
        sessionId: 's1',
        source: 'antigravity-gemini-agent',
        direction: 'output',
        content: JSON.stringify({
          event: 'step_update',
          step_update: {
            conversation_id: 'c1',
            step_index: 2,
            state: 'DONE',
            step_type: 'tool',
            tool_name: 'write_file',
            tool_info: {
              name: 'write_file',
              parameters: { Path: '/ws/src/index.ts' },
              output: '1 line written',
            },
          },
        }),
      } as RawMessage,
      context,
    );
    expect(completed).toEqual([
      {
        type: 'tool_call_completed',
        providerToolCallId: 'c1:step-2',
        status: 'completed',
        isError: false,
        result: JSON.stringify({
          success: true,
          output: '1 line written',
        }),
      },
    ]);

    const finalReply = await parser.parseMessage(
      {
        id: 3,
        sessionId: 's1',
        source: 'antigravity-gemini-agent',
        direction: 'output',
        content: JSON.stringify({
          type: 'item.completed',
          item: {
            type: 'message',
            role: 'assistant',
            content: [{ type: 'output_text', text: 'All done.' }],
          },
        }),
      } as RawMessage,
      context,
    );
    expect(finalReply).toEqual([
      {
        type: 'assistant_message',
        text: 'All done.',
        createdAt: undefined,
      },
    ]);
  });
});
