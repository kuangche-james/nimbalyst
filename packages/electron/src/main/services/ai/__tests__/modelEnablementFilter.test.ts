import { describe, it, expect } from 'vitest';
import { CLAUDE_CODE_VARIANTS, ModelIdentifier } from '@nimbalyst/runtime/ai/server/types';
import {
  dropStaleAllowLists,
  findProvidersWithAllModelsHidden,
  isModelEnabled,
  resolveProviderEnabled,
  type ProviderEnablement,
} from '../modelEnablementFilter';

/**
 * The gate that once hid Fable 5 from the picker (NIM-1486). These lock in the
 * "can't happen again" invariant: an empty/undefined allow-list shows every
 * shipped variant, and the family conveniences (sentinel, -1m) work for the CLI
 * provider too — not just the SDK provider.
 */
describe('isModelEnabled', () => {
  const id = (provider: string, v: string) => ModelIdentifier.create(provider as any, v).combined;

  it('disabled provider hides everything', () => {
    expect(isModelEnabled({ id: 'claude-code:fable', provider: 'claude-code' }, { enabled: false })).toBe(false);
  });

  it('empty allow-list shows every shipped variant (incl. fable) for both families', () => {
    for (const provider of ['claude-code', 'claude-code-cli']) {
      for (const variant of CLAUDE_CODE_VARIANTS) {
        const base = { id: id(provider, variant), provider };
        expect(isModelEnabled(base, { enabled: true })).toBe(true); // undefined list
        expect(isModelEnabled(base, { enabled: true, models: [] })).toBe(true); // empty list
        // ...and the 1M row.
        const oneM = { id: id(provider, `${variant}-1m`), provider };
        expect(isModelEnabled(oneM, { enabled: true, models: [] })).toBe(true);
      }
    }
  });

  it('a non-empty list still restricts to listed ids', () => {
    const entry = { enabled: true, models: ['claude-code:opus'] };
    expect(isModelEnabled({ id: 'claude-code:opus', provider: 'claude-code' }, entry)).toBe(true);
    expect(isModelEnabled({ id: 'claude-code:fable', provider: 'claude-code' }, entry)).toBe(false);
  });

  it('selecting a base variant also surfaces its 1M row — for the CLI provider too', () => {
    // Regression: the old inline filter special-cased only `claude-code`, so a
    // claude-code-cli allow-list would have dropped the -1m row.
    const entry = { enabled: true, models: ['claude-code-cli:sonnet'] };
    expect(isModelEnabled({ id: 'claude-code-cli:sonnet', provider: 'claude-code-cli' }, entry)).toBe(true);
    expect(isModelEnabled({ id: 'claude-code-cli:sonnet-1m', provider: 'claude-code-cli' }, entry)).toBe(true);
  });

  it('the provider-id sentinel means "all of this provider" for both families', () => {
    expect(
      isModelEnabled({ id: 'claude-code:fable', provider: 'claude-code' }, { enabled: true, models: ['claude-code'] }),
    ).toBe(true);
    expect(
      isModelEnabled(
        { id: 'claude-code-cli:fable', provider: 'claude-code-cli' },
        { enabled: true, models: ['claude-code-cli'] },
      ),
    ).toBe(true);
  });

  describe('hiddenModels (denylist)', () => {
    it('hides exactly the listed ids and shows everything else', () => {
      const entry = { enabled: true, hiddenModels: ['claude-code:sonnet'] };
      expect(isModelEnabled({ id: 'claude-code:sonnet', provider: 'claude-code' }, entry)).toBe(false);
      expect(isModelEnabled({ id: 'claude-code:opus', provider: 'claude-code' }, entry)).toBe(true);
    });

    it('is independent per exact id — hiding a base variant leaves its 1M row visible', () => {
      // Denylist rows map 1:1 to picker rows; base and -1m are toggled separately.
      const entry = { enabled: true, hiddenModels: ['claude-code:opus'] };
      expect(isModelEnabled({ id: 'claude-code:opus', provider: 'claude-code' }, entry)).toBe(false);
      expect(isModelEnabled({ id: 'claude-code:opus-1m', provider: 'claude-code' }, entry)).toBe(true);
    });

    it('an empty/undefined hidden set shows everything', () => {
      expect(isModelEnabled({ id: 'claude-code:opus', provider: 'claude-code' }, { enabled: true, hiddenModels: [] })).toBe(true);
      expect(isModelEnabled({ id: 'claude-code:opus', provider: 'claude-code' }, { enabled: true })).toBe(true);
    });

    it('hidden wins over the allow-list — a hidden id is never shown even if allow-listed', () => {
      const entry = { enabled: true, models: ['claude-code:opus'], hiddenModels: ['claude-code:opus'] };
      expect(isModelEnabled({ id: 'claude-code:opus', provider: 'claude-code' }, entry)).toBe(false);
    });

    it('works for the CLI provider so its set can be trimmed independently', () => {
      const entry = { enabled: true, hiddenModels: ['claude-code-cli:haiku'] };
      expect(isModelEnabled({ id: 'claude-code-cli:haiku', provider: 'claude-code-cli' }, entry)).toBe(false);
      expect(isModelEnabled({ id: 'claude-code:haiku', provider: 'claude-code' }, entry)).toBe(true);
    });
  });
});

/**
 * The terminal-CLI provider used to default ON, so its models appeared in the
 * picker for everyone even though the settings toggle rendered off. Using a
 * Claude subscription never required it — Claude Agent already runs on the
 * subscription — so an absent setting must mean disabled.
 */
describe('dropStaleAllowLists', () => {
  const gemini = 'antigravity-gemini-agent';
  const catalog = [
    { id: `${gemini}:gemini-4-argon-medium`, provider: gemini },
    { id: `${gemini}:gemini-3.8-flash-low`, provider: gemini },
    { id: 'openai-codex:gpt-6-sol', provider: 'openai-codex' },
  ];

  it('ignores an allow-list naming only ids the provider no longer offers', () => {
    // The Windows report behind #1653: a Gemini list saved against the old
    // gemini-3.5 catalog hid all six current models while Settings said enabled.
    const entries: Record<string, ProviderEnablement> = {
      [gemini]: {
        enabled: true,
        models: [`${gemini}:gemini-3-flash-agent`, `${gemini}:gemini-3.5-flash-low`],
      },
    };
    const { entries: next, stale } = dropStaleAllowLists(entries, catalog);
    expect(stale).toEqual([gemini]);
    expect(next[gemini]?.models).toBeUndefined();
    expect(catalog.filter((m) => isModelEnabled(m, next[m.provider]))).toHaveLength(2);
    // Input is not mutated.
    expect(entries[gemini].models).toHaveLength(2);
  });

  it('keeps an allow-list that still matches at least one offered model', () => {
    const entries: Record<string, ProviderEnablement> = {
      [gemini]: {
        enabled: true,
        models: [`${gemini}:gemini-3.8-flash-low`, `${gemini}:gemini-3.5-flash-low`],
      },
    };
    const { entries: next, stale } = dropStaleAllowLists(entries, catalog);
    expect(stale).toEqual([]);
    expect(next[gemini]?.models).toEqual(entries[gemini].models);
    expect(catalog.filter((m) => isModelEnabled(m, next[m.provider])).map((m) => m.id)).toEqual([
      `${gemini}:gemini-3.8-flash-low`,
    ]);
  });

  it('leaves lists alone for providers that fetched nothing, and for the Claude Code family', () => {
    const entries: Record<string, ProviderEnablement> = {
      'opencode': { enabled: true, models: ['opencode:anything'] },
      'claude-code': { enabled: true, models: ['claude-code'] },
    };
    const { entries: next, stale } = dropStaleAllowLists(entries, catalog);
    expect(stale).toEqual([]);
    expect(next).toEqual(entries);
  });
});

describe('findProvidersWithAllModelsHidden', () => {
  it('names a provider whose fetched models were all filtered out of the picker', () => {
    // The #1653 report: Gemini fetched six models, a stale allow-list hid them
    // all, and nothing in main.log said so.
    const fetched = [
      { id: 'claude-code:opus', provider: 'claude-code' },
      { id: 'antigravity-gemini-agent:gemini-4-argon-medium', provider: 'antigravity-gemini-agent' },
      { id: 'antigravity-gemini-agent:gemini-3.8-flash-low', provider: 'antigravity-gemini-agent' },
    ];
    const shown = [{ id: 'claude-code:opus', provider: 'claude-code' }];

    expect(findProvidersWithAllModelsHidden(fetched, shown)).toEqual([
      { provider: 'antigravity-gemini-agent', fetched: 2 },
    ]);
  });
});

describe('resolveProviderEnabled', () => {
  it('claude-code-cli is off until explicitly enabled', () => {
    expect(resolveProviderEnabled('claude-code-cli', undefined)).toBe(false);
    expect(resolveProviderEnabled('claude-code-cli', {})).toBe(false);
    expect(resolveProviderEnabled('claude-code-cli', { enabled: true })).toBe(true);
    expect(resolveProviderEnabled('claude-code-cli', { enabled: false })).toBe(false);
  });

  it('claude-code stays on by default — it is the app default agent', () => {
    expect(resolveProviderEnabled('claude-code', undefined)).toBe(true);
    expect(resolveProviderEnabled('claude-code', { enabled: false })).toBe(false);
  });

  it('every other provider is opt-in', () => {
    for (const provider of ['claude', 'openai', 'lmstudio', 'opencode']) {
      expect(resolveProviderEnabled(provider, undefined)).toBe(false);
    }
  });
});
