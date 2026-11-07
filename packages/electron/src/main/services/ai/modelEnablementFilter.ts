import { isClaudeCodeFamily } from '@nimbalyst/runtime/ai/server/types';
import { isHeadlessAgentAvailable } from './headlessAgentAvailability';

export interface ProviderEnablement {
  enabled: boolean;
  /**
   * Optional per-provider allow-list. Empty or undefined means "allow all models
   * this provider offers" — the catalog is the source of truth. A non-empty list
   * restricts the picker to exactly those ids (plus the family conveniences below).
   */
  models?: string[];
  /**
   * Optional per-provider hide-list (denylist). Any model id here is hidden from
   * the picker regardless of the allow-list. Empty/undefined hides nothing, so a
   * newly shipped variant always appears until the user explicitly hides it — this
   * is the primary "trim the picker" control for the Claude Code family, which
   * keeps `models` empty (allow-all) on purpose (NIM-1486).
   */
  hiddenModels?: string[];
}

export interface FilterableModel {
  id: string;
  provider: string;
}

/**
 * Providers that are on before the user has touched anything in Settings.
 *
 * Only `claude-code` (the SDK-backed Claude Agent) ships enabled — it is the
 * app's default agent and needs no configuration. Everything else, including
 * `claude-code-cli`, stays off until explicitly enabled. The CLI provider is
 * purely a preference for driving the terminal `claude` binary; a Claude
 * subscription works fine through the default agent, so it does not belong in
 * the picker for people who never asked for it.
 */
const DEFAULT_ENABLED_PROVIDERS: ReadonlySet<string> = new Set(['claude-code']);

/**
 * Providers whose default is decided by the machine rather than by a constant:
 * on when their CLI is installed and signed in, off otherwise.
 *
 * This is a *default*, never a write. The `??` below means an explicit toggle
 * always wins, so detection can never override a user who turned the provider
 * off — and because nothing is persisted, there is no first-boot decision to
 * get stuck on. See `headlessAgentAvailability.ts`.
 */
const DETECTED_DEFAULT_PROVIDERS: ReadonlySet<string> = new Set([
  'grok-build',
  'cursor-agent',
  // Gemini detects on the presence of the Antigravity app rather than on a
  // sign-in probe; `headlessAgentAvailability.ts` explains why sign-in cannot
  // be asked for cheaply. The `??` above still means an explicit toggle wins.
  'antigravity-gemini-agent',
]);

/**
 * Resolve a provider's enabled state, applying the default for an absent
 * setting. Shared by every "is this provider on?" read so the settings toggle
 * and the model picker cannot disagree about what "never configured" means.
 */
export function resolveProviderEnabled(
  provider: string,
  config: { enabled?: boolean } | undefined,
): boolean {
  if (config?.enabled !== undefined) return config.enabled;
  if (DETECTED_DEFAULT_PROVIDERS.has(provider)) return isHeadlessAgentAvailable(provider);
  return DEFAULT_ENABLED_PROVIDERS.has(provider);
}

/**
 * Providers that fetched models but had every one of them filtered out of the
 * picker. That is the state worth logging: Settings shows the provider
 * enabled, the picker shows nothing, and no error is raised anywhere (#1653).
 * Every other `ai:getModels` call is routine and stays out of `main.log`.
 */
export function findProvidersWithAllModelsHidden(
  fetched: readonly FilterableModel[],
  shown: readonly FilterableModel[],
): Array<{ provider: string; fetched: number }> {
  const shownProviders = new Set(shown.map((model) => model.provider));
  const counts = new Map<string, number>();
  for (const model of fetched) {
    if (shownProviders.has(model.provider)) continue;
    counts.set(model.provider, (counts.get(model.provider) ?? 0) + 1);
  }
  return [...counts].map(([provider, count]) => ({ provider, fetched: count }));
}

/**
 * Drop allow-lists that no longer name anything the provider offers.
 *
 * An allow-list is a snapshot of ids from the catalog the user saw when they
 * curated it. When a provider's catalog turns over completely (Gemini moved
 * from `gemini-3.5-flash-*` to `gemini-4-argon-*` / `gemini-3.8-flash-*`), a
 * stale list hides every model and the provider silently vanishes from the
 * picker while Settings still shows it enabled (#1653). A list that matches
 * nothing cannot be expressing a preference among the current models, so it
 * is treated as absent (allow all). A list that still matches at least one id
 * is honoured as-is; pruning individual stale ids is left to the user.
 *
 * The Claude Code family is skipped: its list carries sentinel entries that
 * `isModelEnabled` interprets, so a catalog comparison would misjudge it.
 *
 * Pure: returns a new map and the ids of the providers whose list was ignored,
 * so the caller can log them.
 */
export function dropStaleAllowLists<T extends Record<string, ProviderEnablement | undefined>>(
  entries: T,
  catalog: readonly FilterableModel[],
): { entries: T; stale: string[] } {
  const idsByProvider = new Map<string, Set<string>>();
  for (const model of catalog) {
    let ids = idsByProvider.get(model.provider);
    if (!ids) {
      ids = new Set();
      idsByProvider.set(model.provider, ids);
    }
    ids.add(model.id);
  }

  const stale: string[] = [];
  const next: Record<string, ProviderEnablement | undefined> = { ...entries };
  for (const [provider, entry] of Object.entries(entries)) {
    const list = entry?.models;
    if (!entry || !list || list.length === 0 || isClaudeCodeFamily(provider)) continue;
    const offered = idsByProvider.get(provider);
    // A provider that fetched nothing has no catalog to compare against; leave
    // its list alone rather than guessing.
    if (!offered || offered.size === 0) continue;
    if (list.some((id) => offered.has(id))) continue;
    stale.push(provider);
    next[provider] = { ...entry, models: undefined };
  }
  return { entries: next as T, stale };
}

/**
 * Single gate deciding whether a catalog model reaches the picker. Extracted from
 * the `ai:getModels` handler so the behavior that once silently hid Fable 5 is
 * unit-tested (NIM-1486).
 *
 * The invariant that prevents recurrence: an empty/undefined allow-list means
 * "show everything". Since nothing curates the claude-code list, it stays empty
 * and every shipped variant is always visible. The special-cases below apply to
 * the whole Claude Code family (`claude-code` AND `claude-code-cli`) so the CLI
 * provider can't drift the way the SDK provider did.
 */
export function isModelEnabled(
  model: FilterableModel,
  entry: ProviderEnablement | undefined,
): boolean {
  if (!entry?.enabled) return false;

  // Denylist wins: a hidden id is never shown, even if it's also allow-listed.
  // Rows map 1:1 to picker rows, so hiding is per exact id (a base variant and
  // its -1m row are hidden independently).
  if (entry.hiddenModels && entry.hiddenModels.includes(model.id)) return false;

  const list = entry.models;
  if (list && list.length > 0) {
    if (isClaudeCodeFamily(model.provider)) {
      // Sentinel: the provider id itself in the list means "all of this provider".
      if (list.includes(model.provider)) return true;
      // Selecting a base variant implicitly enables its 1M extended-context row
      // (e.g. selecting `claude-code:sonnet` also surfaces `claude-code:sonnet-1m`).
      if (model.id.endsWith('-1m')) {
        const baseId = model.id.replace(/-1m$/, '');
        if (list.includes(baseId)) return true;
      }
    }
    return list.includes(model.id);
  }

  // Empty/undefined allow-list => allow all models for this provider.
  return true;
}
