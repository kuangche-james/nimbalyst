/**
 * Model catalog for the `antigravity-gemini-agent` provider.
 *
 * The language server DOES enumerate models -- `GetAvailableModels` returns the
 * full catalog and `GetUserStatus.cascadeModelConfigData.clientModelConfigs`
 * returns the subset the signed-in account may actually use. When the extension
 * owned this provider its manifest hardcoded three model ids, and by the time
 * this moved in-tree that list was already stale: the live catalog carried a
 * Gemini 3.6 Flash family the picker never offered. That drift is exactly
 * NIM-1486, so discovery is the primary source and the seed list below is only
 * a fallback.
 *
 * Discovery never spawns the server. Enumerating models is a passive act (the
 * user opened a picker), and starting a ~120MB language server for it would be
 * a surprising side effect -- so when no endpoint is live the seed list is
 * returned and discovery fills in after the first real turn. Same posture as
 * `GrokBuildProvider.listModelIds()`, which falls back to its default when the
 * CLI cannot be reached.
 *
 * The catalog spans several vendors (Antigravity also fronts Anthropic and
 * OpenAI models). This is the *Gemini* provider, so only Google-served entries
 * are offered; surfacing the others here would put the same model behind two
 * unrelated Nimbalyst providers with different billing stories.
 */

import type { AntigravityModelInfo, AntigravityServerManager } from './AntigravityServerManager';

/** `apiProvider` value the language server reports for Google-served models. */
const GOOGLE_API_PROVIDER = 'API_PROVIDER_GOOGLE_GEMINI';

/**
 * Oldest Flash generation still offered in the picker, as `[major, minor]`.
 *
 * The backend retires Flash generations well before the language server stops
 * listing them: `gemini-3-flash-agent` is still in `GetAvailableModels` (as
 * "Gemini 3.5 Flash (High)") on Antigravity 2.12.2, but a turn against it now
 * fails with "Gemini 3.5 Flash is no longer available. Please switch to Gemini
 * 3.7 Flash in the latest version of Antigravity." -- #1519. Entitlement does
 * not catch this either; the account is still entitled to a model the backend
 * refuses to serve. So the catalog alone cannot be trusted and the floor has
 * to be explicit.
 *
 * A floor rather than a fixed list so the next generation is picked up without
 * a code change: when 3.9 Flash lands, discovery offers it and only the
 * retired tiers stay hidden.
 */
const MIN_FAMILY_GENERATIONS: Readonly<Record<string, readonly [major: number, minor: number]>> = {
  Flash: [3, 7],
  Pro: [3, 7],
  Argon: [4, 0],
};

/**
 * Matches Gemini agent labels (Flash, Argon, Pro) and nothing else.
 *
 * Anchored on both ends deliberately. The catalog also carries "Gemini 3.5
 * Flash Lite" and "Gemini 3.1 Flash Image", which are different products that
 * happen to share the prefix -- an unanchored match would drag them in.
 */
const MODEL_LABEL = /^Gemini (\d+)(?:\.(\d+))? (Flash|Argon|Pro)(?: \((?:High|Medium|Low)\))?$/;

/**
 * Parse a Gemini agent label into `{ family, generation }`, or null when the
 * label is not a supported Gemini agent tier.
 */
function parseGeminiModelLabel(
  displayName: string,
): { family: string; generation: [number, number] } | null {
  const match = MODEL_LABEL.exec(displayName);
  if (!match) return null;
  return {
    family: match[3],
    generation: [Number(match[1]), match[2] === undefined ? 0 : Number(match[2])],
  };
}

/**
 * Format a bare `tieredModelIds` key (e.g. `gemini-4-argon-medium` or
 * `gemini-3.8-flash-high`) into its canonical display label when the server
 * omits `displayName`. Returns `''` for non-tier keys (such as `*-tiered`).
 */
export function formatModelDisplayName(key: string): string {
  const match = /^gemini-(\d+(?:\.\d+)?)-(flash|argon|pro)-(high|medium|low)$/i.exec(key.trim());
  if (!match) return '';
  const [, version, familyRaw, tierRaw] = match;
  const family = familyRaw.charAt(0).toUpperCase() + familyRaw.slice(1).toLowerCase();
  const tier = tierRaw.charAt(0).toUpperCase() + tierRaw.slice(1).toLowerCase();
  return `Gemini ${version} ${family} (${tier})`;
}

/**
 * True when `displayName` is a supported Gemini agent model at or above its
 * family's minimum generation floor.
 *
 * Compared component-wise rather than as a decimal so a future "Gemini 3.10
 * Flash" sorts after 3.9 instead of before 3.2.
 */
export function isOfferedFlashModel(displayName: string): boolean {
  const parsed = parseGeminiModelLabel(displayName);
  if (!parsed) return false;
  const minGen = MIN_FAMILY_GENERATIONS[parsed.family];
  if (!minGen) return false;
  const [major, minor] = parsed.generation;
  const [minMajor, minMinor] = minGen;
  return major !== minMajor ? major > minMajor : minor >= minMinor;
}

/**
 * Fallback catalog, used when `agy models` (or the language server) cannot be
 * reached during passive model enumeration.
 */
export const SEED_GEMINI_MODELS: ReadonlyArray<{ key: string; displayName: string }> =
  Object.freeze([
    { key: 'gemini-4-argon-high', displayName: 'Gemini 4 Argon (High)' },
    { key: 'gemini-4-argon-medium', displayName: 'Gemini 4 Argon (Medium)' },
    { key: 'gemini-4-argon-low', displayName: 'Gemini 4 Argon (Low)' },
    { key: 'gemini-3.8-flash-high', displayName: 'Gemini 3.8 Flash (High)' },
    { key: 'gemini-3.8-flash-medium', displayName: 'Gemini 3.8 Flash (Medium)' },
    { key: 'gemini-3.8-flash-low', displayName: 'Gemini 3.8 Flash (Low)' },
  ]);

/** Default model key for a new Gemini session. */
export const DEFAULT_GEMINI_MODEL_KEY = 'gemini-3.8-flash-medium';

/**
 * Strip the `antigravity-gemini-agent:` namespace off a stored model id.
 *
 * The host persists the namespaced form; the language server only knows the
 * bare key. Callers may hand us either.
 */
export function bareGeminiModelKey(raw: string | undefined | null): string {
  if (!raw) return DEFAULT_GEMINI_MODEL_KEY;
  return raw.includes(':') ? raw.split(':').slice(1).join(':') : raw;
}

/**
 * Google-served models the signed-in account may use, newest-looking first.
 *
 * `entitledEnums` is the set of model enums from `clientModelConfigs`. It is
 * the account's entitlement, not the build's catalog: a model present in
 * `GetAvailableModels` but absent here will fail at request time, so offering
 * it would only produce a confusing error after the user picked it. When the
 * entitlement set is empty (an older server that does not report it) the
 * catalog is used unfiltered rather than showing nothing.
 *
 * Also restricted to Flash agents at or above `MIN_FLASH_GENERATION`. Retired
 * tiers stay in the live catalog and stay entitled long after the backend
 * stops serving them, so neither signal can be relied on -- see #1519.
 *
 * Existing sessions are not rewritten. A session row that persisted a retired
 * key keeps resolving, because resolution reads the stored key and never
 * consults this list; such a session will keep failing at request time until
 * the user picks a current model.
 */
export function selectGeminiModels(
  catalog: Map<string, AntigravityModelInfo>,
  entitledEnums: ReadonlySet<string>,
): Array<{ key: string; displayName: string }> {
  const out: Array<{ key: string; displayName: string }> = [];
  for (const info of catalog.values()) {
    if (info.apiProvider && info.apiProvider !== GOOGLE_API_PROVIDER) continue;
    if (!info.apiProvider && !info.key.toLowerCase().startsWith('gemini-')) continue;
    const enumOrKey = info.enum ?? info.key;
    if (entitledEnums.size > 0 && !entitledEnums.has(enumOrKey) && !entitledEnums.has(info.key)) {
      continue;
    }
    const displayName = info.displayName || formatModelDisplayName(info.key);
    if (!displayName) continue;
    if (!isOfferedFlashModel(displayName)) continue;
    out.push({ key: info.key, displayName });
  }
  return out.sort((a, b) => a.displayName.localeCompare(b.displayName));
}

/**
 * Parse the tabular output of `agy models` and return Google-served Gemini
 * entries (`gemini-*`). The CLI separates the id from the display name with a
 * single tab (`gemini-4-argon-medium\tGemini 4 Argon (Medium)`); older notes
 * showed two or more spaces, so both are accepted. A pattern that only took
 * spaces parsed nothing on a real Windows install and fell back to the seed
 * list on every read (#1653).
 */
export function parseAgyModelList(
  output: string,
): Array<{ key: string; displayName: string }> {
  const seen = new Set<string>();
  const out: Array<{ key: string; displayName: string }> = [];
  for (const rawLine of output.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    const match = /^([a-z0-9][a-z0-9._-]*)(?:\t+| {2,})\s*(.+?)\s*$/i.exec(line);
    if (!match) continue;
    const [, key, displayName] = match;
    if (!key.toLowerCase().startsWith('gemini-')) continue;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ key, displayName });
  }
  return out;
}

/** Read the account's entitled model enums out of a raw GetUserStatus payload. */
export function entitledModelEnums(userStatus: unknown): Set<string> {
  const enums = new Set<string>();
  const configs = (userStatus as {
    cascadeModelConfigData?: { clientModelConfigs?: Array<{ modelOrAlias?: { model?: unknown } }> };
  } | null | undefined)?.cascadeModelConfigData?.clientModelConfigs;
  if (!Array.isArray(configs)) return enums;
  for (const config of configs) {
    const name = config?.modelOrAlias?.model;
    if (typeof name === 'string' && name) enums.add(name);
  }
  return enums;
}

/**
 * Discover the model list, falling back to the seed when the server is not
 * already running or either RPC fails.
 */
export async function discoverGeminiModels(
  server: AntigravityServerManager,
): Promise<Array<{ key: string; displayName: string }>> {
  const endpoint = server.currentEndpoint();
  if (!endpoint) return [...SEED_GEMINI_MODELS];
  try {
    const [catalog, userStatus] = await Promise.all([
      server.getAvailableModels(endpoint),
      server.getUserStatus(endpoint).catch(() => null),
    ]);
    const models = selectGeminiModels(catalog, entitledModelEnums(userStatus));
    return models.length > 0 ? models : [...SEED_GEMINI_MODELS];
  } catch {
    return [...SEED_GEMINI_MODELS];
  }
}
