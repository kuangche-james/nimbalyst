import React, { useCallback, useEffect, useState } from 'react';
import { ProviderConfig } from '../../Settings/SettingsView';
import { SettingsToggle } from '../SettingsToggle';
import { AlphaBadge, SETTINGS_ALPHA_TOOLTIP } from '../../common/AlphaBadge';

/**
 * Settings panel for the Gemini agent.
 *
 * Deliberately not `HeadlessCliProviderPanel`: that panel's whole middle is
 * "install this CLI with this command, then run this login command", and
 * neither applies here. Antigravity ships the `agy` CLI alongside the desktop
 * application; you install it and sign in once, and Nimbalyst runs Gemini
 * through that CLI.
 *
 * What this replaces is worth naming, because it is the reason this file
 * exists. While Gemini shipped as an extension, its pane opened with a "Native
 * code grant required" banner asking the user to grant permission to execute
 * native code, listed a `nimbalyst-database-write` catalog permission, and
 * pointed at Installed Extensions to revoke it — first-party code asking the
 * user for consent to run itself. There is no grant here, because there is
 * nothing to grant: this is Nimbalyst's own code in Nimbalyst's own process,
 * held to the same bar as every other built-in provider.
 *
 * So the pane answers the three questions every other provider pane answers:
 * is the tool installed, are you signed in, and what happens to your files.
 */

type InstallStatus = 'checking' | 'installed' | 'not-installed';

interface GeminiPanelProps {
  config: ProviderConfig;
  onToggle: (enabled: boolean) => void;
}

const ANTIGRAVITY_URL = 'https://antigravity.google';

export function GeminiPanel({ config, onToggle }: GeminiPanelProps) {
  const [status, setStatus] = useState<InstallStatus>('checking');
  const [installPath, setInstallPath] = useState<string | null>(null);

  const check = useCallback(async () => {
    setStatus('checking');
    try {
      const availability = await window.electronAPI.aiGetHeadlessAgentAvailability?.();
      const gemini = availability?.['antigravity-gemini-agent'];
      setInstallPath(gemini?.executablePath ?? null);
      setStatus(gemini?.installed ? 'installed' : 'not-installed');
    } catch {
      setStatus('not-installed');
    }
  }, []);

  useEffect(() => {
    void check();
  }, [check]);

  return (
    <div className="provider-panel gemini-provider-panel flex flex-col">
      <div className="provider-panel-header mb-6 pb-4 border-b border-[var(--nim-border)]">
        <h3 className="provider-panel-title text-xl font-semibold leading-tight mb-2 text-[var(--nim-text)] flex items-center gap-2">
          Gemini
          <AlphaBadge size="sm" tooltip={SETTINGS_ALPHA_TOOLTIP} />
        </h3>
        <p className="provider-panel-description text-sm leading-relaxed text-[var(--nim-text-muted)]">
          Google&apos;s Gemini models, run through the official Antigravity CLI (<code>agy</code>).
          Uses the Google account you are already signed into there.
        </p>
      </div>

      <div className="provider-panel-section py-4 mb-4 border-b border-[var(--nim-border)]">
        <h4 className="provider-panel-section-title text-base font-semibold mb-3 text-[var(--nim-text)]">
          Antigravity
        </h4>

        {status === 'checking' && (
          <p className="text-[13px] text-[var(--nim-text-muted)]">Looking for Antigravity...</p>
        )}

        {status === 'installed' && (
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[var(--nim-success)] shrink-0" />
              <span className="text-[13px] text-[var(--nim-text)]">Installed</span>
            </div>
            {installPath && (
              <code className="block text-[12px] text-[var(--nim-code-text)] bg-[var(--nim-code-bg)] px-3 py-2 rounded mt-2 select-text break-all">
                {installPath}
              </code>
            )}
          </div>
        )}

        {status === 'not-installed' && (
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="w-2 h-2 rounded-full bg-[var(--nim-warning)] shrink-0" />
              <span className="text-[13px] text-[var(--nim-text)]">Not found</span>
            </div>
            <p className="text-[13px] text-[var(--nim-text-muted)] mb-3 leading-relaxed">
              Install the Antigravity CLI (<code>agy</code>) and sign in once. Nimbalyst does
              not install it for you. The Antigravity IDE alone is not enough: Nimbalyst runs
              Gemini through the CLI.
            </p>
            <button
              className="inline-flex items-center justify-center py-2 px-4 rounded-md text-sm font-medium cursor-pointer transition-all bg-[var(--nim-surface)] text-[var(--nim-text)] border border-[var(--nim-border)] hover:bg-[var(--nim-surface-hover)]"
              onClick={() => void check()}
            >
              Check again
            </button>
          </div>
        )}

        <p className="text-[13px] text-[var(--nim-text-muted)] mt-3 leading-relaxed">
          See{' '}
          <a
            href={ANTIGRAVITY_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[var(--nim-primary)] hover:underline"
          >
            antigravity.google
          </a>
          {' '}for downloads and sign-in.
        </p>
      </div>

      <SettingsToggle
        variant="enable"
        name="Enable Gemini"
        checked={config.enabled || false}
        onChange={onToggle}
      />

      {status === 'installed' && (
        <p className="text-[13px] text-[var(--nim-text-muted)] mt-2 leading-relaxed">
          On by default because Antigravity is installed. Turning it off here is remembered.
        </p>
      )}

      {config.enabled && (
        <>
          <div className="provider-panel-section py-4 mb-4 border-b border-[var(--nim-border)]">
            <h4 className="provider-panel-section-title text-base font-semibold mb-3 text-[var(--nim-text)]">Authentication</h4>
            <p className="text-[13px] text-[var(--nim-text-muted)] mb-3 leading-relaxed">
              Sign in once with <code>agy</code> or inside Antigravity. The CLI uses that
              login, so there is nothing to enter here.
            </p>
            <p className="text-[13px] text-[var(--nim-text-muted)]">
              No API key is required, and Nimbalyst never reads one from your environment.
            </p>
          </div>

          <div className="provider-panel-section py-4 mb-4 border-b border-[var(--nim-border)]">
            <h4 className="provider-panel-section-title text-base font-semibold mb-3 text-[var(--nim-text)]">Permissions</h4>
            <p className="text-[13px] text-[var(--nim-text-muted)] leading-relaxed">
              The CLI runs headless and cannot pause to ask, so Gemini only runs in projects
              set to &quot;Allow edits only&quot; or &quot;Allow everything&quot; (Project Settings &gt;
              Agent Permissions). In either mode Gemini runs shell commands as well as file
              edits without asking; the &quot;Allow edits only&quot; limits apply to other agents,
              not to Gemini. A project set to &quot;Ask every time&quot; refuses the turn up front.
            </p>
          </div>

          <div className="provider-panel-section py-4 mb-4 border-b border-[var(--nim-border)]">
            <h4 className="provider-panel-section-title text-base font-semibold mb-3 text-[var(--nim-text)]">File tracking</h4>
            <p className="text-[13px] text-[var(--nim-text-muted)] leading-relaxed">
              The Antigravity CLI performs this agent&apos;s writes. Nimbalyst reads each file
              as the CLI reports the edit starting and finishing, so diff review has a real
              before-and-after for edits it sees in time; when the CLI has already written
              the file, Nimbalyst reconstructs the previous contents where that is
              unambiguous and otherwise relies on its own file watcher. Deletes and renames
              go through shell commands; the watcher catches those too, so one can
              occasionally be attributed to the wrong turn.
            </p>
          </div>

          <div className="provider-panel-section py-4 mb-4 border-b border-[var(--nim-border)] last:border-b-0 last:mb-0 last:pb-0">
            <h4 className="provider-panel-section-title text-base font-semibold mb-3 text-[var(--nim-text)]">Context usage</h4>
            <p className="text-[13px] text-[var(--nim-text-muted)] leading-relaxed">
              The Antigravity CLI reports token counts at the end of each turn, so the
              context meter reflects what Gemini actually used. Remaining quota is shown in
              the usage indicator.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
