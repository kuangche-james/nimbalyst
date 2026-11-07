import React, { useCallback } from 'react';

/**
 * Custom event name used to ask the renderer to open the project trust
 * chooser (the same "Allow edits only / Agent-verified / Allow everything"
 * menu the shield icon shows). Listened for by `App.tsx`. The detail carries
 * the session's workspace so a host with several projects open can target the
 * right one.
 */
export const CHANGE_PERMISSION_MODE_EVENT = 'nimbalyst:change-permission-mode';

export interface ChangePermissionModeEventDetail {
  workspacePath?: string;
}

/**
 * Recognise the up-front refusal a headless CLI provider (Gemini via `agy`,
 * Cursor Agent) emits when the project's permission mode would require
 * per-tool prompts the CLI cannot deliver. Matched on the one clause every
 * such message shares, so a provider can word the rest for its own CLI.
 */
export function isPermissionModeRequiredError(text: string | null | undefined): boolean {
  if (!text) return false;
  return /no per-tool approval (?:for Nimbalyst )?to intercept/i.test(text);
}

const injectPermissionModeRequiredWidgetStyles = () => {
  const styleId = 'permission-mode-required-widget-styles';
  if (document.getElementById(styleId)) return;

  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = `
    .permission-mode-required-widget {
      background-color: color-mix(in srgb, var(--nim-warning) 8%, transparent);
      border: 1px solid color-mix(in srgb, var(--nim-warning) 30%, transparent);
    }
  `;
  document.head.appendChild(style);
};

interface PermissionModeRequiredWidgetProps {
  /** The provider's refusal text, shown in full so the reason is not lost. */
  message: string;
  workspacePath?: string;
}

/**
 * Actionable replacement for the generic red error row when a headless CLI
 * provider refuses a turn because the project asks before tools run. The
 * button opens the trust chooser; the user picks a mode and resends.
 */
export const PermissionModeRequiredWidget: React.FC<PermissionModeRequiredWidgetProps> = ({
  message,
  workspacePath,
}) => {
  React.useEffect(() => {
    injectPermissionModeRequiredWidgetStyles();
  }, []);

  const handleClick = useCallback(() => {
    const detail: ChangePermissionModeEventDetail = { workspacePath };
    window.dispatchEvent(new CustomEvent(CHANGE_PERMISSION_MODE_EVENT, { detail }));
  }, [workspacePath]);

  return (
    <div
      className="permission-mode-required-widget my-4 p-4 rounded-lg flex flex-col gap-3"
      data-testid="permission-mode-required-widget"
    >
      <div className="text-sm font-medium text-[var(--nim-text)]">
        This agent needs a permission mode that does not prompt
      </div>
      <p className="text-[13px] text-[var(--nim-text-muted)] leading-relaxed select-text">
        {message}
      </p>
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={handleClick}
          className="inline-flex items-center justify-center py-2 px-4 rounded-md text-sm font-medium cursor-pointer transition-all bg-[var(--nim-primary)] text-white hover:bg-[var(--nim-primary-hover)]"
          data-testid="permission-mode-required-change"
        >
          Change permission mode
        </button>
      </div>
    </div>
  );
};
