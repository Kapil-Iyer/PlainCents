import * as React from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { getDemoStatus } from "@/api/health";
import type { DataMode } from "@/types/common";

interface AppStateValue {
  mode: DataMode;
  canLoadDemo: boolean;
  isLoading: boolean;
  /** Call after any mutation that may change data_mode (import confirm,
   * manual transaction create, holding create, demo load/clear). */
  refresh: () => void;
}

const AppStateContext = React.createContext<AppStateValue | null>(null);

export const APP_STATE_QUERY_KEY = ["demo-status"] as const;

/**
 * TRD §9.7: fetches /api/demo/status once on load, exposes `mode` to the
 * rest of the app. Defaults to EMPTY while loading so nothing briefly
 * flashes a DEMO banner before the first response arrives.
 */
export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const queryClient = useQueryClient();
  const { data, isLoading } = useQuery({
    queryKey: APP_STATE_QUERY_KEY,
    queryFn: getDemoStatus,
  });

  const refresh = React.useCallback(() => {
    queryClient.invalidateQueries({ queryKey: APP_STATE_QUERY_KEY });
  }, [queryClient]);

  const value: AppStateValue = {
    mode: data?.mode ?? "EMPTY",
    canLoadDemo: data?.can_load_demo ?? true,
    isLoading,
    refresh,
  };

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState(): AppStateValue {
  const ctx = React.useContext(AppStateContext);
  if (!ctx) throw new Error("useAppState must be used within AppStateProvider");
  return ctx;
}
