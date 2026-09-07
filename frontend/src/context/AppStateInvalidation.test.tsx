import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AppStateProvider, useAppState } from "@/context/AppStateContext";
import { useDeleteTransaction } from "@/hooks/useTransactions";
import { useDeleteHolding } from "@/hooks/useHoldings";

/**
 * Regression test for a real bug reported via manual testing: deleting the
 * last real transaction (or holding) correctly flips the backend's
 * data_mode back to EMPTY (AppStateService.maybe_transition_to_empty,
 * already covered by backend tests), but the frontend kept showing REAL --
 * blocking "Load demo data" -- until something unrelated happened to
 * refetch /api/demo/status. Root cause: useDeleteTransaction and
 * useDeleteHolding invalidated their own list + dashboard queries on
 * success, but never APP_STATE_QUERY_KEY, unlike their `create` siblings.
 *
 * These tests mount the REAL AppStateProvider (not a mock of it) alongside
 * the real delete hooks, and prove the mode/canLoadDemo the provider
 * exposes actually updates after a delete -- i.e. that the fix is the
 * missing invalidation, not just that invalidateQueries was *called*.
 */
vi.mock("@/api/transactions", () => ({
  listTransactions: vi.fn(),
  createTransaction: vi.fn(),
  updateTransaction: vi.fn(),
  deleteTransaction: vi.fn(),
}));

vi.mock("@/api/holdings", () => ({
  listHoldings: vi.fn(),
  createHolding: vi.fn(),
  updateHolding: vi.fn(),
  deleteHolding: vi.fn(),
  refreshPrices: vi.fn(),
}));

vi.mock("@/api/health", () => ({
  getHealth: vi.fn(),
  getDemoStatus: vi.fn(),
}));

vi.mock("@/api/demo", () => ({
  loadDemo: vi.fn(),
  clearDemo: vi.fn(),
  clearRealData: vi.fn(),
}));

function makeWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <AppStateProvider>{children}</AppStateProvider>
      </QueryClientProvider>
    );
  }
  return Wrapper;
}

describe("delete mutations invalidate app-state (real-data-deleted-to-zero regression)", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("deleting the last real transaction refetches app-state and unlocks Load demo data", async () => {
    const { getDemoStatus } = await import("@/api/health");
    const { deleteTransaction } = await import("@/api/transactions");
    vi.mocked(getDemoStatus)
      .mockResolvedValueOnce({ mode: "REAL", can_load_demo: false })
      .mockResolvedValue({ mode: "EMPTY", can_load_demo: true });
    vi.mocked(deleteTransaction).mockResolvedValue({ id: 1, deleted: true });

    const Wrapper = makeWrapper();
    const { result } = renderHook(
      () => ({ appState: useAppState(), deleteMutation: useDeleteTransaction() }),
      { wrapper: Wrapper },
    );

    await waitFor(() => expect(result.current.appState.mode).toBe("REAL"));
    expect(result.current.appState.canLoadDemo).toBe(false);

    result.current.deleteMutation.mutate(1);

    await waitFor(() => expect(deleteTransaction).toHaveBeenCalledWith(1));
    await waitFor(() => expect(result.current.appState.mode).toBe("EMPTY"));
    expect(result.current.appState.canLoadDemo).toBe(true);
    // The bug's signature: a SECOND call to getDemoStatus after the delete,
    // not just the initial mount fetch.
    expect(getDemoStatus).toHaveBeenCalledTimes(2);
  });

  it("deleting the last real holding refetches app-state and unlocks Load demo data", async () => {
    const { getDemoStatus } = await import("@/api/health");
    const { deleteHolding } = await import("@/api/holdings");
    vi.mocked(getDemoStatus)
      .mockResolvedValueOnce({ mode: "REAL", can_load_demo: false })
      .mockResolvedValue({ mode: "EMPTY", can_load_demo: true });
    vi.mocked(deleteHolding).mockResolvedValue({ id: 7, deleted: true });

    const Wrapper = makeWrapper();
    const { result } = renderHook(
      () => ({ appState: useAppState(), deleteMutation: useDeleteHolding() }),
      { wrapper: Wrapper },
    );

    await waitFor(() => expect(result.current.appState.mode).toBe("REAL"));

    result.current.deleteMutation.mutate(7);

    await waitFor(() => expect(deleteHolding).toHaveBeenCalledWith(7));
    await waitFor(() => expect(result.current.appState.mode).toBe("EMPTY"));
    expect(result.current.appState.canLoadDemo).toBe(true);
    expect(getDemoStatus).toHaveBeenCalledTimes(2);
  });
});
