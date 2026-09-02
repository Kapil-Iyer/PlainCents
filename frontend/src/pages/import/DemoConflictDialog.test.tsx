import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/utils";

import { DemoConflictDialog } from "@/pages/import/DemoConflictDialog";

vi.mock("@/api/demo", () => ({
  loadDemo: vi.fn(),
  clearDemo: vi.fn(),
}));

describe("DemoConflictDialog", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("clears demo data and calls onRetry on confirm — no more 501 handling", async () => {
    const user = userEvent.setup();
    const { clearDemo } = await import("@/api/demo");
    vi.mocked(clearDemo).mockResolvedValue({ mode: "EMPTY", cleared: true, summary: {} });
    const onRetry = vi.fn();
    const onOpenChange = vi.fn();

    renderWithProviders(
      <DemoConflictDialog open onOpenChange={onOpenChange} onRetry={onRetry} />,
    );

    await user.click(screen.getByRole("button", { name: /Clear demo data & retry/ }));

    await waitFor(() => expect(clearDemo).toHaveBeenCalled());
    await waitFor(() => expect(onRetry).toHaveBeenCalled());
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("shows an error message and does not retry if clearing fails", async () => {
    const user = userEvent.setup();
    const { clearDemo } = await import("@/api/demo");
    vi.mocked(clearDemo).mockRejectedValue(new Error("network error"));
    const onRetry = vi.fn();

    renderWithProviders(
      <DemoConflictDialog open onOpenChange={vi.fn()} onRetry={onRetry} />,
    );

    await user.click(screen.getByRole("button", { name: /Clear demo data & retry/ }));

    await waitFor(() =>
      expect(screen.getByText("Couldn't clear demo data. Please try again.")).toBeInTheDocument(),
    );
    expect(onRetry).not.toHaveBeenCalled();
  });
});
