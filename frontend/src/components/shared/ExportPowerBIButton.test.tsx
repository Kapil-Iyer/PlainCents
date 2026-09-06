import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/utils";

import { ExportPowerBIButton } from "@/components/shared/ExportPowerBIButton";

vi.mock("@/api/export", () => ({
  downloadPowerBIExport: vi.fn(),
}));

describe("ExportPowerBIButton", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("triggers the download on click", async () => {
    const user = userEvent.setup();
    const { downloadPowerBIExport } = await import("@/api/export");
    vi.mocked(downloadPowerBIExport).mockResolvedValue(undefined);

    renderWithProviders(<ExportPowerBIButton />);
    await user.click(screen.getByRole("button", { name: /Export for Power BI/ }));

    expect(downloadPowerBIExport).toHaveBeenCalled();
  });

  it("shows an error toast if the export fails, without crashing", async () => {
    const user = userEvent.setup();
    const { downloadPowerBIExport } = await import("@/api/export");
    vi.mocked(downloadPowerBIExport).mockRejectedValue(new Error("network error"));

    renderWithProviders(<ExportPowerBIButton />);
    await user.click(screen.getByRole("button", { name: /Export for Power BI/ }));

    await waitFor(() => expect(screen.getByText("Couldn't generate the export")).toBeInTheDocument());
  });
});
