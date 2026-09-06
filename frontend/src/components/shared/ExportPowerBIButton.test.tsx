import { screen, waitFor, within } from "@testing-library/react";
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

  it("opens a workflow dialog rather than downloading immediately", async () => {
    const user = userEvent.setup();
    const { downloadPowerBIExport } = await import("@/api/export");

    renderWithProviders(<ExportPowerBIButton />);
    await user.click(screen.getByRole("button", { name: /Export for Power BI/ }));

    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(downloadPowerBIExport).not.toHaveBeenCalled();
  });

  it("never claims a live Power BI connection or an automatic template", async () => {
    const user = userEvent.setup();

    renderWithProviders(<ExportPowerBIButton />);
    await user.click(screen.getByRole("button", { name: /Export for Power BI/ }));
    const dialog = await screen.findByRole("dialog");

    expect(within(dialog).getByText(/never live-connected/)).toBeInTheDocument();
    expect(within(dialog).queryByText(/download.*template/i)).not.toBeInTheDocument();
  });

  it("Download data pack triggers the export", async () => {
    const user = userEvent.setup();
    const { downloadPowerBIExport } = await import("@/api/export");
    vi.mocked(downloadPowerBIExport).mockResolvedValue(undefined);

    renderWithProviders(<ExportPowerBIButton />);
    await user.click(screen.getByRole("button", { name: /Export for Power BI/ }));
    const dialog = await screen.findByRole("dialog");
    await user.click(within(dialog).getByRole("button", { name: "Download data pack" }));

    expect(downloadPowerBIExport).toHaveBeenCalled();
  });

  it("shows an error toast if the export fails, without crashing", async () => {
    const user = userEvent.setup();
    const { downloadPowerBIExport } = await import("@/api/export");
    vi.mocked(downloadPowerBIExport).mockRejectedValue(new Error("network error"));

    renderWithProviders(<ExportPowerBIButton />);
    await user.click(screen.getByRole("button", { name: /Export for Power BI/ }));
    const dialog = await screen.findByRole("dialog");
    await user.click(within(dialog).getByRole("button", { name: "Download data pack" }));

    await waitFor(() => expect(screen.getByText("Couldn't generate the export")).toBeInTheDocument());
  });

  it("offers the real, safe Power BI theme file, explicitly labeled as a theme, not a template", async () => {
    const user = userEvent.setup();

    renderWithProviders(<ExportPowerBIButton />);
    await user.click(screen.getByRole("button", { name: /Export for Power BI/ }));
    const dialog = await screen.findByRole("dialog");

    const themeLink = within(dialog).getByRole("link", { name: /Power BI theme/ });
    expect(themeLink).toHaveAttribute("href", "/powerbi_theme.json");
  });

  it("the setup guide is available and never claims automation Power BI export doesn't do", async () => {
    const user = userEvent.setup();

    renderWithProviders(<ExportPowerBIButton />);
    await user.click(screen.getByRole("button", { name: /Export for Power BI/ }));
    const dialog = await screen.findByRole("dialog");
    await user.click(within(dialog).getByText("View setup guide"));

    expect(within(dialog).getByText(/No PlainCents Power BI template exists yet/)).toBeInTheDocument();
    expect(within(dialog).getAllByText(/Get Data/).length).toBeGreaterThan(0);
  });
});
