import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/utils";
import type { ImportPreview, ImportResult } from "@/types/import";

import { ImportPage } from "@/pages/Import";

vi.mock("@/api/imports", () => ({
  createImport: vi.fn(),
  confirmImport: vi.fn(),
}));
vi.mock("@/api/demo", () => ({
  loadDemo: vi.fn(),
  clearDemo: vi.fn(),
}));

const preview: ImportPreview = {
  batch_id: 42,
  detected_bank: "TD",
  rows_valid: 10,
  rows_unparseable: 1,
  rows_duplicate: 2,
  rows_skipped_credit: 0,
  rows_skipped_currency: 0,
  date_range: { from: "2026-01-01", to: "2026-01-31" },
  sample_rows: [
    { date: "2026-01-05", merchant: "Loblaws", amount: -50, predicted_category: "Food & Dining", is_duplicate: false },
  ],
  status: "previewing",
  categorization_available: true,
};

const result: ImportResult = {
  batch_id: 42,
  rows_imported: 8,
  rows_skipped_unparseable: 1,
  rows_skipped_duplicate: 2,
  rows_skipped_credit: 0,
  rows_skipped_currency: 0,
  status: "confirmed",
};

describe("ImportPage", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("shows the preview after upload, then the result after confirm", async () => {
    const user = userEvent.setup();
    const { createImport, confirmImport } = await import("@/api/imports");
    vi.mocked(createImport).mockResolvedValue(preview);
    vi.mocked(confirmImport).mockResolvedValue(result);

    renderWithProviders(<ImportPage />);

    const file = new File(["date,merchant,amount\n"], "td_export.csv", { type: "text/csv" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);

    await user.click(screen.getByRole("button", { name: /Upload & preview/i }));

    await screen.findByText("Preview");
    expect(createImport).toHaveBeenCalledWith(file, "Auto");
    expect(screen.getByText("Loblaws")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Confirm import" }));

    await screen.findByText("Import complete");
    expect(confirmImport).toHaveBeenCalledWith(42);
    expect(screen.getByText("8")).toBeInTheDocument();
  });

  it("blocks confirmation with a clear message on a 503 from the model", async () => {
    const user = userEvent.setup();
    const { createImport, confirmImport } = await import("@/api/imports");
    const { ApiError } = await import("@/types/common");
    vi.mocked(createImport).mockResolvedValue({ ...preview, categorization_available: false });
    vi.mocked(confirmImport).mockRejectedValue(
      new ApiError(503, { error: "categorization_unavailable", message: "unavailable", details: {} }),
    );

    renderWithProviders(<ImportPage />);
    const file = new File(["x"], "td_export.csv", { type: "text/csv" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);
    await user.click(screen.getByRole("button", { name: /Upload & preview/i }));

    await screen.findByText("Preview");
    await user.click(screen.getByRole("button", { name: "Confirm import" }));

    await waitFor(() =>
      expect(screen.getByText(/can't be confirmed yet/)).toBeInTheDocument(),
    );
  });
});
