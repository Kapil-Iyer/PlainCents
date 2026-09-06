import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { renderWithProviders } from "@/test/utils";
import type { ImportPreview } from "@/types/import";

import { ImportPreviewCard } from "@/pages/import/ImportPreviewCard";

function preview(overrides: Partial<ImportPreview> = {}): ImportPreview {
  return {
    batch_id: 1,
    detected_bank: "RBC",
    rows_valid: 1,
    rows_unparseable: 0,
    rows_duplicate: 0,
    rows_skipped_credit: 0,
    rows_skipped_currency: 0,
    date_range: { from: "2026-01-01", to: "2026-01-31" },
    sample_rows: [
      {
        date: "2026-01-05",
        merchant: "FAKE MERCHANT",
        amount: -10,
        predicted_category: "Other",
        remembered_category: null,
        effective_category: "Other",
        decision_source: "ambiguous_e_transfer",
        is_duplicate: false,
      },
    ],
    status: "previewing",
    categorization_available: true,
    ...overrides,
  };
}

const noop = () => {};

describe("ImportPreviewCard", () => {
  it("distinguishes a purposeless E-Transfer served as Other, same as the confirmed view", () => {
    renderWithProviders(
      <ImportPreviewCard preview={preview()} onConfirm={noop} onCancel={noop} pending={false} />,
    );

    const caption = screen.getByText("(E-Transfer)");
    expect(caption).toBeInTheDocument();
    expect(caption).toHaveAttribute(
      "title",
      "Purpose could not be determined from the bank description",
    );
  });

  it("shows a distinct caption for genuine structural Other (no merchant name)", () => {
    renderWithProviders(
      <ImportPreviewCard
        preview={preview({
          sample_rows: [
            {
              date: "2026-01-05",
              merchant: "ABM WITHDRAWAL",
              amount: -40,
              predicted_category: "Other",
              remembered_category: null,
              effective_category: "Other",
              decision_source: "structural_other",
              is_duplicate: false,
            },
          ],
        })}
        onConfirm={noop}
        onCancel={noop}
        pending={false}
      />,
    );

    expect(screen.getByText("(no merchant name)")).toBeInTheDocument();
    expect(screen.queryByText("(E-Transfer)")).toBeNull();
  });
});
