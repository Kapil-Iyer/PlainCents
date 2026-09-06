import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { renderWithProviders } from "@/test/utils";
import type { TransactionResponse } from "@/types/transaction";

import { CategoryBadge } from "@/pages/transactions/CategoryBadge";

function txn(overrides: Partial<TransactionResponse>): TransactionResponse {
  return {
    id: 1,
    date: "2026-01-15",
    merchant: "FAKE MERCHANT",
    raw_description: null,
    amount: -10,
    bank_source: "RBC",
    predicted_category: "Other",
    confirmed_category: null,
    effective_category: "Other",
    is_manual_override: false,
    decision_source: null,
    created_at: "2026-01-15T00:00:00Z",
    updated_at: "2026-01-15T00:00:00Z",
    ...overrides,
  };
}

describe("CategoryBadge", () => {
  it("shows a genuine miscellaneous Other with no secondary caption", () => {
    renderWithProviders(
      <CategoryBadge transaction={txn({ decision_source: "structural_other" })} />,
    );
    expect(screen.getByTitle("Predicted by the categorization model")).toHaveTextContent("Other");
    expect(screen.getByText("(no merchant name)")).toBeInTheDocument();
  });

  it("distinguishes a purposeless E-Transfer served as Other", () => {
    renderWithProviders(
      <CategoryBadge transaction={txn({ decision_source: "ambiguous_e_transfer" })} />,
    );
    expect(screen.getByTitle("Predicted by the categorization model")).toHaveTextContent("Other");
    const caption = screen.getByText("(E-Transfer)");
    expect(caption).toBeInTheDocument();
    expect(caption).toHaveAttribute(
      "title",
      "Purpose could not be determined from the bank description",
    );
  });

  it("shows a low-confidence caption for a non-E-Transfer abstention", () => {
    renderWithProviders(
      <CategoryBadge
        transaction={txn({ decision_source: "low_confidence_other", predicted_category: "Other" })}
      />,
    );
    expect(screen.getByText("(low confidence)")).toBeInTheDocument();
  });

  it("shows a recognized-merchant caption for a gazetteer decision", () => {
    renderWithProviders(
      <CategoryBadge
        transaction={txn({
          decision_source: "gazetteer",
          predicted_category: "Subscriptions",
          effective_category: "Subscriptions",
        })}
      />,
    );
    expect(screen.getByText("(recognized merchant)")).toBeInTheDocument();
  });

  it("shows no caption for an ordinary model prediction", () => {
    renderWithProviders(
      <CategoryBadge
        transaction={txn({
          decision_source: "model",
          predicted_category: "Food & Dining",
          effective_category: "Food & Dining",
        })}
      />,
    );
    expect(screen.queryByText(/no merchant name|E-Transfer|low confidence|recognized merchant/)).toBeNull();
  });

  it("never shows a system caption once a human has corrected the category", () => {
    renderWithProviders(
      <CategoryBadge
        transaction={txn({
          decision_source: "ambiguous_e_transfer",
          confirmed_category: "Food & Dining",
          effective_category: "Food & Dining",
          is_manual_override: true,
        })}
      />,
    );
    expect(screen.getByTitle("Confirmed by you")).toHaveTextContent("Food & Dining");
    expect(screen.queryByText("(E-Transfer)")).toBeNull();
  });

  it("handles a null decision_source (pre-migration row) without a caption", () => {
    renderWithProviders(<CategoryBadge transaction={txn({ decision_source: null })} />);
    expect(screen.getByTitle("Predicted by the categorization model")).toHaveTextContent("Other");
    expect(
      screen.queryByText(/no merchant name|E-Transfer|low confidence|recognized merchant/),
    ).toBeNull();
  });
});
