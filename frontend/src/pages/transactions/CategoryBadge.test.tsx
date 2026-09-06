import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/utils";
import type { TransactionResponse } from "@/types/transaction";

import { CategoryBadge } from "@/pages/transactions/CategoryBadge";

vi.mock("@/api/transactions", () => ({
  updateTransaction: vi.fn(),
}));

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

  // -- advisory suggestion + one-click "Use" -------------------------------

  it("shows a Suggested chip with a Use button for a low-confidence abstention", () => {
    renderWithProviders(
      <CategoryBadge
        transaction={txn({
          decision_source: "low_confidence_other",
          predicted_category: "Other",
          model_category: "Transport",
        })}
      />,
    );
    expect(screen.getByText("Suggested: Transport")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Use Transport" })).toBeInTheDocument();
    // Never a confidence percentage.
    expect(screen.queryByText(/%/)).toBeNull();
  });

  it("clicking Use sets confirmed_category via the normal update path", async () => {
    const user = userEvent.setup();
    const { updateTransaction } = await import("@/api/transactions");
    vi.mocked(updateTransaction).mockResolvedValue(
      txn({ confirmed_category: "Transport", effective_category: "Transport", is_manual_override: true }),
    );

    renderWithProviders(
      <CategoryBadge
        transaction={txn({
          id: 42,
          decision_source: "low_confidence_other",
          predicted_category: "Other",
          model_category: "Transport",
        })}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Use Transport" }));

    await waitFor(() =>
      expect(updateTransaction).toHaveBeenCalledWith(42, { confirmed_category: "Transport" }),
    );
  });

  it("does not suggest for structural_other even if model_category were present", () => {
    renderWithProviders(
      <CategoryBadge
        transaction={txn({ decision_source: "structural_other", model_category: "Transport" })}
      />,
    );
    expect(screen.queryByText(/Suggested:/)).toBeNull();
  });

  it("does not suggest for ambiguous_e_transfer even if model_category were present", () => {
    renderWithProviders(
      <CategoryBadge
        transaction={txn({ decision_source: "ambiguous_e_transfer", model_category: "Food & Dining" })}
      />,
    );
    expect(screen.queryByText(/Suggested:/)).toBeNull();
  });

  it("does not suggest for an ordinary model/gazetteer decision", () => {
    renderWithProviders(
      <CategoryBadge
        transaction={txn({
          decision_source: "model",
          predicted_category: "Transport",
          effective_category: "Transport",
          model_category: "Transport",
        })}
      />,
    );
    expect(screen.queryByText(/Suggested:/)).toBeNull();
  });

  it("does not suggest when model_category is missing or matches predicted_category", () => {
    renderWithProviders(
      <CategoryBadge transaction={txn({ decision_source: "low_confidence_other", model_category: null })} />,
    );
    expect(screen.queryByText(/Suggested:/)).toBeNull();
  });

  it("never shows a suggestion once a human has corrected the category", () => {
    renderWithProviders(
      <CategoryBadge
        transaction={txn({
          decision_source: "low_confidence_other",
          model_category: "Transport",
          confirmed_category: "Food & Dining",
          effective_category: "Food & Dining",
          is_manual_override: true,
        })}
      />,
    );
    expect(screen.queryByText(/Suggested:/)).toBeNull();
  });
});
