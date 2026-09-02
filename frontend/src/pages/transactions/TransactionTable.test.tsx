import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/utils";
import type { TransactionResponse } from "@/types/transaction";

import { TransactionTable } from "@/pages/transactions/TransactionTable";

vi.mock("@/api/transactions", () => ({
  deleteTransaction: vi.fn().mockResolvedValue({ id: 1, deleted: true }),
}));

const mockTransactions: TransactionResponse[] = [
  {
    id: 1,
    date: "2026-01-15",
    merchant: "Loblaws",
    raw_description: null,
    amount: -84.32,
    bank_source: "TD",
    predicted_category: "Food & Dining",
    confirmed_category: null,
    effective_category: "Food & Dining",
    is_manual_override: false,
    created_at: "2026-01-15T00:00:00Z",
    updated_at: "2026-01-15T00:00:00Z",
  },
  {
    id: 2,
    date: "2026-01-16",
    merchant: "Netflix",
    raw_description: null,
    amount: -16.99,
    bank_source: "TD",
    predicted_category: "Entertainment",
    confirmed_category: "Subscriptions",
    effective_category: "Subscriptions",
    is_manual_override: true,
    created_at: "2026-01-16T00:00:00Z",
    updated_at: "2026-01-16T00:00:00Z",
  },
];

describe("TransactionTable", () => {
  it("renders rows with predicted vs confirmed category badges", () => {
    renderWithProviders(<TransactionTable transactions={mockTransactions} />);

    expect(screen.getByText("Loblaws")).toBeInTheDocument();
    expect(screen.getByText("Netflix")).toBeInTheDocument();

    // Predicted category shows a "Predicted by the categorization model" badge.
    expect(screen.getByTitle("Predicted by the categorization model")).toHaveTextContent(
      "Food & Dining",
    );
    // Confirmed/corrected category shows a distinct "Confirmed by you" badge.
    expect(screen.getByTitle("Confirmed by you")).toHaveTextContent("Subscriptions");
  });

  it("requires confirmation before deleting a transaction", async () => {
    const user = userEvent.setup();
    const { deleteTransaction } = await import("@/api/transactions");
    renderWithProviders(<TransactionTable transactions={mockTransactions} />);

    await user.click(screen.getByRole("button", { name: "Delete Loblaws" }));

    // The confirm dialog appears; the delete API must not fire yet.
    expect(await screen.findByText("Delete this transaction?")).toBeInTheDocument();
    expect(deleteTransaction).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Delete" }));

    expect(deleteTransaction).toHaveBeenCalledWith(1);
  });
});
