import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/utils";
import type { TransactionListResponse, TransactionResponse } from "@/types/transaction";

import { TransactionsPage } from "@/pages/Transactions";

vi.mock("@/api/transactions", () => ({
  listTransactions: vi.fn(),
  createTransaction: vi.fn(),
  updateTransaction: vi.fn(),
  deleteTransaction: vi.fn(),
}));

const baseTransaction: TransactionResponse = {
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
};

function listResponse(items: TransactionResponse[]): TransactionListResponse {
  return { items, total: items.length, page: 1, page_size: 25 };
}

describe("TransactionsPage", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("renders an empty state when the API returns no transactions", async () => {
    const { listTransactions } = await import("@/api/transactions");
    vi.mocked(listTransactions).mockResolvedValue(listResponse([]));

    renderWithProviders(<TransactionsPage />);

    expect(await screen.findByText("No transactions yet")).toBeInTheDocument();
  });

  it("updates the badge after a category correction is confirmed", async () => {
    const user = userEvent.setup();
    const { listTransactions, updateTransaction } = await import("@/api/transactions");

    vi.mocked(listTransactions).mockResolvedValueOnce(listResponse([baseTransaction]));
    vi.mocked(updateTransaction).mockResolvedValue({
      ...baseTransaction,
      confirmed_category: "Shopping",
      effective_category: "Shopping",
      is_manual_override: true,
    });

    renderWithProviders(<TransactionsPage />);

    await screen.findByText("Loblaws");
    expect(screen.getByTitle("Predicted by the categorization model")).toBeInTheDocument();

    // After the correction is saved, the next list refetch reflects it.
    vi.mocked(listTransactions).mockResolvedValueOnce(
      listResponse([
        {
          ...baseTransaction,
          confirmed_category: "Shopping",
          effective_category: "Shopping",
          is_manual_override: true,
        },
      ]),
    );

    await user.click(screen.getByRole("button", { name: "Edit Loblaws" }));
    const dialog = await screen.findByRole("dialog");
    await user.click(within(dialog).getByLabelText(/Category/));
    await user.click(await screen.findByRole("option", { name: "Shopping" }));
    await user.click(within(dialog).getByRole("button", { name: "Save changes" }));

    await waitFor(() => expect(updateTransaction).toHaveBeenCalledWith(1, expect.objectContaining({
      confirmed_category: "Shopping",
    })));

    await waitFor(() =>
      expect(screen.getByTitle("Confirmed by you")).toHaveTextContent("Shopping"),
    );
  });
});
