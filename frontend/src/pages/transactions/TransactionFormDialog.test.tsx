import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/utils";

import { TransactionFormDialog } from "@/pages/transactions/TransactionFormDialog";

vi.mock("@/api/transactions", () => ({
  listTransactions: vi.fn(),
  createTransaction: vi.fn(),
  updateTransaction: vi.fn(),
  deleteTransaction: vi.fn(),
}));

/**
 * The manual Add Transaction form already had the exact optional-category
 * UX the North Star describes -- "Category (optional override)" defaulting
 * to "Let the model predict" -- so this patch didn't need to change it.
 * These tests cover that it actually behaves the way the backend fix now
 * assumes: leaving it on the default sends confirmed_category: null (Auto-
 * categorize, runs the shared decision path), and picking one sends the
 * human's explicit choice as authoritative.
 */
describe("TransactionFormDialog -- manual add category UX", () => {
  it("defaults the category selector to Auto-categorize (Let the model predict)", () => {
    renderWithProviders(<TransactionFormDialog open onOpenChange={vi.fn()} />);

    expect(screen.getByRole("combobox")).toHaveTextContent("Let the model predict");
  });

  it("describes that manual entries are categorized automatically", () => {
    renderWithProviders(<TransactionFormDialog open onOpenChange={vi.fn()} />);

    expect(
      screen.getByText("Manually entered transactions are categorized automatically."),
    ).toBeInTheDocument();
  });

  it("lists every real category as selectable, alongside the default", async () => {
    const user = userEvent.setup();
    renderWithProviders(<TransactionFormDialog open onOpenChange={vi.fn()} />);

    await user.click(screen.getByRole("combobox"));

    expect(screen.getByRole("option", { name: "Let the model predict" })).toBeInTheDocument();
    for (const category of ["Food & Dining", "Transport", "Healthcare", "Other"]) {
      expect(screen.getByRole("option", { name: category })).toBeInTheDocument();
    }
  });

  it("submitting with the default sends confirmed_category: null (Auto-categorize)", async () => {
    const user = userEvent.setup();
    const { createTransaction } = await import("@/api/transactions");
    vi.mocked(createTransaction).mockResolvedValue({
      id: 1,
      date: "2026-01-15",
      merchant: "GENERIC RETAILER 4471",
      raw_description: null,
      amount: 42.5,
      bank_source: null,
      predicted_category: "Other",
      confirmed_category: null,
      effective_category: "Other",
      is_manual_override: false,
      decision_source: "low_confidence_other",
      model_category: "Shopping",
      created_at: "2026-01-15T00:00:00Z",
      updated_at: "2026-01-15T00:00:00Z",
    });

    renderWithProviders(<TransactionFormDialog open onOpenChange={vi.fn()} />);
    await user.type(screen.getByLabelText("Merchant"), "GENERIC RETAILER 4471");
    await user.type(screen.getByLabelText("Amount"), "42.50");
    await user.click(screen.getByRole("button", { name: "Add transaction" }));

    await waitFor(() =>
      expect(createTransaction).toHaveBeenCalledWith(
        expect.objectContaining({ merchant: "GENERIC RETAILER 4471", confirmed_category: null }),
      ),
    );
  });

  it("choosing an explicit category sends it as confirmed_category, authoritative over auto-categorization", async () => {
    const user = userEvent.setup();
    const { createTransaction } = await import("@/api/transactions");
    vi.mocked(createTransaction).mockResolvedValue({
      id: 2,
      date: "2026-01-15",
      merchant: "GENERIC RETAILER 4471",
      raw_description: null,
      amount: 42.5,
      bank_source: null,
      predicted_category: "Shopping",
      confirmed_category: "Healthcare",
      effective_category: "Healthcare",
      is_manual_override: true,
      decision_source: "model",
      model_category: "Shopping",
      created_at: "2026-01-15T00:00:00Z",
      updated_at: "2026-01-15T00:00:00Z",
    });

    renderWithProviders(<TransactionFormDialog open onOpenChange={vi.fn()} />);
    await user.type(screen.getByLabelText("Merchant"), "GENERIC RETAILER 4471");
    await user.type(screen.getByLabelText("Amount"), "42.50");
    await user.click(screen.getByRole("combobox"));
    await user.click(screen.getByRole("option", { name: "Healthcare" }));
    await user.click(screen.getByRole("button", { name: "Add transaction" }));

    await waitFor(() =>
      expect(createTransaction).toHaveBeenCalledWith(
        expect.objectContaining({ confirmed_category: "Healthcare" }),
      ),
    );
  });
});
