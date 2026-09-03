import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";

import { HowItWorksPage } from "@/pages/HowItWorks";

function renderAt(path = "/how-it-works") {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <HowItWorksPage />
    </MemoryRouter>,
  );
}

describe("HowItWorksPage", () => {
  it("renders the Overview pipeline by default", () => {
    renderAt();

    expect(screen.getByRole("heading", { name: "How It Works" })).toBeInTheDocument();
    expect(screen.getByText("PlainCents in one pipeline")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
  });

  it("switches sections on tab click", async () => {
    const user = userEvent.setup();
    renderAt();

    await user.click(screen.getByRole("tab", { name: "Categorization" }));

    expect(screen.getByRole("tab", { name: "Categorization" })).toHaveAttribute("aria-selected", "true");
    // AnimatePresence exit/enter is real-timer-driven — wait for the new
    // panel's content rather than asserting synchronously post-click.
    expect(await screen.findByText("Candidates benchmarked on VALIDATION")).toBeInTheDocument();
  });

  it("opens the categorization tab directly from a #categorization hash", () => {
    renderAt("/how-it-works#categorization");

    expect(screen.getByRole("tab", { name: "Categorization" })).toHaveAttribute("aria-selected", "true");
    // All three benchmarked candidates must appear, not only the winner.
    expect(screen.getByText("K-Means")).toBeInTheDocument();
    expect(screen.getByText("TF-IDF + Linear SVM")).toBeInTheDocument();
    expect(screen.getByText("TF-IDF + Logistic Regression")).toBeInTheDocument();
    // Evidence tier is visible on the card, not tooltip-only.
    expect(screen.getByText("Tier B evidence")).toBeInTheDocument();
  });

  it("opens the forecasting tab directly from a #forecasting hash and preserves strategy variants", () => {
    renderAt("/how-it-works#forecasting");

    expect(screen.getByRole("tab", { name: "Forecasting" })).toHaveAttribute("aria-selected", "true");
    // Ridge/RF each appear as two separate rows (last-known vs recursive),
    // never collapsed into one score.
    expect(screen.getAllByText("Ridge")).toHaveLength(2);
    expect(screen.getAllByText("Random Forest")).toHaveLength(2);
    expect(screen.getAllByText("Last-known")).toHaveLength(2);
    expect(screen.getAllByText("Recursive")).toHaveLength(2);
    // Naive's strategy is displayed as N/A, not blank or invented.
    expect(screen.getAllByText("N/A")).toHaveLength(2); // Naive + Seasonal Naive
  });

  it("lists claims the product does not make on the Limitations tab", async () => {
    const user = userEvent.setup();
    renderAt();

    await user.click(screen.getByRole("tab", { name: "Limitations & Evidence" }));

    expect(
      await screen.findByText("PlainCents categorizes real-world bank transactions at 42.2% accuracy."),
    ).toBeInTheDocument();
    expect(screen.getByText("PlainCents automatically retrains from user corrections.")).toBeInTheDocument();
  });

  it("shows the human-in-the-loop predicted/confirmed/effective chain", async () => {
    const user = userEvent.setup();
    renderAt();

    await user.click(screen.getByRole("tab", { name: "Human-in-the-Loop" }));

    expect(await screen.findByText("predicted_category")).toBeInTheDocument();
    expect(screen.getByText("confirmed_category")).toBeInTheDocument();
    expect(
      screen.getByText("effective_category = COALESCE(confirmed_category, predicted_category)"),
    ).toBeInTheDocument();
    expect(screen.getByText(/do NOT trigger automatic retraining/)).toBeInTheDocument();
  });
});
