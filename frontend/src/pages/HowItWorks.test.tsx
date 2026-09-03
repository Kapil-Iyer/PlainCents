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

  describe("Human-in-the-loop interactive demo (Phase 11C-B)", () => {
    async function openHitl() {
      const user = userEvent.setup();
      renderAt();
      await user.click(screen.getByRole("tab", { name: "Human-in-the-Loop" }));
      await screen.findByText("predicted_category");
      return user;
    }

    it("starts with a predicted category and no confirmation", async () => {
      await openHitl();

      // Predicted badge shows the initial prediction.
      expect(screen.getAllByText("Transport").length).toBeGreaterThan(0);
      // confirmed_category is not yet set.
      expect(screen.getByText("not corrected")).toBeInTheDocument();
      // No API/network layer is involved — this is local state only, so the
      // picker is a plain <select>, not something wired to a query/mutation.
      expect(screen.getByLabelText("Correct the predicted category")).toBeInTheDocument();
    });

    it("updates confirmed_category on correction while preserving predicted_category", async () => {
      const user = await openHitl();

      await user.selectOptions(screen.getByLabelText("Correct the predicted category"), "Entertainment");

      // A confirmed badge for the new category appears...
      expect(screen.getAllByText("Entertainment").length).toBeGreaterThan(0);
      // ...while the original prediction is still shown, never overwritten.
      expect(screen.getAllByText("Transport").length).toBeGreaterThan(0);
      expect(screen.queryByText("not corrected")).not.toBeInTheDocument();
    });

    it("moves the downstream total from the predicted bucket to the confirmed bucket", async () => {
      const user = await openHitl();

      expect(screen.getByTestId("downstream-count-Transport")).toHaveTextContent("3");
      expect(screen.getByTestId("downstream-count-Entertainment")).toHaveTextContent("1");

      await user.selectOptions(screen.getByLabelText("Correct the predicted category"), "Entertainment");

      expect(screen.getByTestId("downstream-count-Transport")).toHaveTextContent("2");
      expect(screen.getByTestId("downstream-count-Entertainment")).toHaveTextContent("2");
    });
  });

  describe("Evaluation Methodology (Phase 11C-B)", () => {
    it("shows merchant-group partition sizes for TRAIN/VALIDATION/FINAL_TEST", async () => {
      const user = userEvent.setup();
      renderAt();

      await user.click(screen.getByRole("tab", { name: "Evaluation Methodology" }));

      expect(await screen.findByText("TRAIN")).toBeInTheDocument();
      expect(screen.getByText("VALIDATION")).toBeInTheDocument();
      expect(screen.getByText("FINAL_TEST")).toBeInTheDocument();
      expect(screen.getByText("133 rows · 47 merchant groups")).toBeInTheDocument();
      expect(screen.getByText("50 rows · 17 merchant groups")).toBeInTheDocument();
      expect(screen.getByText("45 rows · 17 merchant groups")).toBeInTheDocument();
    });

    it("switches to the forecasting temporal view and back", async () => {
      const user = userEvent.setup();
      renderAt();

      await user.click(screen.getByRole("tab", { name: "Evaluation Methodology" }));
      await screen.findByText("TRAIN");

      await user.click(screen.getByRole("button", { name: "Forecasting timeline" }));
      expect(await screen.findByText(/Reserved \(FINAL\)/)).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Categorization split" }));
      expect(await screen.findByText("TRAIN")).toBeInTheDocument();
    });
  });
});
