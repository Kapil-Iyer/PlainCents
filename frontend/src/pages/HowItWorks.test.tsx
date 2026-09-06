import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeAll, describe, expect, it, vi } from "vitest";

import { HowItWorksPage } from "@/pages/HowItWorks";

beforeAll(() => {
  // jsdom implements neither, and the page uses both: an IntersectionObserver
  // to highlight the current section, and scrollIntoView when a section is
  // selected. Stubbing them keeps the tests about content and semantics
  // rather than about scroll physics jsdom cannot simulate anyway.
  vi.stubGlobal(
    "IntersectionObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  Element.prototype.scrollIntoView = vi.fn();
  // The video section probes for a walkthrough recording that is not in the
  // repository. Default every probe to "not found", which is the real state.
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, headers: new Headers() }));
});

function renderAt(path = "/how-it-works") {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <HowItWorksPage />
    </MemoryRouter>,
  );
}

describe("HowItWorksPage", () => {
  it("leads with the product premise, before any technical detail", () => {
    renderAt();

    expect(screen.getByRole("heading", { name: "How It Works" })).toBeInTheDocument();
    const overview = document.getElementById("overview")!;
    expect(within(overview).getByText("What is PlainCents?")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", {
        name: /reads your bank statements and tells you where the money actually went/i,
      }),
    ).toBeInTheDocument();
  });

  it("says plainly what the product will not do", () => {
    renderAt();

    expect(screen.getByText(/What it deliberately doesn't do/)).toBeInTheDocument();
    expect(screen.getByText(/Connect to your bank/)).toBeInTheDocument();
    expect(screen.getByText(/Track income/)).toBeInTheDocument();
  });

  it("renders every section on one page rather than hiding them behind tabs", () => {
    renderAt();

    for (const id of [
      "overview",
      "walkthrough",
      "video",
      "categorization",
      "memory",
      "forecasting",
      "evaluation",
      "limitations",
    ]) {
      expect(document.getElementById(id)).not.toBeNull();
    }
  });

  it("keeps the #categorization deep link from Transactions working", async () => {
    renderAt("/how-it-works#categorization");

    expect(document.getElementById("categorization")).not.toBeNull();
    // The page defers the scroll to the next frame so the sections exist
    // before it measures them.
    await waitFor(() => expect(Element.prototype.scrollIntoView).toHaveBeenCalled());
  });

  it("keeps the #forecasting deep link from Forecast working", () => {
    renderAt("/how-it-works#forecasting");

    expect(document.getElementById("forecasting")).not.toBeNull();
  });

  describe("app walkthrough", () => {
    it("starts on the first step and advances", async () => {
      const user = userEvent.setup();
      renderAt();

      const walkthrough = within(document.getElementById("walkthrough")!);
      expect(walkthrough.getByText("Step 1 of 10")).toBeInTheDocument();
      expect(walkthrough.getByText("Start empty, or load the demo")).toBeInTheDocument();

      await user.click(walkthrough.getByRole("button", { name: /Next/ }));

      expect(await walkthrough.findByText("Step 2 of 10")).toBeInTheDocument();
      expect(walkthrough.getByText("Upload a bank CSV")).toBeInTheDocument();
    });

    it("wraps around when stepping back from the first step", async () => {
      const user = userEvent.setup();
      renderAt();

      const walkthrough = within(document.getElementById("walkthrough")!);
      await user.click(walkthrough.getByRole("button", { name: /Previous/ }));

      expect(await walkthrough.findByText("Step 10 of 10")).toBeInTheDocument();
    });

    it("explains the dashboard's day-aligned vs. full-month comparison", async () => {
      const user = userEvent.setup();
      renderAt();

      const walkthrough = within(document.getElementById("walkthrough")!);
      // Step 8 is "Read the dashboard" -- click Next 7 times to reach it.
      for (let i = 0; i < 7; i++) {
        await user.click(walkthrough.getByRole("button", { name: /Next/ }));
      }

      expect(await walkthrough.findByText("Read the dashboard")).toBeInTheDocument();
      expect(walkthrough.getByText(/day-aligned/)).toBeInTheDocument();
    });
  });

  describe("video walkthrough", () => {
    it("shows an honest placeholder when no recording is present", async () => {
      renderAt();

      expect(
        await screen.findByText(/The walkthrough hasn't been recorded yet/),
      ).toBeInTheDocument();
      // The expected drop-in path is named, so it is actionable rather than
      // just an apology.
      expect(
        screen.getByText("frontend/public/media/plaincents-walkthrough.mp4"),
      ).toBeInTheDocument();
    });

    it("lists what the recording will cover", () => {
      renderAt();

      expect(screen.getByText("What the recording covers")).toBeInTheDocument();
      expect(screen.getByText(/Importing a real bank CSV/)).toBeInTheDocument();
    });
  });

  describe("categorization journey", () => {
    it("shows the system and human columns as separate stored values", () => {
      renderAt();

      const journey = document.getElementById("categorization")!;
      expect(within(journey).getByText("What the system decided")).toBeInTheDocument();
      expect(within(journey).getByText("What you decided")).toBeInTheDocument();
      expect(within(journey).getAllByText("predicted_category").length).toBeGreaterThan(0);
      expect(within(journey).getAllByText("confirmed_category").length).toBeGreaterThan(0);
    });

    it("explains why a description with no merchant name is not classified", async () => {
      const user = userEvent.setup();
      renderAt();

      await user.click(screen.getByRole("tab", { name: "Nothing to categorize" }));

      expect(await screen.findByText("E-TRANSFER SENT")).toBeInTheDocument();
      expect(screen.getByText(/guessing here isn't classification, it's invention/)).toBeInTheDocument();
    });

    it("explains abstention on a merchant it cannot place", async () => {
      const user = userEvent.setup();
      renderAt();

      await user.click(screen.getByRole("tab", { name: "Not confident enough" }));

      expect(await screen.findByText(/top two categories are nearly tied/)).toBeInTheDocument();
    });

    it("shows the model's advisory model_category suggestion on an abstained row, never as a percentage", async () => {
      const user = userEvent.setup();
      renderAt();

      await user.click(screen.getByRole("tab", { name: "Not confident enough" }));

      const journey = document.getElementById("categorization")!;
      expect(await within(journey).findByText("model_category")).toBeInTheDocument();
      expect(within(journey).getByText("Advisory suggestion")).toBeInTheDocument();
      expect(
        within(journey).getByText(/never shown as a confidence percentage/),
      ).toBeInTheDocument();
    });

    it("does not show the advisory model_category panel on a row that was never abstained", () => {
      renderAt();

      // Default selected tab is "The model decided" (a clean, non-abstained
      // prediction) -- model_category would just repeat predicted_category
      // there, so the panel must not render.
      const journey = document.getElementById("categorization")!;
      expect(within(journey).queryByText("Advisory suggestion")).not.toBeInTheDocument();
    });
  });

  describe("correction memory", () => {
    it("walks through prediction, correction and reuse", async () => {
      const user = userEvent.setup();
      renderAt();

      expect(screen.getByText("PlainCents makes a call")).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: /February/ }));

      expect(await screen.findByText("Your correction is reused")).toBeInTheDocument();
      // The model's own answer survives the correction — that is the point.
      const memory = within(document.getElementById("memory")!);
      // The system's answer and the user's both remain on screen: the point
      // is that a correction does not overwrite the prediction. "Shopping"
      // appears twice by design — once as the stored correction, once as the
      // effective category everything downstream uses.
      expect(memory.getByText("Healthcare")).toBeInTheDocument();
      expect(memory.getAllByText("Shopping")).toHaveLength(2);
    });

    it("states that a system-assigned category never counts as a correction", () => {
      renderAt();

      expect(
        screen.getByText(/never counts as a correction, so it can never/),
      ).toBeInTheDocument();
    });
  });

  describe("forecast explainer", () => {
    it("shows the arithmetic for the selected history", async () => {
      const user = userEvent.setup();
      renderAt();

      expect(screen.getByText(/\$300\.00 \+ \$450\.00 \+ \$600\.00\) ÷ 3 =/)).toBeInTheDocument();

      await user.click(screen.getByRole("radio", { name: "Flat" }));

      expect(await screen.findByText(/\$420\.00 \+ \$410\.00 \+ \$430\.00\) ÷ 3 =/)).toBeInTheDocument();
    });

    it("does not claim three months is as accurate as six or twelve", () => {
      renderAt();

      expect(
        screen.getByText(/not a finding that three months forecasts as well as six or twelve/),
      ).toBeInTheDocument();
    });
  });

  describe("evidence", () => {
    it("leads with the sealed-test score, not the score the model was chosen on", () => {
      renderAt();

      const evaluation = document.getElementById("evaluation")!;
      expect(within(evaluation).getByText("0.58")).toBeInTheDocument();
      expect(
        within(evaluation).getByText("Macro-F1 on held-out merchants"),
      ).toBeInTheDocument();
    });

    it("says on the card, not in a tooltip, that the corpus is fabricated", () => {
      renderAt();

      const evaluation = document.getElementById("evaluation")!;
      expect(within(evaluation).getByText("fabricated")).toBeInTheDocument();
      expect(
        within(evaluation).getByText(/every merchant in it was invented for the benchmark/),
      ).toBeInTheDocument();
    });

    it("names the benchmark as privacy-safe and deployment-oriented, never as real-world accuracy", () => {
      renderAt();

      expect(
        screen.getByText(/privacy-safe, deployment-oriented benchmark/),
      ).toBeInTheDocument();
      expect(screen.getByText(/It is not a real-world accuracy figure/)).toBeInTheDocument();
    });

    it("lists every configuration that was tried, not only the winner", async () => {
      const user = userEvent.setup();
      renderAt();

      await user.click(screen.getByText(/Every configuration tried \(11\)/));

      expect(
        await screen.findByText(/Word \+ character TF-IDF union \(char 2–6\)/),
      ).toBeInTheDocument();
      expect(screen.getByText("Complement Naive Bayes on the union")).toBeInTheDocument();
      expect(screen.getByText("Linear SVM on the union")).toBeInTheDocument();
      expect(
        screen.getByText("The previous production recipe (word TF-IDF, 200 features)"),
      ).toBeInTheDocument();
    });
  });

  describe("limitations", () => {
    it("states the limits plainly rather than hiding them", () => {
      renderAt();

      expect(
        screen.getByText("The evaluation corpus is fabricated, not real"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("The model never learns from your corrections automatically"),
      ).toBeInTheDocument();
      expect(screen.getByText("PlainCents does not track income")).toBeInTheDocument();
    });

    it("lists the claims the product refuses to make", () => {
      renderAt();

      expect(screen.getByText("Things PlainCents does not claim")).toBeInTheDocument();
      expect(
        screen.getByText("PlainCents learns from your corrections and retrains itself."),
      ).toBeInTheDocument();
      expect(
        screen.getByText("Three months of history forecasts as accurately as six or twelve."),
      ).toBeInTheDocument();
    });
  });
});
