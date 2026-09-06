import { describe, expect, it } from "vitest";

import { getCategorySuggestion } from "@/lib/decisionSource";

describe("getCategorySuggestion", () => {
  it("suggests the model's category for a low-confidence abstention with a usable opinion", () => {
    expect(
      getCategorySuggestion({
        decisionSource: "low_confidence_other",
        modelCategory: "Transport",
        predictedCategory: "Other",
      }),
    ).toBe("Transport");
  });

  it("suggests nothing for structural_other, even with a model_category present", () => {
    expect(
      getCategorySuggestion({
        decisionSource: "structural_other",
        modelCategory: "Transport",
        predictedCategory: "Other",
      }),
    ).toBeNull();
  });

  it("suggests nothing for ambiguous_e_transfer", () => {
    expect(
      getCategorySuggestion({
        decisionSource: "ambiguous_e_transfer",
        modelCategory: "Food & Dining",
        predictedCategory: "Other",
      }),
    ).toBeNull();
  });

  it("suggests nothing for a plain model decision", () => {
    expect(
      getCategorySuggestion({
        decisionSource: "model",
        modelCategory: "Transport",
        predictedCategory: "Transport",
      }),
    ).toBeNull();
  });

  it("suggests nothing for a gazetteer decision", () => {
    expect(
      getCategorySuggestion({
        decisionSource: "gazetteer",
        modelCategory: null,
        predictedCategory: "Subscriptions",
      }),
    ).toBeNull();
  });

  it("suggests nothing when model_category is missing", () => {
    expect(
      getCategorySuggestion({ decisionSource: "low_confidence_other", modelCategory: null }),
    ).toBeNull();
    expect(
      getCategorySuggestion({ decisionSource: "low_confidence_other", modelCategory: undefined }),
    ).toBeNull();
  });

  it("suggests nothing when the model's own opinion was also Other", () => {
    expect(
      getCategorySuggestion({
        decisionSource: "low_confidence_other",
        modelCategory: "Other",
        predictedCategory: "Other",
      }),
    ).toBeNull();
  });

  it("suggests nothing when model_category equals the served predicted_category", () => {
    expect(
      getCategorySuggestion({
        decisionSource: "low_confidence_other",
        modelCategory: "Transport",
        predictedCategory: "Transport",
      }),
    ).toBeNull();
  });
});
