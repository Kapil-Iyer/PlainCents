/**
 * Static step definitions for the How It Works "Overview" pipeline diagram.
 * Purely presentational — mirrors the product's actual route/flow order
 * (router.tsx), not a claim requiring ML evidence citation.
 */
export interface PipelineStep {
  id: string;
  label: string;
  description: string;
}

export const PIPELINE_STEPS: PipelineStep[] = [
  { id: "csv", label: "Bank CSV", description: "You upload a TD export." },
  { id: "normalize", label: "Normalize", description: "Rows are cleaned & standardized." },
  { id: "categorize", label: "Categorize", description: "TF-IDF + Logistic Regression predicts a category." },
  { id: "confirm", label: "Confirm", description: "You can correct any prediction." },
  { id: "forecast", label: "Forecast", description: "Naive model projects +1/+2/+3 months." },
  { id: "insights", label: "Insights", description: "Dashboards, Transactions, Portfolio." },
];
