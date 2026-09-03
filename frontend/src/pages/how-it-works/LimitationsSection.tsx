import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  CATEGORIZATION_EVIDENCE_QUALIFIER,
  CATEGORIZATION_FINAL_RESULT,
  FORECASTING_EVIDENCE_QUALIFIER,
  FORECASTING_FINAL_RESULT,
  NOT_SUPPORTED_CLAIMS,
  RETRAINING_QUALIFIER,
  TD_IMPORT_QUALIFIER,
} from "@/data/methodology";

import { EvidenceBadge } from "@/pages/how-it-works/EvidenceBadge";

/** This section is meant to read as trust-building transparency, not legal
 * fine print — plain statements of what the evidence is and isn't, up
 * front, not buried in a footnote. */
export function LimitationsSection() {
  return (
    <div className="flex flex-col gap-5">
      <Card>
        <CardHeader>
          <CardTitle>Limitations & Evidence</CardTitle>
          <CardDescription>
            What the numbers on this page actually prove — and what they deliberately don't.
          </CardDescription>
        </CardHeader>
      </Card>

      <Card>
        <CardHeader className="flex-row items-center justify-between gap-3">
          <CardTitle>Categorization evidence</CardTitle>
          <EvidenceBadge tier="Tier B" />
        </CardHeader>
        <CardContent className="flex flex-col gap-2 text-sm text-muted-foreground">
          <p>{CATEGORIZATION_EVIDENCE_QUALIFIER}</p>
          <p>
            Independently curated benchmark — {CATEGORIZATION_FINAL_RESULT.nRows} rows,{" "}
            {CATEGORIZATION_FINAL_RESULT.nMerchantGroups} merchant groups in the held-out slice. Not real
            bank export data. A single held-out measurement, not a distribution.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex-row items-center justify-between gap-3">
          <CardTitle>Forecasting evidence</CardTitle>
          <EvidenceBadge tier="Synthetic" />
        </CardHeader>
        <CardContent className="flex flex-col gap-2 text-sm text-muted-foreground">
          <p>{FORECASTING_EVIDENCE_QUALIFIER}</p>
          <p>
            {FORECASTING_FINAL_RESULT.nPredictions} predictions across a fully synthetic 24-month dataset,
            categorized read-only by the production categorization model. Never real spending data.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Other qualifiers worth knowing</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3 text-sm text-muted-foreground">
          <p>{TD_IMPORT_QUALIFIER}</p>
          <p>{RETRAINING_QUALIFIER}</p>
        </CardContent>
      </Card>

      <Card variant="elevated">
        <CardHeader>
          <CardTitle>Claims we deliberately don't make</CardTitle>
          <CardDescription>
            Anything on this list is unsupported by current evidence and is never asserted elsewhere in the
            product.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ul className="flex flex-col gap-2 text-sm text-muted-foreground">
            {NOT_SUPPORTED_CLAIMS.map((claim, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-destructive" aria-hidden>✕</span>
                <span className="line-through decoration-destructive/50">{claim}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
