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

function QualifiedRow({
  tier,
  headline,
  note,
}: {
  tier: "Tier B" | "Synthetic" | null;
  headline: string;
  note: string;
}) {
  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-warning/30 bg-warning/5 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-semibold">{headline}</span>
        {tier && <EvidenceBadge tier={tier} />}
      </div>
      <p className="text-xs text-muted-foreground">{note}</p>
    </div>
  );
}

/** This section is meant to read as trust-building transparency, not legal
 * fine print — an evidence ladder using the claim matrix's own verdict
 * terms (SUPPORTED_WITH_QUALIFICATION / NOT_SUPPORTED), all rendered
 * unconditionally: no hover, tooltip, animation, or disclosure gates any
 * qualification here. Deliberately restrained — no spectacle needed. */
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
        <CardHeader>
          <CardTitle>SUPPORTED_WITH_QUALIFICATION</CardTitle>
          <CardDescription>
            True numbers, each tied to a named evidence tier — never described beyond it.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <QualifiedRow
            tier="Tier B"
            headline={`${CATEGORIZATION_FINAL_RESULT.accuracyPct.toFixed(1)}% accuracy / ${CATEGORIZATION_FINAL_RESULT.macroF1.toFixed(4)} macro F1`}
            note={CATEGORIZATION_EVIDENCE_QUALIFIER}
          />
          <QualifiedRow
            tier="Synthetic"
            headline={`${FORECASTING_FINAL_RESULT.combinedWapePct.toFixed(2)}% WAPE`}
            note={FORECASTING_EVIDENCE_QUALIFIER}
          />
          <QualifiedRow tier={null} headline="TD CSV import" note={TD_IMPORT_QUALIFIER} />
          <QualifiedRow tier={null} headline="No automatic retraining" note={RETRAINING_QUALIFIER} />
        </CardContent>
      </Card>

      <Card variant="elevated">
        <CardHeader>
          <CardTitle>NOT_SUPPORTED</CardTitle>
          <CardDescription>Never asserted anywhere else in the product.</CardDescription>
        </CardHeader>
        <CardContent>
          <ul className="flex flex-col gap-2 text-sm text-muted-foreground">
            {NOT_SUPPORTED_CLAIMS.map((claim, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-destructive" aria-hidden>
                  ✕
                </span>
                <span className="line-through decoration-destructive/50">{claim}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
