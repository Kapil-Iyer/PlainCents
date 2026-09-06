import { AlertTriangle, ArrowRight, Loader2, UserCheck } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { describeDecisionSource } from "@/lib/decisionSource";
import { cn, formatCurrency, formatDate } from "@/lib/utils";
import type { ImportSampleRow, ImportPreview } from "@/types/import";

interface ImportPreviewCardProps {
  preview: ImportPreview;
  onConfirm: () => void;
  onCancel: () => void;
  pending: boolean;
  error?: string | null;
}

export function ImportPreviewCard({
  preview,
  onConfirm,
  onCancel,
  pending,
  error,
}: ImportPreviewCardProps) {
  const reduceMotion = useReducedMotion();
  const hasExclusions = preview.rows_skipped_credit > 0 || preview.rows_skipped_currency > 0;
  const toImport = preview.rows_valid - preview.rows_duplicate;
  const remembered = preview.sample_rows.filter((r) => r.remembered_category).length;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Preview</CardTitle>
        <p className="text-sm text-muted-foreground">
          Detected format: <span className="font-medium text-foreground">{preview.detected_bank}</span>
          {" · "}Nothing has been imported yet — this is what will happen if you confirm.
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        {/* Import UX copy: "Valid rows" told the user a parser fact, not a
         * product one. What they need to know is how many purchases will
         * land in their account, and what is being left out and why. */}
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Spend rows to import" value={toImport} emphasis />
          <Stat label="Already imported" value={preview.rows_duplicate} />
          <Stat label="Unparseable" value={preview.rows_unparseable} />
          <Stat
            label="Date range"
            value={
              preview.date_range.from && preview.date_range.to
                ? `${formatDate(preview.date_range.from)} – ${formatDate(preview.date_range.to)}`
                : "—"
            }
          />
        </div>

        {hasExclusions && (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            {preview.rows_skipped_credit > 0 && (
              <Stat label="Credits / inflows skipped" value={preview.rows_skipped_credit} />
            )}
            {preview.rows_skipped_currency > 0 && (
              <Stat label="Unsupported currency" value={preview.rows_skipped_currency} />
            )}
          </div>
        )}

        {!preview.categorization_available && (
          <Notice tone="warning">
            The categorization model is unavailable right now. Rows can still be previewed, but
            confirming this import will be blocked until the model is back — nothing will be
            imported without a category.
          </Notice>
        )}

        {preview.rows_unparseable > 0 && (
          <Notice>
            {preview.rows_unparseable} row{preview.rows_unparseable === 1 ? "" : "s"} couldn&apos;t
            be read and will be skipped. Some export formats do this inconsistently — it
            doesn&apos;t necessarily mean the file itself is broken.
          </Notice>
        )}

        {preview.rows_duplicate > 0 && (
          <Notice>
            {preview.rows_duplicate} row{preview.rows_duplicate === 1 ? " is" : "s are"} already in
            your account and will be skipped, so re-importing an overlapping statement is safe.
          </Notice>
        )}

        {hasExclusions && (
          <Notice>
            {preview.rows_skipped_credit > 0 &&
              `${preview.rows_skipped_credit} credit${preview.rows_skipped_credit === 1 ? "" : "s"} / inflow${preview.rows_skipped_credit === 1 ? "" : "s"} ${preview.rows_skipped_credit === 1 ? "was" : "were"} recognized and skipped — PlainCents tracks spending, not income. `}
            {preview.rows_skipped_currency > 0 &&
              `${preview.rows_skipped_currency} row${preview.rows_skipped_currency === 1 ? "" : "s"} in an unsupported currency ${preview.rows_skipped_currency === 1 ? "was" : "were"} skipped rather than converted at a made-up rate.`}
          </Notice>
        )}

        {remembered > 0 && (
          <Notice tone="success" icon={UserCheck}>
            {remembered} row{remembered === 1 ? "" : "s"} below already {remembered === 1 ? "uses" : "use"} a
            category you set yourself for that merchant.
          </Notice>
        )}

        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full min-w-[680px] text-sm">
            <caption className="sr-only">
              First {preview.sample_rows.length} rows of this import, with the category each will
              be filed under.
            </caption>
            <thead>
              <tr className="border-b border-border bg-muted/50 text-left text-xs font-medium uppercase tracking-wide text-muted-foreground">
                <th scope="col" className="px-4 py-2">Date</th>
                <th scope="col" className="px-4 py-2">Merchant</th>
                <th scope="col" className="px-4 py-2 text-right">Amount</th>
                <th scope="col" className="px-4 py-2">Category</th>
                <th scope="col" className="px-4 py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {preview.sample_rows.map((row, i) => (
                <motion.tr
                  key={`${row.date}-${row.merchant}-${i}`}
                  initial={reduceMotion ? false : { opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.2, delay: reduceMotion ? 0 : i * 0.025 }}
                  className="border-b border-border last:border-0"
                >
                  <td className="whitespace-nowrap px-4 py-2 text-muted-foreground">
                    {formatDate(row.date)}
                  </td>
                  <td className="px-4 py-2 font-medium">{row.merchant}</td>
                  <td className="whitespace-nowrap px-4 py-2 text-right tabular-nums">
                    {formatCurrency(row.amount)}
                  </td>
                  <td className="px-4 py-2">
                    <CategoryCell row={row} />
                  </td>
                  <td className="px-4 py-2">
                    {row.is_duplicate ? (
                      <Badge variant="warning">Already imported</Badge>
                    ) : (
                      <Badge variant="success">New</Badge>
                    )}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>

        {error && (
          <p className="text-sm text-destructive" role="alert">
            {error}
          </p>
        )}
      </CardContent>
      <CardFooter className="justify-end gap-2">
        <Button variant="outline" onClick={onCancel} disabled={pending}>
          Cancel
        </Button>
        <Button onClick={onConfirm} disabled={pending}>
          {pending && <Loader2 className="h-4 w-4 animate-spin" />}
          Import {toImport} {toImport === 1 ? "transaction" : "transactions"}
        </Button>
      </CardFooter>
    </Card>
  );
}

/**
 * Shows the category the row will actually be filed under, and where it came
 * from.
 *
 * This exists because Preview and Confirm used to disagree: Preview showed
 * the raw model output while Confirm separately applied ambiguity routing and
 * remembered corrections. Both now run one shared decision, and this cell
 * makes that decision legible instead of showing a bare label.
 */
function CategoryCell({ row }: { row: ImportSampleRow }) {
  if (row.effective_category === null) {
    return <span className="text-muted-foreground">—</span>;
  }

  if (row.remembered_category) {
    return (
      <span className="flex flex-wrap items-center gap-1.5">
        <span className="text-xs text-muted-foreground line-through">{row.predicted_category}</span>
        <ArrowRight className="h-3 w-3 text-muted-foreground" aria-hidden />
        <span className="font-medium">{row.remembered_category}</span>
        <Badge variant="secondary" className="gap-1">
          <UserCheck className="h-3 w-3" aria-hidden />
          Your category
        </Badge>
      </span>
    );
  }

  const note = describeDecisionSource(row.decision_source);

  return (
    <span className="flex flex-wrap items-center gap-1.5">
      <span>{row.effective_category}</span>
      {note && (
        <span className="text-xs text-muted-foreground" title={note.explanation}>
          ({note.label})
        </span>
      )}
    </span>
  );
}

function Notice({
  children,
  tone = "muted",
  icon: Icon = AlertTriangle,
}: {
  children: React.ReactNode;
  tone?: "muted" | "warning" | "success";
  icon?: typeof AlertTriangle;
}) {
  return (
    <div
      className={cn(
        "flex items-start gap-2 rounded-md border px-3 py-2 text-sm",
        tone === "warning" && "border-warning/30 bg-warning/10 text-warning",
        tone === "success" && "border-success/30 bg-success/10 text-success",
        tone === "muted" && "border-border bg-muted/50 text-muted-foreground",
      )}
    >
      <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <span>{children}</span>
    </div>
  );
}

function Stat({
  label,
  value,
  emphasis,
}: {
  label: string;
  value: string | number;
  emphasis?: boolean;
}) {
  return (
    <div
      className={cn(
        "rounded-md px-3 py-2",
        emphasis ? "bg-primary/10 ring-1 ring-inset ring-primary/25" : "bg-muted/50",
      )}
    >
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className={cn("text-lg font-semibold tabular-nums", emphasis && "text-primary")}>{value}</p>
    </div>
  );
}
