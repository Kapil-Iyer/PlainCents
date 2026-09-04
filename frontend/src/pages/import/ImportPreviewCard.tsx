import { AlertTriangle, Loader2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { formatCurrency, formatDate } from "@/lib/utils";
import type { ImportPreview } from "@/types/import";

interface ImportPreviewCardProps {
  preview: ImportPreview;
  onConfirm: () => void;
  onCancel: () => void;
  pending: boolean;
  error?: string | null;
}

export function ImportPreviewCard({ preview, onConfirm, onCancel, pending, error }: ImportPreviewCardProps) {
  const hasExclusions = preview.rows_skipped_credit > 0 || preview.rows_skipped_currency > 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Preview</CardTitle>
        <p className="text-sm text-muted-foreground">
          Detected format: <span className="font-medium text-foreground">{preview.detected_bank}</span>
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Valid rows" value={preview.rows_valid} />
          <Stat label="Duplicates" value={preview.rows_duplicate} />
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
              <Stat label="Credits excluded" value={preview.rows_skipped_credit} />
            )}
            {preview.rows_skipped_currency > 0 && (
              <Stat label="Unsupported currency" value={preview.rows_skipped_currency} />
            )}
          </div>
        )}

        {!preview.categorization_available && (
          <div className="flex items-center gap-2 rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-sm text-warning">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            The categorization model is unavailable right now. Rows can still be previewed, but
            confirming this import will be blocked until the model is back — nothing will be
            imported without a category.
          </div>
        )}

        {preview.rows_unparseable > 0 && (
          <div className="flex items-center gap-2 rounded-md border border-border bg-muted/50 px-3 py-2 text-sm text-muted-foreground">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            {preview.rows_unparseable} row{preview.rows_unparseable === 1 ? "" : "s"} couldn't be
            parsed and will be skipped. This can happen with some export formats inconsistently
            (e.g. certain deposit-only rows) — it doesn't necessarily mean the file itself is
            broken.
          </div>
        )}

        {hasExclusions && (
          <div className="flex items-center gap-2 rounded-md border border-border bg-muted/50 px-3 py-2 text-sm text-muted-foreground">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            {preview.rows_skipped_credit > 0 &&
              `${preview.rows_skipped_credit} credit/deposit row${preview.rows_skipped_credit === 1 ? "" : "s"} ${preview.rows_skipped_credit === 1 ? "was" : "were"} recognized and intentionally excluded — PlainCents tracks spending, not income. `}
            {preview.rows_skipped_currency > 0 &&
              `${preview.rows_skipped_currency} row${preview.rows_skipped_currency === 1 ? "" : "s"} in an unsupported currency ${preview.rows_skipped_currency === 1 ? "was" : "were"} excluded rather than converted.`}
          </div>
        )}

        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full min-w-[600px] text-sm">
            <thead>
              <tr className="border-b border-border bg-muted/50 text-left text-xs font-medium uppercase tracking-wide text-muted-foreground">
                <th className="px-4 py-2">Date</th>
                <th className="px-4 py-2">Merchant</th>
                <th className="px-4 py-2 text-right">Amount</th>
                <th className="px-4 py-2">Predicted category</th>
                <th className="px-4 py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {preview.sample_rows.map((row, i) => (
                <tr key={i} className="border-b border-border last:border-0">
                  <td className="whitespace-nowrap px-4 py-2 text-muted-foreground">{formatDate(row.date)}</td>
                  <td className="px-4 py-2 font-medium">{row.merchant}</td>
                  <td className="whitespace-nowrap px-4 py-2 text-right tabular-nums">
                    {formatCurrency(row.amount)}
                  </td>
                  <td className="px-4 py-2 text-muted-foreground">
                    {row.predicted_category ?? "—"}
                  </td>
                  <td className="px-4 py-2">
                    {row.is_duplicate ? (
                      <Badge variant="warning">Duplicate</Badge>
                    ) : (
                      <Badge variant="success">New</Badge>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {error && <p className="text-sm text-destructive">{error}</p>}
      </CardContent>
      <CardFooter className="justify-end gap-2">
        <Button variant="outline" onClick={onCancel} disabled={pending}>
          Cancel
        </Button>
        <Button onClick={onConfirm} disabled={pending}>
          {pending && <Loader2 className="h-4 w-4 animate-spin" />}
          Confirm import
        </Button>
      </CardFooter>
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md bg-muted/50 px-3 py-2">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-lg font-semibold tabular-nums">{value}</p>
    </div>
  );
}
