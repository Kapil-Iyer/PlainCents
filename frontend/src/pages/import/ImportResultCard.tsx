import { CheckCircle2 } from "lucide-react";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ImportResult } from "@/types/import";

export function ImportResultCard({ result, onImportAnother }: { result: ImportResult; onImportAnother: () => void }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <CheckCircle2 className="h-4 w-4 text-success" />
          Import complete
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="grid grid-cols-3 gap-4">
          <Stat label="Imported" value={result.rows_imported} />
          <Stat label="Skipped (duplicate)" value={result.rows_skipped_duplicate} />
          <Stat label="Skipped (unparseable)" value={result.rows_skipped_unparseable} />
        </div>
        {(result.rows_skipped_credit > 0 || result.rows_skipped_currency > 0) && (
          <div className="grid grid-cols-2 gap-4">
            {result.rows_skipped_credit > 0 && (
              <Stat label="Credits excluded" value={result.rows_skipped_credit} />
            )}
            {result.rows_skipped_currency > 0 && (
              <Stat label="Unsupported currency" value={result.rows_skipped_currency} />
            )}
          </div>
        )}
        <div className="flex gap-2">
          <Button asChild>
            <Link to="/transactions">View transactions</Link>
          </Button>
          <Button variant="outline" onClick={onImportAnother}>
            Import another file
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md bg-muted/50 px-3 py-2">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-lg font-semibold tabular-nums">{value}</p>
    </div>
  );
}
