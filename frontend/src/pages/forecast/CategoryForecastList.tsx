import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { formatCurrency, formatMonthLabel } from "@/lib/utils";
import type { ForecastPrediction } from "@/types/forecast";

interface CategoryForecastListProps {
  predictions: ForecastPrediction[];
}

const OFFSETS = [1, 2, 3] as const;

/** PRD Section 11.8: per-category +1/+2/+3 display, with unavailable
 * categories explained (TRD Section 12.5) rather than shown as $0. */
export function CategoryForecastList({ predictions }: CategoryForecastListProps) {
  const byCategory = new Map<string, Map<number, ForecastPrediction>>();
  for (const p of predictions) {
    const row = byCategory.get(p.category) ?? new Map<number, ForecastPrediction>();
    row.set(p.month_offset, p);
    byCategory.set(p.category, row);
  }
  const categories = Array.from(byCategory.keys()).sort((a, b) => a.localeCompare(b));

  const monthHeader = (offset: number) => {
    const sample = predictions.find((p) => p.month_offset === offset);
    return sample ? formatMonthLabel(sample.forecast_month, "short") : `+${offset} mo`;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Per-category forecast</CardTitle>
        <CardDescription>Predicted spend for the next 3 months, by category</CardDescription>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left text-xs text-muted-foreground">
              <th className="py-2 pr-4 font-medium">Category</th>
              {OFFSETS.map((offset) => (
                <th key={offset} className="py-2 pr-4 font-medium">
                  {monthHeader(offset)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {categories.map((category) => {
              const row = byCategory.get(category)!;
              return (
                <tr key={category} className="border-b border-border last:border-0">
                  <td className="py-2 pr-4 font-medium">{category}</td>
                  {OFFSETS.map((offset) => {
                    const p = row.get(offset);
                    return (
                      <td key={offset} className="py-2 pr-4 tabular-nums">
                        {p?.is_available ? (
                          formatCurrency(p.predicted_amount ?? 0)
                        ) : (
                          <Badge
                            variant="outline"
                            className="whitespace-nowrap text-muted-foreground"
                            title={p?.unavailable_reason ?? undefined}
                          >
                            Not enough history
                          </Badge>
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}
