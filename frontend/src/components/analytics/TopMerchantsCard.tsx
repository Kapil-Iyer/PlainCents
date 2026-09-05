import { useState } from "react";
import { motion, useReducedMotion } from "framer-motion";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/shared/EmptyState";
import { colorForCategory } from "@/constants/chartColors";
import { useTopMerchants } from "@/hooks/useAnalytics";
import { formatCurrency, formatDate } from "@/lib/utils";

import { ChartCardSkeleton, SegmentedControl } from "@/components/analytics/primitives";

const WINDOWS = [3, 6, 12] as const;
type Window = (typeof WINDOWS)[number];
const LIMIT = 8;

/**
 * "Which specific places take the biggest share of my money?"
 *
 * Category totals answer where money goes in the abstract; this answers it
 * concretely, which is usually the more actionable of the two.
 *
 * Merchants are grouped by their stable identity, not by raw description.
 * That matters more than it sounds: a real bank embeds a different card
 * suffix or store number on every transaction, so grouping by the raw text
 * would split one pharmacy into a dozen single-transaction rows and it would
 * never appear in a top-N list at all.
 */
export function TopMerchantsCard() {
  const [months, setMonths] = useState<Window>(6);
  const { data, isLoading, isError } = useTopMerchants(LIMIT, months);
  const reduceMotion = useReducedMotion();

  if (isLoading) return <ChartCardSkeleton title="Top merchants" />;
  if (isError || !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Top merchants</CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState
            title="Couldn't load top merchants"
            description="Something went wrong talking to the server."
          />
        </CardContent>
      </Card>
    );
  }

  const largest = Math.max(...data.items.map((i) => i.total_spend), 1);

  return (
    <Card>
      <CardHeader className="gap-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <CardTitle>Top merchants</CardTitle>
            <CardDescription>
              {data.items.length > 0
                ? `These ${data.items.length} account for ${data.top_n_share_pct}% of ${formatCurrency(data.total_spend)} across ${data.distinct_merchants} merchants.`
                : "Where your money actually goes, merchant by merchant."}
            </CardDescription>
          </div>
          <SegmentedControl
            label="Time range"
            value={String(months)}
            onChange={(v) => setMonths(Number(v) as Window)}
            options={WINDOWS.map((w) => ({ value: String(w), label: `${w}m` }))}
          />
        </div>
      </CardHeader>
      <CardContent>
        {data.items.length === 0 ? (
          <EmptyState
            title="No merchants in this period"
            description="Import a bank statement or widen the time range."
          />
        ) : (
          <ul className="flex flex-col gap-3">
            {data.items.map((item, i) => (
              <li key={item.merchant_key ?? item.merchant} className="flex flex-col gap-1.5">
                <div className="flex items-baseline justify-between gap-3">
                  <span className="min-w-0 truncate text-sm font-medium" title={item.merchant}>
                    {item.merchant}
                  </span>
                  <span className="shrink-0 text-sm tabular-nums">
                    {formatCurrency(item.total_spend)}
                  </span>
                </div>
                <div
                  className="h-1.5 overflow-hidden rounded-full bg-muted"
                  role="img"
                  aria-label={`${item.merchant}: ${formatCurrency(item.total_spend)}, ${item.pct_of_total}% of spend`}
                >
                  <motion.span
                    className="block h-full rounded-full"
                    style={{ background: colorForCategory(item.category ?? "Other") }}
                    initial={
                      reduceMotion
                        ? { width: `${(item.total_spend / largest) * 100}%` }
                        : { width: 0 }
                    }
                    animate={{ width: `${(item.total_spend / largest) * 100}%` }}
                    transition={{ duration: 0.35, delay: reduceMotion ? 0 : i * 0.04 }}
                  />
                </div>
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                  {item.category && <Badge variant="secondary">{item.category}</Badge>}
                  <span className="tabular-nums">
                    {item.transaction_count}{" "}
                    {item.transaction_count === 1 ? "transaction" : "transactions"}
                  </span>
                  <span className="tabular-nums">
                    {formatCurrency(item.average_transaction)} avg
                  </span>
                  <span className="tabular-nums">{item.pct_of_total}% of spend</span>
                  {item.last_seen && <span>last {formatDate(item.last_seen)}</span>}
                </div>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
