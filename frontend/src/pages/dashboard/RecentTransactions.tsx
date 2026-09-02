import { Link } from "react-router-dom";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/shared/EmptyState";
import { formatCurrency, formatDate } from "@/lib/utils";
import type { TransactionResponse } from "@/types/transaction";

import { CategoryBadge } from "@/pages/transactions/CategoryBadge";

interface RecentTransactionsProps {
  transactions: TransactionResponse[];
}

/** PRD §11.7: "a list of recent transactions." */
export function RecentTransactions({ transactions }: RecentTransactionsProps) {
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle>Recent transactions</CardTitle>
        <Link to="/transactions" className="text-xs font-medium text-primary hover:underline">
          View all
        </Link>
      </CardHeader>
      <CardContent>
        {transactions.length === 0 ? (
          <EmptyState
            title="No transactions yet"
            description="Recent activity will show up here once you import or add a transaction."
            className="border-none py-10"
          />
        ) : (
          <ul className="flex flex-col divide-y divide-border">
            {transactions.map((txn) => (
              <li key={txn.id} className="flex items-center justify-between gap-3 py-2.5 text-sm">
                <div className="flex min-w-0 flex-col">
                  <span className="truncate font-medium">{txn.merchant}</span>
                  <span className="text-xs text-muted-foreground">{formatDate(txn.date)}</span>
                </div>
                <CategoryBadge transaction={txn} />
                <span className="whitespace-nowrap font-medium tabular-nums">
                  {formatCurrency(txn.amount)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
