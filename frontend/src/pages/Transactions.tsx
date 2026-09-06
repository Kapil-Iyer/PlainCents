import { useState } from "react";
import { Plus, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/shared/EmptyState";
import { Pagination } from "@/components/shared/Pagination";
import { TableSkeleton } from "@/components/shared/LoadingState";
import { useTransactionsQuery } from "@/hooks/useTransactions";
import type { TransactionListParams } from "@/types/transaction";

import { CategoryTrendCard } from "@/components/analytics/CategoryTrendCard";
import { TopMerchantsCard } from "@/components/analytics/TopMerchantsCard";
import { TransactionFilters } from "@/pages/transactions/TransactionFilters";
import { TransactionFormDialog } from "@/pages/transactions/TransactionFormDialog";
import { TransactionTable } from "@/pages/transactions/TransactionTable";
import {
  TabPanel,
  TransactionsTabs,
  type TransactionsTab,
} from "@/pages/transactions/TransactionsTabs";

const DEFAULT_FILTERS: TransactionListParams = { sort: "-date", page: 1, page_size: 25 };

export function TransactionsPage() {
  const [filters, setFilters] = useState<TransactionListParams>(DEFAULT_FILTERS);
  const [createOpen, setCreateOpen] = useState(false);
  const [tab, setTab] = useState<TransactionsTab>("list");

  const { data, isLoading, isError, isPlaceholderData } = useTransactionsQuery(filters);

  const hasActiveFilters =
    !!filters.search || !!filters.category || !!filters.date_from || !!filters.date_to;

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Transactions</h1>
          <p className="text-sm text-muted-foreground">
            View, correct, and manage every transaction currently in your account.
          </p>
          <Link
            to="/how-it-works#categorization"
            className="mt-1 inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
          >
            <Sparkles className="h-3 w-3" />
            How was this predicted?
          </Link>
        </div>
        <Button onClick={() => setCreateOpen(true)}>
          <Plus className="h-4 w-4" />
          Add transaction
        </Button>
      </div>

      <div data-tour="transactions-tabs">
        <TransactionsTabs active={tab} onChange={setTab} />
      </div>

      <TabPanel id="list" active={tab === "list"}>
        <div className="flex flex-col gap-5">
          <TransactionFilters filters={filters} onChange={setFilters} />

          {isLoading ? (
            <TableSkeleton rows={8} columns={5} />
          ) : isError ? (
            <EmptyState
              title="Couldn't load transactions"
              description="Something went wrong talking to the server. Try refreshing the page."
            />
          ) : !data || data.items.length === 0 ? (
            hasActiveFilters ? (
              <EmptyState
                title="No transactions match your filters"
                description="Try clearing the search, category, or date filters."
                action={
                  <Button variant="outline" onClick={() => setFilters(DEFAULT_FILTERS)}>
                    Clear filters
                  </Button>
                }
              />
            ) : (
              <EmptyState
                title="No transactions yet"
                description="Import a bank statement or add a transaction manually to get started."
                action={
                  <Button onClick={() => setCreateOpen(true)}>
                    <Plus className="h-4 w-4" />
                    Add transaction
                  </Button>
                }
              />
            )
          ) : (
            <div className={isPlaceholderData ? "opacity-60 transition-opacity" : undefined}>
              <TransactionTable transactions={data.items} />
              <Pagination
                page={data.page}
                pageSize={data.page_size}
                total={data.total}
                onPageChange={(page) => setFilters((f) => ({ ...f, page }))}
              />
            </div>
          )}
        </div>
      </TabPanel>

      <TabPanel id="insights" active={tab === "insights"}>
        {/* Analytics deliberately ignore the list's filters: the filters are
         * for finding a specific transaction, while these answer questions
         * about the whole picture and carry their own time-range controls.
         * Silently applying a search box to a 12-month trend would be a
         * quietly wrong chart. */}
        <div className="flex flex-col gap-5">
          <CategoryTrendCard />
          <TopMerchantsCard />
        </div>
      </TabPanel>

      <TransactionFormDialog open={createOpen} onOpenChange={setCreateOpen} />
    </div>
  );
}
