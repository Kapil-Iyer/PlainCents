import { useState } from "react";
import { Plus, Wallet } from "lucide-react";

import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/shared/EmptyState";
import { TableSkeleton } from "@/components/shared/LoadingState";
import { useToast } from "@/components/shared/Toast";
import { useHoldingsQuery, useRefreshPrices } from "@/hooks/useHoldings";
import { ApiError } from "@/types/common";

import { HoldingFormDialog } from "@/pages/portfolio/HoldingFormDialog";
import { HoldingsTable } from "@/pages/portfolio/HoldingsTable";
import { RefreshPricesButton } from "@/pages/portfolio/RefreshPricesButton";

export function PortfolioPage() {
  const [createOpen, setCreateOpen] = useState(false);
  const { data: holdings, isLoading, isError } = useHoldingsQuery();
  const refreshMutation = useRefreshPrices();
  const { toast } = useToast();

  const handleRefresh = () => {
    refreshMutation.mutate(undefined, {
      onSuccess: (result) => {
        // TRD §13.4: a per-ticker refresh failure is transient UI feedback,
        // never a persisted field — surfaced here from the response, not
        // fabricated or stored.
        if (result.failed.length > 0) {
          toast({
            title:
              result.failed.length === 1
                ? `Couldn't refresh ${result.failed[0].ticker}`
                : `Couldn't refresh ${result.failed.length} tickers`,
            description: result.failed.map((f) => f.ticker).join(", "),
            variant: "destructive",
          });
        } else {
          toast({ title: "Prices refreshed" });
        }
      },
      onError: (err) => {
        toast({
          title: "Couldn't refresh prices",
          description: err instanceof ApiError ? err.message : "Please try again.",
          variant: "destructive",
        });
      },
    });
  };

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Portfolio</h1>
          <p className="text-sm text-muted-foreground">
            Track your holdings and their latest known market value.
          </p>
        </div>
        <div className="flex gap-2">
          {holdings && holdings.length > 0 && (
            <RefreshPricesButton isPending={refreshMutation.isPending} onClick={handleRefresh} />
          )}
          <Button onClick={() => setCreateOpen(true)}>
            <Plus className="h-4 w-4" />
            Add holding
          </Button>
        </div>
      </div>

      {isLoading ? (
        <TableSkeleton rows={5} columns={7} />
      ) : isError ? (
        <EmptyState
          icon={Wallet}
          title="Couldn't load holdings"
          description="Something went wrong talking to the server. Try refreshing the page."
        />
      ) : !holdings || holdings.length === 0 ? (
        <EmptyState
          icon={Wallet}
          title="No holdings yet"
          description="Add a holding to start tracking its value. Prices only update when you click Refresh Prices."
          action={
            <Button onClick={() => setCreateOpen(true)}>
              <Plus className="h-4 w-4" />
              Add holding
            </Button>
          }
        />
      ) : (
        <HoldingsTable holdings={holdings} />
      )}

      <HoldingFormDialog open={createOpen} onOpenChange={setCreateOpen} />
    </div>
  );
}
