import { useState } from "react";
import { Download, ExternalLink, Loader2 } from "lucide-react";
import { useMutation } from "@tanstack/react-query";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useToast } from "@/components/shared/Toast";
import { downloadPowerBIExport } from "@/api/export";
import { ApiError } from "@/types/common";

import { Disclosure } from "@/pages/how-it-works/Disclosure";

/**
 * PATCH D (Power BI + Portfolio completion pass): a small guided workflow,
 * not a single confusing one-click button. PlainCents generates a
 * point-in-time SNAPSHOT of the live database on demand -- Power BI itself
 * is never live-connected to the app, nothing here launches Power BI
 * Desktop, and no .pbix is generated programmatically. No "Download
 * template" button is offered: audited, no real .pbix/.pbit exists in this
 * repository (see powerbi/v2/README.md) -- claiming otherwise would be
 * dishonest. The one real, safe, reusable Power BI artifact that DOES exist
 * (a color theme) is offered explicitly labeled as a theme, never as a
 * template.
 */
export function ExportPowerBIButton() {
  const [open, setOpen] = useState(false);
  const { toast } = useToast();
  const mutation = useMutation({
    mutationFn: downloadPowerBIExport,
    onError: (err) => {
      toast({
        title: "Couldn't generate the export",
        description: err instanceof ApiError ? err.message : "Please try again.",
        variant: "destructive",
      });
    },
  });

  return (
    <>
      <Button data-tour="export-powerbi" variant="outline" size="sm" onClick={() => setOpen(true)}>
        <Download className="h-4 w-4" />
        Export for Power BI
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Export for Power BI</DialogTitle>
            <DialogDescription>
              PlainCents generates a Power BI-ready snapshot of your current data. Power BI itself
              is never live-connected — refresh the file after downloading a new snapshot to update
              your dashboard there.
            </DialogDescription>
          </DialogHeader>

          <ol className="flex flex-col gap-2 text-sm text-muted-foreground">
            <li>1. Download your data pack below.</li>
            <li>2. Open the setup guide and load the tables into Power BI Desktop.</li>
            <li>3. Refresh Power BI whenever you download a new snapshot.</li>
          </ol>

          <div className="flex flex-wrap gap-2">
            <Button onClick={() => mutation.mutate()} disabled={mutation.isPending}>
              {mutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Download className="h-4 w-4" />
              )}
              Download data pack
            </Button>
            <Button variant="outline" asChild>
              <a href="/powerbi_theme.json" download="plaincents_theme.json">
                <Download className="h-4 w-4" />
                Download Power BI theme (optional)
              </a>
            </Button>
          </div>

          <Disclosure summary="View setup guide">
            <p>
              No PlainCents Power BI template exists yet, so this walks through building the report
              once — it takes a few minutes, and future updates only need Refresh.
            </p>
            <ol className="flex flex-col gap-1.5">
              <li>1. Extract the downloaded ZIP to a folder you'll remember.</li>
              <li>
                2. In Power BI Desktop: <strong className="text-foreground">Get Data → Text/CSV</strong>{" "}
                (or <strong className="text-foreground">Get Data → Folder</strong>) and load
                transactions.csv, category_summary.csv, portfolio.csv, and forecast.csv.
              </li>
              <li>
                3. A blank <code className="text-foreground">avg_cost</code> or{" "}
                <code className="text-foreground">pnl</code> cell means "unknown," not zero — Power
                BI's SUM/AVERAGE already skip blanks correctly.
              </li>
              <li>
                4. Optionally apply the downloaded theme: View → Themes → Browse for themes.
              </li>
              <li>
                5. Next time: download a new data pack, extract it over the same folder, and click
                Refresh in Power BI Desktop — no rebuilding needed.
              </li>
            </ol>
            <p className="flex items-center gap-1.5 text-xs">
              <ExternalLink className="h-3 w-3" aria-hidden />
              Full column-by-column schema and suggested visuals: <code>powerbi/v2/</code> in the
              PlainCents repository.
            </p>
          </Disclosure>
        </DialogContent>
      </Dialog>
    </>
  );
}
