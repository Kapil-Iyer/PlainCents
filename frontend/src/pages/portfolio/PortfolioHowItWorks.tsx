import { Calculator, Info, RefreshCw, ShieldQuestion, Wallet } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const ITEMS: { icon: typeof Wallet; title: string; body: string }[] = [
  {
    icon: Wallet,
    title: "Current value",
    body: "Shares × the latest known market price. Available as soon as you've refreshed prices at least once, whether or not you know your cost basis.",
  },
  {
    icon: Calculator,
    title: "Average cost & cost basis",
    body: "Average cost is the price you paid per share, on average. Cost basis is shares × average cost. Both are optional -- PlainCents never fabricates one from the current price or any default.",
  },
  {
    icon: ShieldQuestion,
    title: "Unrealized P&L",
    body: "Current value − cost basis. If average cost is unknown, P&L can't be honestly calculated -- value still works, and you can add a cost basis (typed directly, or calculated from your purchases) at any time.",
  },
  {
    icon: RefreshCw,
    title: "Price freshness",
    body: "Refreshing prices updates current price, value, and P&L (when cost is known) -- it never changes shares or average cost. A demo holding's price is labeled \"Demo snapshot\" until you refresh it with a real quote; a real holding always shows the timestamp of its last genuine fetch.",
  },
];

/**
 * A concise Portfolio-specific explainer (Portfolio + Power BI completion
 * pass) -- kept short and scoped to what a viewer of THIS page needs, not a
 * restatement of the global How It Works page's own Portfolio mention.
 */
export function PortfolioHowItWorks() {
  return (
    <div data-tour="portfolio-how-it-works" className="flex flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">How your portfolio works</h2>
        <p className="text-sm text-muted-foreground">The math behind every number above.</p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {ITEMS.map((item) => (
          <Card key={item.title}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm font-semibold">
                <item.icon className="h-4 w-4 text-primary" aria-hidden />
                {item.title}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <p className="text-sm leading-relaxed text-muted-foreground">{item.body}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="border-border-strong/60 bg-elevated">
        <CardContent className="flex gap-2.5 pt-6">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
          <p className="text-xs leading-relaxed text-muted-foreground">
            Portfolio holdings are entirely separate from spending: they never affect your
            transaction totals, category analytics, Spending Pace, What Changed, or spending
            forecasts. Portfolio tracking here is informational only and does not provide
            investment advice or recommendations.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
