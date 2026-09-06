import { useState } from "react";
import { Plus, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { formatCurrency } from "@/lib/utils";
import { weightedAverageCost } from "@/lib/portfolioMath";

interface LotRow {
  id: number;
  shares: string;
  price: string;
}

let nextRowId = 0;
function emptyRow(): LotRow {
  return { id: nextRowId++, shares: "", price: "" };
}

interface PurchaseLotCalculatorProps {
  onApply: (avgCost: number) => void;
  onCancel: () => void;
}

/**
 * "Calculate average cost" helper (Portfolio + Power BI completion pass).
 *
 * NOT tax-lot accounting: nothing entered here is persisted as its own
 * record. This is a convenience calculator only -- it computes one
 * generic weighted average (sum(shares_i * price_i) / sum(shares_i)) from
 * whatever purchase rows the user types, and the caller stores only the
 * resulting number as the holding's avg_cost, exactly as if the user had
 * typed that average in directly.
 */
export function PurchaseLotCalculator({ onApply, onCancel }: PurchaseLotCalculatorProps) {
  const [rows, setRows] = useState<LotRow[]>(() => [emptyRow(), emptyRow()]);

  const parsedLots = rows.map((r) => ({ shares: Number(r.shares), price: Number(r.price) }));
  const average = weightedAverageCost(parsedLots);

  const updateRow = (id: number, field: "shares" | "price", value: string) => {
    setRows((prev) => prev.map((r) => (r.id === id ? { ...r, [field]: value } : r)));
  };

  const addRow = () => setRows((prev) => [...prev, emptyRow()]);
  const removeRow = (id: number) => setRows((prev) => prev.filter((r) => r.id !== id));

  return (
    <div className="flex flex-col gap-3 rounded-md border border-border bg-muted/30 p-3">
      <p className="text-xs text-muted-foreground">
        Enter each purchase (shares and price per share). This calculates a weighted average --
        it doesn&apos;t save these individual purchases anywhere.
      </p>

      <div className="flex flex-col gap-2">
        {rows.map((row, i) => (
          <div key={row.id} className="flex items-end gap-2">
            <div className="flex-1">
              {i === 0 && <Label className="text-xs">Shares</Label>}
              <Input
                type="number"
                step="any"
                min="0"
                placeholder="5"
                value={row.shares}
                onChange={(e) => updateRow(row.id, "shares", e.target.value)}
              />
            </div>
            <div className="flex-1">
              {i === 0 && <Label className="text-xs">Price/share</Label>}
              <Input
                type="number"
                step="0.01"
                min="0"
                placeholder="100.00"
                value={row.price}
                onChange={(e) => updateRow(row.id, "price", e.target.value)}
              />
            </div>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label="Remove purchase"
              disabled={rows.length <= 1}
              onClick={() => removeRow(row.id)}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        ))}
      </div>

      <Button type="button" variant="outline" size="sm" onClick={addRow} className="self-start">
        <Plus className="h-4 w-4" />
        Add purchase
      </Button>

      <div className="flex items-center justify-between border-t border-border pt-3">
        <div>
          <p className="text-xs text-muted-foreground">Calculated average cost</p>
          <p className="text-lg font-semibold tabular-nums">
            {average === null ? "—" : formatCurrency(average)}
          </p>
        </div>
        <div className="flex gap-2">
          <Button type="button" variant="ghost" size="sm" onClick={onCancel}>
            Cancel
          </Button>
          <Button
            type="button"
            size="sm"
            disabled={average === null}
            onClick={() => average !== null && onApply(average)}
          >
            Use this average cost
          </Button>
        </div>
      </div>
    </div>
  );
}
