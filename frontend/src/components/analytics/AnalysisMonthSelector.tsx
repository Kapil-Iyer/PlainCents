import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useAvailableMonths } from "@/hooks/useDashboard";
import { formatMonthLabel } from "@/lib/utils";

function currentCalendarMonth(): string {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`;
}

interface AnalysisMonthSelectorProps {
  /** `undefined` means "the current calendar month" (the default). */
  value: string | undefined;
  onChange: (month: string | undefined) => void;
}

/**
 * The ONE shared analysis-month clock: this single control drives the
 * Change KPI, Spending Pace, and Category Movers together (product
 * decision — never one selector per card, see Dashboard.tsx).
 *
 * Offers every month the user actually has data in (never an arbitrary,
 * possibly-empty calendar picker), plus the true current calendar month
 * even before it has any transactions of its own, so the default selection
 * is always a valid, selectable option.
 */
export function AnalysisMonthSelector({ value, onChange }: AnalysisMonthSelectorProps) {
  const { data } = useAvailableMonths();
  const thisMonth = currentCalendarMonth();
  const months = Array.from(new Set([thisMonth, ...(data?.months ?? [])])).sort((a, b) =>
    b.localeCompare(a),
  );

  // Nothing to choose between yet (a fresh, single-month dataset) — showing
  // a one-option dropdown would be noise, not a real control.
  if (months.length <= 1) return null;

  return (
    <Select value={value ?? thisMonth} onValueChange={(v) => onChange(v === thisMonth ? undefined : v)}>
      <SelectTrigger className="w-44">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {months.map((month) => (
          <SelectItem key={month} value={month}>
            {month === thisMonth ? "This month" : formatMonthLabel(month)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
