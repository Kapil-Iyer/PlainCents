import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useAvailableMonths } from "@/hooks/useDashboard";
import { formatMonthLabel } from "@/lib/utils";

function currentCalendarMonth(): string {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`;
}

interface AnalysisMonthSelectorProps {
  /** `undefined` means "no explicit user selection yet" — the backend
   * applies its own current-month-default resolution (today's calendar
   * month, unless it has no data and an earlier month does — see
   * backend.services.date_windows.resolve_default_analysis_month). Once the
   * user picks anything from this dropdown (including "This month"), that
   * choice is passed through explicitly from then on and always wins. */
  value: string | undefined;
  /** The month actually being shown right now (`summary.period.current`,
   * from the dashboard-summary response) — used to highlight the right
   * option when `value` is still undefined, since that resolved month may
   * be an earlier month, not necessarily today's calendar month. */
  resolvedMonth: string;
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
 * is always a valid, selectable option, and a user can always explicitly
 * pick today's own empty month even when the app defaults elsewhere.
 */
export function AnalysisMonthSelector({ value, resolvedMonth, onChange }: AnalysisMonthSelectorProps) {
  const { data } = useAvailableMonths();
  const thisMonth = currentCalendarMonth();
  const months = Array.from(new Set([thisMonth, ...(data?.months ?? [])])).sort((a, b) =>
    b.localeCompare(a),
  );

  // Nothing to choose between yet (a fresh, single-month dataset) — showing
  // a one-option dropdown would be noise, not a real control.
  if (months.length <= 1) return null;

  return (
    <Select value={value ?? resolvedMonth} onValueChange={onChange}>
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
