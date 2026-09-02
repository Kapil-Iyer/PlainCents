import { Search } from "lucide-react";

import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { CATEGORIES } from "@/constants/categories";
import type { TransactionListParams } from "@/types/transaction";

interface TransactionFiltersProps {
  filters: TransactionListParams;
  onChange: (next: TransactionListParams) => void;
}

// Matches backend/repositories/transaction_repository.py's `sort` parsing:
// a bare column name sorts ascending, a "-" prefix sorts descending.
// Allowed columns: date, amount, merchant, created_at.
const SORT_OPTIONS = [
  { value: "-date", label: "Date (newest first)" },
  { value: "date", label: "Date (oldest first)" },
  { value: "-amount", label: "Amount (highest first)" },
  { value: "amount", label: "Amount (lowest first)" },
];

export function TransactionFilters({ filters, onChange }: TransactionFiltersProps) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="relative min-w-56 flex-1">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          placeholder="Search merchant…"
          className="pl-8"
          value={filters.search ?? ""}
          onChange={(e) => onChange({ ...filters, search: e.target.value, page: 1 })}
        />
      </div>

      <Select
        value={filters.category ?? "all"}
        onValueChange={(value) =>
          onChange({ ...filters, category: value === "all" ? undefined : value, page: 1 })
        }
      >
        <SelectTrigger className="w-48">
          <SelectValue placeholder="All categories" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All categories</SelectItem>
          {CATEGORIES.map((c) => (
            <SelectItem key={c} value={c}>
              {c}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Input
        type="date"
        className="w-40"
        value={filters.date_from ?? ""}
        onChange={(e) => onChange({ ...filters, date_from: e.target.value || undefined, page: 1 })}
        aria-label="From date"
      />
      <Input
        type="date"
        className="w-40"
        value={filters.date_to ?? ""}
        onChange={(e) => onChange({ ...filters, date_to: e.target.value || undefined, page: 1 })}
        aria-label="To date"
      />

      <Select
        value={filters.sort ?? "-date"}
        onValueChange={(value) => onChange({ ...filters, sort: value, page: 1 })}
      >
        <SelectTrigger className="w-52">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {SORT_OPTIONS.map((opt) => (
            <SelectItem key={opt.value} value={opt.value}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
