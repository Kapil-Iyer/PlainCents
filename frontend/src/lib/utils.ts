import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Format a number as CAD currency, e.g. -42.50 -> "-$42.50". */
export function formatCurrency(amount: number): string {
  return amount.toLocaleString("en-CA", {
    style: "currency",
    currency: "CAD",
  });
}

/** Format a "YYYY-MM-DD" date string for display without timezone drift. */
export function formatDate(date: string): string {
  const [year, month, day] = date.split("-").map(Number);
  return new Date(year, month - 1, day).toLocaleDateString("en-CA", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/** Format a "YYYY-MM" month string, e.g. "2026-09" -> "September 2026". */
export function formatMonthLabel(month: string, style: "long" | "short" = "long"): string {
  const [year, m] = month.split("-").map(Number);
  return new Date(year, m - 1, 1).toLocaleDateString("en-CA", {
    year: style === "long" ? "numeric" : "2-digit",
    month: style,
  });
}
