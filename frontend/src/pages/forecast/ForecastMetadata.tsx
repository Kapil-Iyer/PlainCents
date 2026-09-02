interface ForecastMetadataProps {
  generatedAt: string;
  monthsAvailable: number;
}

/** TRD Section 5.6/Section 6: the "generated at" timestamp is required
 * context for a forecast that's only refreshed on demand, not live. */
export function ForecastMetadata({ generatedAt, monthsAvailable }: ForecastMetadataProps) {
  const date = new Date(generatedAt);
  const label = Number.isNaN(date.getTime())
    ? generatedAt
    : date.toLocaleString("en-CA", { dateStyle: "medium", timeStyle: "short" });

  return (
    <p className="text-xs text-muted-foreground">
      Generated {label} · based on {monthsAvailable} {monthsAvailable === 1 ? "month" : "months"} of history
    </p>
  );
}
