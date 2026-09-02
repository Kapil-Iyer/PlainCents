import type { LucideIcon } from "lucide-react";

import { EmptyState } from "@/components/shared/EmptyState";

interface PlaceholderPageProps {
  title: string;
  icon: LucideIcon;
  phase: number;
}

/** Dashboard/Forecast/Portfolio land in Phases 6-8 (Build Plan §8). */
export function PlaceholderPage({ title, icon, phase }: PlaceholderPageProps) {
  return (
    <div>
      <h1 className="mb-6 text-xl font-semibold">{title}</h1>
      <EmptyState
        icon={icon}
        title={`${title} is coming in Phase ${phase}`}
        description="This section isn't built yet. Import your transactions and manage them from the Transactions page in the meantime."
      />
    </div>
  );
}
