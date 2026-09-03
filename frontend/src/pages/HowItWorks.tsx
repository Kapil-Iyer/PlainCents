import { useEffect, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useLocation, useNavigate } from "react-router-dom";

import { cn } from "@/lib/utils";

import { CategorizationSection } from "@/pages/how-it-works/CategorizationSection";
import { EvaluationSection } from "@/pages/how-it-works/EvaluationSection";
import { ForecastingSection } from "@/pages/how-it-works/ForecastingSection";
import { HumanInLoopSection } from "@/pages/how-it-works/HumanInLoopSection";
import { LimitationsSection } from "@/pages/how-it-works/LimitationsSection";
import { PipelineDiagram } from "@/pages/how-it-works/PipelineDiagram";

const SECTIONS = [
  { id: "overview", label: "Overview" },
  { id: "categorization", label: "Categorization" },
  { id: "forecasting", label: "Forecasting" },
  { id: "human-in-the-loop", label: "Human-in-the-Loop" },
  { id: "evaluation", label: "Evaluation Methodology" },
  { id: "limitations", label: "Limitations & Evidence" },
] as const;

type SectionId = (typeof SECTIONS)[number]["id"];

function isSectionId(value: string): value is SectionId {
  return SECTIONS.some((s) => s.id === value);
}

/**
 * Transparent AI/ML methodology page (Phase 11B Deliverable B). In-page
 * tabs, not a separate tabbed app shell — Sidebar/AppShell are unchanged.
 * Reads the URL hash on mount/change so contextual links from Transactions
 * ("How was this predicted?" -> #categorization) and Forecast ("Why this
 * model?" -> #forecasting) land on the right section.
 */
export function HowItWorksPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const prefersReducedMotion = useReducedMotion();

  const hashId = location.hash.replace("#", "");
  const initial = isSectionId(hashId) ? hashId : "overview";
  const [active, setActive] = useState<SectionId>(initial);

  useEffect(() => {
    const id = location.hash.replace("#", "");
    if (isSectionId(id) && id !== active) {
      setActive(id);
    }
    // Only react to hash changes (external navigation into this page) —
    // `active` itself is intentionally excluded so a tab click below
    // doesn't get overridden by a stale hash on the next render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.hash]);

  const handleSelect = (id: SectionId) => {
    setActive(id);
    navigate(`/how-it-works#${id}`, { replace: true });
  };

  return (
    <div className="flex flex-col gap-5">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">How It Works</h1>
        <p className="text-sm text-muted-foreground">
          Transparent AI and ML methodology behind PlainCents — every number sourced from committed
          evaluation reports.
        </p>
      </div>

      <div role="tablist" aria-label="How It Works sections" className="flex flex-wrap gap-1 rounded-lg border border-border bg-card p-1">
        {SECTIONS.map((section) => (
          <button
            key={section.id}
            type="button"
            role="tab"
            aria-selected={active === section.id}
            onClick={() => handleSelect(section.id)}
            className={cn(
              "rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground",
              active === section.id && "bg-primary text-primary-foreground hover:bg-primary hover:text-primary-foreground",
            )}
          >
            {section.label}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={active}
          id={active}
          role="tabpanel"
          initial={prefersReducedMotion ? undefined : { opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={prefersReducedMotion ? undefined : { opacity: 0, y: -8 }}
          transition={{ duration: 0.2 }}
        >
          {active === "overview" && <PipelineDiagram />}
          {active === "categorization" && <CategorizationSection />}
          {active === "forecasting" && <ForecastingSection />}
          {active === "human-in-the-loop" && <HumanInLoopSection />}
          {active === "evaluation" && <EvaluationSection />}
          {active === "limitations" && <LimitationsSection />}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
