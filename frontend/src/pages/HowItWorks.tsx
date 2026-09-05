import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { cn } from "@/lib/utils";

import { AppWalkthroughSection } from "@/pages/how-it-works/AppWalkthroughSection";
import { DecisionJourneySection } from "@/pages/how-it-works/DecisionJourneySection";
import { ForecastExplainerSection } from "@/pages/how-it-works/ForecastExplainerSection";
import { IntroSection } from "@/pages/how-it-works/IntroSection";
import { MemorySection } from "@/pages/how-it-works/MemorySection";
import { MlgEvaluationSection } from "@/pages/how-it-works/MlgEvaluationSection";
import { MlgLimitationsSection } from "@/pages/how-it-works/MlgLimitationsSection";
import { PipelineDiagram } from "@/pages/how-it-works/PipelineDiagram";
import { VideoWalkthroughSection } from "@/pages/how-it-works/VideoWalkthroughSection";

/**
 * Section ids are also URL hashes. Two of them — `categorization` and
 * `forecasting` — are load-bearing: Transactions links to
 * `#categorization` ("How was this predicted?") and Forecast links to
 * `#forecasting` ("Why this model?"). Renaming either breaks a link from
 * another page, so the old names are kept even where a more descriptive one
 * exists.
 */
const SECTIONS = [
  { id: "overview", label: "What is PlainCents?" },
  { id: "walkthrough", label: "Using the app" },
  { id: "video", label: "Video" },
  { id: "categorization", label: "Categorization" },
  { id: "memory", label: "Your corrections" },
  { id: "forecasting", label: "Forecasting" },
  { id: "evaluation", label: "Evidence" },
  { id: "limitations", label: "Limitations" },
] as const;

type SectionId = (typeof SECTIONS)[number]["id"];

function isSectionId(value: string): value is SectionId {
  return SECTIONS.some((s) => s.id === value);
}

/**
 * How It Works: a single scrolled page with a sticky section rail, rather
 * than the tab panels this page used to be.
 *
 * Tabs were the wrong shape for this content. The sections build on each
 * other — the premise explains the walkthrough, the walkthrough motivates
 * the categorization pipeline, the pipeline is what the evaluation
 * measures, and the limitations qualify all of it — and tabs actively hide
 * that ordering by making every section look like a peer you might skip.
 * One scroll with a rail keeps the narrative while still allowing a jump
 * straight to a section from another page's deep link.
 */
export function HowItWorksPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [active, setActive] = useState<SectionId>("overview");
  // Set while a click-driven scroll is in flight, so the observer below
  // doesn't fight it by highlighting every section the scroll passes over.
  const scrollingTo = useRef<SectionId | null>(null);

  const scrollToSection = useCallback((id: SectionId) => {
    const el = document.getElementById(id);
    if (!el) return;
    scrollingTo.current = id;
    setActive(id);
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    window.setTimeout(() => {
      scrollingTo.current = null;
    }, 600);
  }, []);

  // Deep links from Transactions/Forecast land here with a hash.
  useEffect(() => {
    const id = location.hash.replace("#", "");
    if (isSectionId(id)) {
      // rAF so the section elements exist before we measure them.
      requestAnimationFrame(() => scrollToSection(id));
    }
  }, [location.hash, scrollToSection]);

  // Highlight whichever section is nearest the top of the viewport.
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (scrollingTo.current) return;
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (visible && isSectionId(visible.target.id)) {
          setActive(visible.target.id);
        }
      },
      // A band across the upper-middle of the viewport: a section counts as
      // "current" once its heading has cleared the top, not when its last
      // pixel leaves the bottom.
      { rootMargin: "-96px 0px -55% 0px", threshold: 0 },
    );

    for (const section of SECTIONS) {
      const el = document.getElementById(section.id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, []);

  const handleSelect = (id: SectionId) => {
    scrollToSection(id);
    navigate(`/how-it-works#${id}`, { replace: true });
  };

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">How It Works</h1>
        <p className="text-sm text-muted-foreground">
          What PlainCents does, how it decides, and what it can&apos;t tell you.
        </p>
      </div>

      {/* Sticky rail.
       *
       * The negative `top` is load-bearing, not a nudge. AppShell's <main> is
       * the scroll container and carries p-4/sm:p-6, and a sticky offset
       * resolves against that padded content box — so a plain `top-0` parks
       * the rail 16/24px below the viewport edge and leaves a strip of
       * scrolled content visible above it. Matching the container's padding
       * exactly pulls the rail flush to the top of the scrollport, which is
       * where a sticky nav has to sit to actually cover what passes under it.
       * These values must stay in step with AppShell's padding. */}
      <nav
        aria-label="Page sections"
        className="sticky -top-4 z-10 -mx-1 overflow-x-auto rounded-lg border border-border bg-card/95 px-1 py-1 backdrop-blur supports-[backdrop-filter]:bg-card/80 sm:-top-6"
      >
        <ul className="flex min-w-max gap-1">
          {SECTIONS.map((section) => (
            <li key={section.id}>
              <button
                type="button"
                onClick={() => handleSelect(section.id)}
                aria-current={active === section.id ? "true" : undefined}
                className={cn(
                  "whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  active === section.id
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                )}
              >
                {section.label}
              </button>
            </li>
          ))}
        </ul>
      </nav>

      {/* scroll-mt clears the sticky rail so a jumped-to heading isn't
       * hidden underneath it. */}
      <Section id="overview">
        <IntroSection />
      </Section>

      <Section id="walkthrough">
        <AppWalkthroughSection />
      </Section>

      <Section id="video">
        <VideoWalkthroughSection />
      </Section>

      <Section id="categorization">
        <div className="flex flex-col gap-6">
          <PipelineDiagram />
          <DecisionJourneySection />
        </div>
      </Section>

      <Section id="memory">
        <MemorySection />
      </Section>

      <Section id="forecasting">
        <ForecastExplainerSection />
      </Section>

      <Section id="evaluation">
        <MlgEvaluationSection />
      </Section>

      <Section id="limitations">
        <MlgLimitationsSection />
      </Section>
    </div>
  );
}

function Section({ id, children }: { id: SectionId; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20">
      {children}
    </section>
  );
}
