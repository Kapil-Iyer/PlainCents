import { Outlet } from "react-router-dom";

import { DemoBanner } from "@/components/shared/DemoBanner";
import { DemoReentryBanner } from "@/components/shared/DemoReentryBanner";
import { Sidebar } from "@/components/layout/Sidebar";
import { TopNav } from "@/components/layout/TopNav";
import { TourOverlay } from "@/components/tour/TourOverlay";
import { GuidedTourProvider } from "@/context/GuidedTourContext";

export function AppShell() {
  return (
    <GuidedTourProvider>
      <div className="flex h-screen w-full overflow-hidden bg-background">
        <Sidebar />
        <div className="flex min-w-0 flex-1 flex-col">
          <TopNav />
          <DemoBanner />
          <DemoReentryBanner />
          <main className="flex-1 overflow-y-auto bg-background p-4 sm:p-6">
            <div className="mx-auto max-w-6xl">
              <Outlet />
            </div>
          </main>
        </div>
      </div>
      <TourOverlay />
    </GuidedTourProvider>
  );
}
