import type { ReactElement, ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import { ToastHost } from "@/components/shared/Toast";
import { AppStateProvider } from "@/context/AppStateContext";
import { GuidedTourProvider } from "@/context/GuidedTourContext";

export function renderWithProviders(ui: ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ToastHost>
          <MemoryRouter>
            {/* Real AppStateProvider, not a mock -- its /api/demo/status
             * query is unmocked here and simply never resolves in a page
             * test that doesn't stub @/api/health, which is fine: mode
             * falls back to its documented "EMPTY" default (see
             * AppStateContext) rather than throwing. Only added because
             * PortfolioPage now reads useAppState() directly (Demo price
             * notice); every other caller of renderWithProviders is
             * unaffected. */}
            <AppStateProvider>
              <GuidedTourProvider>{children}</GuidedTourProvider>
            </AppStateProvider>
          </MemoryRouter>
        </ToastHost>
      </QueryClientProvider>
    );
  }

  return render(ui, { wrapper: Wrapper });
}
