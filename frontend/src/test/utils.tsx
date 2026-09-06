import type { ReactElement, ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import { ToastHost } from "@/components/shared/Toast";
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
            <GuidedTourProvider>{children}</GuidedTourProvider>
          </MemoryRouter>
        </ToastHost>
      </QueryClientProvider>
    );
  }

  return render(ui, { wrapper: Wrapper });
}
