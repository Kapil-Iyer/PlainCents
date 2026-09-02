import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { DemoBanner } from "@/components/shared/DemoBanner";

const useAppState = vi.fn();
vi.mock("@/context/AppStateContext", () => ({
  useAppState: () => useAppState(),
}));

describe("DemoBanner", () => {
  it("renders while mode is DEMO", () => {
    useAppState.mockReturnValue({ mode: "DEMO" });

    render(<DemoBanner />);

    expect(screen.getByText(/Demo Data/)).toBeInTheDocument();
  });

  it("renders nothing when mode is EMPTY", () => {
    useAppState.mockReturnValue({ mode: "EMPTY" });

    const { container } = render(<DemoBanner />);

    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when mode is REAL", () => {
    useAppState.mockReturnValue({ mode: "REAL" });

    const { container } = render(<DemoBanner />);

    expect(container).toBeEmptyDOMElement();
  });
});
