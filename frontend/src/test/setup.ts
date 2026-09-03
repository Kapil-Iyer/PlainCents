import "@testing-library/jest-dom/vitest";

// Radix UI's Select/Dialog primitives call these DOM APIs, which jsdom does
// not implement — polyfill them so component tests don't crash.
if (!Element.prototype.hasPointerCapture) {
  Element.prototype.hasPointerCapture = () => false;
}
if (!Element.prototype.setPointerCapture) {
  Element.prototype.setPointerCapture = () => {};
}
if (!Element.prototype.releasePointerCapture) {
  Element.prototype.releasePointerCapture = () => {};
}
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}

// Recharts' ResponsiveContainer (Dashboard charts, Phase 6) observes its
// container size via ResizeObserver, which jsdom does not implement.
if (typeof ResizeObserver === "undefined") {
  // @ts-expect-error -- minimal jsdom polyfill, not a real observer.
  global.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

// Phase 11B: Framer Motion's `whileInView` (How It Works reveals) observes
// via IntersectionObserver, which jsdom does not implement.
if (typeof IntersectionObserver === "undefined") {
  // @ts-expect-error -- minimal jsdom polyfill, not a real observer.
  global.IntersectionObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return [];
    }
  };
}

// The Phase 10 product walkthrough checks prefers-reduced-motion via
// matchMedia, which jsdom does not implement — default to "no preference".
if (typeof window !== "undefined" && !window.matchMedia) {
  window.matchMedia = (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }) as MediaQueryList;
}
