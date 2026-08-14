import "@testing-library/jest-dom/vitest";

/**
 * jsdom performs no layout, so every element measures 0x0 and has no
 * ResizeObserver. The virtualized run table asks its scroll container how tall it
 * is and would render zero rows, making assertions about row content fail for a
 * reason that has nothing to do with the component.
 *
 * These shims give the test environment a fixed viewport. They are deliberately
 * confined to the test setup -- production measurement is the browser's job.
 */

const VIEWPORT_HEIGHT = 640;
const VIEWPORT_WIDTH = 1200;

if (!("ResizeObserver" in globalThis)) {
  class ResizeObserverStub implements ResizeObserver {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  }
  globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;
}

Object.defineProperty(HTMLElement.prototype, "offsetHeight", {
  configurable: true,
  get(this: HTMLElement) {
    return this.classList.contains("table__body") ? VIEWPORT_HEIGHT : VIEWPORT_HEIGHT;
  },
});

Object.defineProperty(HTMLElement.prototype, "offsetWidth", {
  configurable: true,
  get: () => VIEWPORT_WIDTH,
});

HTMLElement.prototype.getBoundingClientRect = function getBoundingClientRect(): DOMRect {
  return {
    x: 0,
    y: 0,
    top: 0,
    left: 0,
    right: VIEWPORT_WIDTH,
    bottom: VIEWPORT_HEIGHT,
    width: VIEWPORT_WIDTH,
    height: VIEWPORT_HEIGHT,
    toJSON: () => ({}),
  } as DOMRect;
};
