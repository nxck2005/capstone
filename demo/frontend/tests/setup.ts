import '@testing-library/jest-dom/vitest'

// jsdom has no PointerEvent; without this, pointer events in tests lose clientX.
if (typeof window !== 'undefined' && !('PointerEvent' in window)) {
  class PointerEventPolyfill extends MouseEvent {
    pointerId: number
    constructor(type: string, init: PointerEventInit = {}) { super(type, init); this.pointerId = init.pointerId ?? 0 }
  }
  Object.defineProperty(window, 'PointerEvent', { value: PointerEventPolyfill })
}
