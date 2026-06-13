import '@testing-library/jest-dom';

// jsdom doesn't implement matchMedia, which the theme code calls at render
// time (prefers-color-scheme). Provide a minimal stub so component tests can
// mount anything that reads the media query.
if (!window.matchMedia) {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
    }),
  });
}
