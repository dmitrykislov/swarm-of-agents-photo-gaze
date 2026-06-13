export default {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.ts?(x)', '**/?(*.)+(spec|test).ts?(x)'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
  setupFilesAfterEnv: ['<rootDir>/jest.setup.ts'],
  // Components import their own CSS and image assets; stub them so jest (which
  // only understands JS/TS) can load the modules under test.
  moduleNameMapper: {
    '\\.(css|less|scss|sass)$': '<rootDir>/jest/styleMock.js',
    '\\.(jpg|jpeg|png|gif|svg|webp|avif)$': '<rootDir>/jest/fileMock.js',
  },
};
