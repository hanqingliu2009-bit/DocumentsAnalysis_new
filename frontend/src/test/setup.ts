// Test setup file for Vitest
import '@testing-library/jest-dom'

// Mock fetch globally
global.fetch = vi.fn()

// Clean up after each test
import { cleanup } from '@testing-library/react'
import { afterEach, vi } from 'vitest'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})
