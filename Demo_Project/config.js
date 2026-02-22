/**
 * CortexOS Demo – API configuration.
 * Defaults point to the CortexOS API on the same EC2 deployment.
 * Override in the app UI (stored in localStorage) or set window.CORTEX_API_URL before loading.
 */
window.CORTEX_DEMO_CONFIG = {
  /** Base URL of the CortexOS API (no trailing slash). Same host as this demo on EC2. */
  apiBaseUrl: window.CORTEX_API_URL || 'http://3.87.235.87:8000',
  /** Default user ID for demo (UUID). */
  defaultUserId: '550e8400-e29b-41d4-a716-446655440000',
};
