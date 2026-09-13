import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  testMatch: "*.spec.js",
  timeout: 180000,
  workers: 1,
  use: { baseURL: "http://127.0.0.1:8765", headless: true },
  webServer: {
    command: "python3 -m http.server 8765 --bind 127.0.0.1 --directory site",
    url: "http://127.0.0.1:8765",
    reuseExistingServer: !process.env.CI,
    stderr: "ignore",
  },
});
