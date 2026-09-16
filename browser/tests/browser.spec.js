import { test, expect } from "@playwright/test";
import { fileURLToPath } from "node:url";

// Block external requests: the prepared site must work with only static assets.
test.beforeEach(async ({ context }) => {
  await context.route("**/*", route => {
    const url = new URL(route.request().url());
    return url.hostname === "127.0.0.1" || url.protocol === "blob:" ? route.continue() : route.abort();
  });
});

test("create and plot a simple domain", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("status")).toContainText("IGAPolarTorus ·", { timeout: 120000 });
  await page.locator("#domain").selectOption("Cuboid");
  await page.getByRole("button", { name: "Construct domain" }).click();
  await expect(page.getByRole("status")).toContainText("Cuboid ·");
  await expect(page.locator("#diagnostics")).toBeVisible();
  const paintedPixels = await page.locator("#view").evaluate(canvas => {
    const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let painted = 0;
    for (let i = 3; i < pixels.length; i += 4) if (pixels[i] > 0) painted++;
    return painted;
  });
  expect(paintedPixels).toBeGreaterThan(0);
});

test("construct, change parameters, report errors, save and reopen", async ({ page }) => {
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/");
  await expect(page.getByRole("status")).toContainText("IGAPolarTorus ·", { timeout: 120000 });
  await expect(page.locator("#diagnostics")).toBeVisible();
  await page.locator("#domain").selectOption("PoweredEllipticCylinder");
  await page.getByRole("button", { name: "Construct domain" }).click();
  await expect(page.getByRole("status")).toContainText("PoweredEllipticCylinder ·");
  await page.locator("#domain").selectOption("Tokamak");
  await page.locator("#params").fill(JSON.stringify({ num_elements: [4, 12], degree: [2, 3], xi_param: "equal_angle" }));
  await page.getByRole("button", { name: "Construct domain" }).click();
  await expect(page.getByRole("status")).toContainText("Tokamak ·");
  await page.locator("#eqdsk").setInputFiles(fileURLToPath(new URL(
    "../../src/struphy/fields_background/mhd_equil/eqdsk/data/AUGNLED_g031213.00830.high", import.meta.url,
  )));
  await expect(page.locator("#flux-note")).toContainText("AUGNLED");
  await page.getByRole("button", { name: "Construct domain" }).click();
  await expect(page.getByRole("status")).toContainText("Tokamak ·", { timeout: 90000 });
  const before = await page.locator("#coordinates").textContent();
  const downloaded = page.waitForEvent("download");
  await page.getByRole("button", { name: "Save geometry" }).click();
  const download = await downloaded;
  await page.locator("#archive").setInputFiles(await download.path());
  await expect(page.getByRole("status")).toContainText("PoloidalSplineTorus ·");
  await expect(page.locator("#coordinates")).toHaveText(before);
  await page.locator("#params").fill("invalid json");
  await page.getByRole("button", { name: "Construct domain" }).click();
  await expect(page.locator("#status")).toHaveClass("error");
  await expect(page.getByRole("button", { name: "Construct domain" })).toBeEnabled();
  await page.locator("#domain").selectOption("Cuboid");
  await page.getByRole("button", { name: "Construct domain" }).click();
  await expect(page.getByRole("status")).toContainText("Cuboid ·");
  await page.screenshot({ path: "test-results/geometry.png", fullPage: true });
  expect(errors).toEqual([]);
});
