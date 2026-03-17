import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm"],
  dts: true,
  splitting: false,
  sourcemap: true,
  clean: true,
  target: "node20",
  // Bundle the workspace "known" package so the plugin is self-contained
  noExternal: ["known"],
  // No banner — bundled core handles its own createRequire
});
