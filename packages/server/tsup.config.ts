import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm"],
  dts: true,
  splitting: false,
  sourcemap: true,
  clean: true,
  target: "node20",
  banner: {
    js: `#!/usr/bin/env node
import { createRequire } from 'module'; const require = createRequire(import.meta.url);`,
  },
});
