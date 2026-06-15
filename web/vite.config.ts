import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// `base: "./"` keeps asset URLs relative so the built dashboard works whether
// it is served from a domain root, a sub-path, or opened from disk.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "./",
});
