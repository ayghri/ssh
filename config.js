// Edit these URLs after uploading the contents of ./build/ to S3.
// The defaults point at the local ./build/ directory so you can serve
// the site from `python3 -m http.server` and test before deploying.
//
// All asset URLs must allow CORS GET from wherever index.html is hosted.
// For S3: enable CORS on the bucket with AllowedOrigins: ["*"] (or your
// specific origin) and AllowedMethods: ["GET"].

export const CONFIG = {
  // Cache-buster suffix `?v=N` forces browsers to re-fetch when assets
  // are regenerated. Bump on every rebuild of build/ contents.
  MODEL_URL: "./build/e6.onnx?v=3",
  META_URL: "./build/meta.json?v=3",
  STATS_URL: "./build/obs_yearly_stats.bin?v=3",
  KEEP_MASK_URL: "./build/keep_mask.bin?v=3",
  COS_LAT_URL: "./build/cos_lat_per_cell.bin?v=3",

  // ONNX Runtime Web build to load.
  //
  // - `ort.min.mjs` (default below): vanilla wasm; works in any modern
  //   browser. WebGPU is auto-used when `navigator.gpu` is available;
  //   otherwise it falls back to wasm cleanly. Safest pick.
  //
  // - `ort.webgpu.min.mjs`: WebGPU-only build with the `jsep` wasm.
  //   Faster when WebGPU is on, but **aborts** when WebGPU is missing
  //   (the jsep wasm can't init without `navigator.gpu`). Don't use
  //   unless you're certain every visitor's browser has WebGPU.
  // Pinned to v1.20.1 because:
  //   - ESM (`ort.min.mjs`) is required by app.js' dynamic import,
  //     and only exists from v1.19.2 onwards.
  //   - This bundle uses the threaded wasm, which needs
  //     SharedArrayBuffer (i.e. crossOriginIsolated). Serve the site
  //     with ./serve.py (sets COOP/COEP) -- plain
  //     `python3 -m http.server` will NOT work and the wasm aborts.
  ORT_URL: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs",

  // Base URL used by ort to fetch its .wasm/.mjs shards. Must end in `/`.
  ORT_WASM_BASE: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/",
};
