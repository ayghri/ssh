// E6 browser inference -- ONNX Runtime Web + WebGPU.
//
// Flow:
//   1. fetch meta + keep_mask + cos_lat + stats in parallel
//   2. dynamic import of ort.webgpu, create session
//   3. populate dropdowns
//   4. on Run: build (1, N_keep, n_input_channels) -> session.run() ->
//      scale to mm/yr -> demean area-weighted -> render

import { CONFIG } from "./config.js";

const statusEl = document.getElementById("status");
const anchorSel = document.getElementById("anchor");
const horizonSel = document.getElementById("horizon");
const runBtn = document.getElementById("run");
const titleEl = document.getElementById("output-title");
const vmaxEl = document.getElementById("vmax-info");
const mapCanvas = document.getElementById("map");
const cbarCanvas = document.getElementById("colorbar");
const mapCtx = mapCanvas.getContext("2d");
const cbarCtx = cbarCanvas.getContext("2d");
mapCtx.imageSmoothingEnabled = false;
// Optional second pair of canvases for AVISO observed comparison.
const obsTitleEl = document.getElementById("obs-title");
const obsVmaxEl = document.getElementById("vmax-obs-info");
const obsMapCanvas = document.getElementById("map-obs");
const obsCbarCanvas = document.getElementById("colorbar-obs");
const obsMapCtx = obsMapCanvas ? obsMapCanvas.getContext("2d") : null;
const obsCbarCtx = obsCbarCanvas ? obsCbarCanvas.getContext("2d") : null;
if (obsMapCtx) obsMapCtx.imageSmoothingEnabled = false;

// Scale each canvas's internal pixel buffer by min(2, devicePixelRatio)
// while keeping its CSS size unchanged. Combined with the bilinear
// smoothing in renderFieldTo, this approximately doubles the effective
// rendering resolution -- a much closer match to matplotlib pcolormesh
// than the original 1:1 canvas. We cap at 2x to avoid bloating draw
// time on 3x phones without much visible payoff.
function scaleCanvasForHiDPI(canvas, ctx) {
  if (!canvas || !ctx) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (dpr === 1) return;
  const cssW = canvas.width;
  const cssH = canvas.height;
  canvas.style.width = cssW + "px";
  canvas.style.height = cssH + "px";
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  // Drawing coords stay in CSS space; the upscale happens automatically
  // through the transform set on the context.
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
scaleCanvasForHiDPI(mapCanvas, mapCtx);
scaleCanvasForHiDPI(cbarCanvas, cbarCtx);
scaleCanvasForHiDPI(obsMapCanvas, obsMapCtx);
scaleCanvasForHiDPI(obsCbarCanvas, obsCbarCtx);

// ---- State ----
let META = null;
let KEEP = null;           // Uint8Array(64800)
let KEEP_IDX = null;       // Int32Array(N_keep) -> flat 0..64799 index in grid
let COS_LAT = null;        // Float32Array(N_keep)
let STATS = null;          // Float16 raw bytes wrapped in Uint16Array;
                            // shape (n_years, 4, 5, N_keep) C-order.
let STATS_F32_LAST_SLAB = null; // optional cache: not strictly needed
let OBS_TRENDS = null;     // Uint16Array(n_rows * n_keep), fp16 packed
let OBS_TRENDS_SHAPE = null;
let OBS_TRENDS_INDEX = null; // Map "anchor|horizon" -> row index
let SESSION = null;
let ORT = null;

const setStatus = (msg, isError = false) => {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", isError);
};

// ---- float16 -> float32 ----
// IEEE 754 half-precision decoder. JS lacks a built-in.
function f16toF32(u16) {
  const s = (u16 & 0x8000) >> 15;
  const e = (u16 & 0x7C00) >> 10;
  const f = u16 & 0x03FF;
  if (e === 0) {
    // subnormal or zero
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  } else if (e === 0x1F) {
    return f ? NaN : ((s ? -1 : 1) * Infinity);
  }
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function f16ArrayToF32(u16arr) {
  const out = new Float32Array(u16arr.length);
  for (let i = 0; i < u16arr.length; i++) out[i] = f16toF32(u16arr[i]);
  return out;
}

// ---- RdBu_r colormap, 256 entries (RGB 0..255) ----
// Matplotlib RdBu_r endpoints sampled at 9 stops, linearly interpolated to 256.
const RDBU_R_STOPS = [
  [0.019608, 0.188235, 0.380392],  // 0.0
  [0.129412, 0.400000, 0.674510],  // 0.125
  [0.262745, 0.576471, 0.764706],  // 0.25
  [0.572549, 0.772549, 0.870588],  // 0.375
  [0.819608, 0.898039, 0.941176],  // 0.5  -- center
  [0.992157, 0.858824, 0.780392],  // 0.625
  [0.956863, 0.647059, 0.509804],  // 0.75
  [0.839216, 0.376471, 0.301961],  // 0.875
  [0.403922, 0.000000, 0.121569],  // 1.0
].map(([r, g, b]) => [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]);

function buildLut(n = 256) {
  const lut = new Uint8ClampedArray(n * 3);
  const ns = RDBU_R_STOPS.length;
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);                       // 0..1
    const f = t * (ns - 1);
    const i0 = Math.floor(f);
    const i1 = Math.min(i0 + 1, ns - 1);
    const tt = f - i0;
    const a = RDBU_R_STOPS[i0];
    const b = RDBU_R_STOPS[i1];
    lut[i * 3 + 0] = Math.round(a[0] + (b[0] - a[0]) * tt);
    lut[i * 3 + 1] = Math.round(a[1] + (b[1] - a[1]) * tt);
    lut[i * 3 + 2] = Math.round(a[2] + (b[2] - a[2]) * tt);
  }
  return lut;
}

const LUT = buildLut(256);
const LAND_RGB = [217, 217, 217];   // light grey

// ---- Fetch with progress ----
async function fetchProgress(url, label) {
  const res = await fetch(url, { cache: "default" });
  if (!res.ok) throw new Error(`fetch ${url} -> HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length") || 0);
  if (!total || !res.body || !res.body.getReader) {
    return new Uint8Array(await res.arrayBuffer());
  }
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    setStatus(`loading ${label}… ${(received / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`);
  }
  const out = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

// ---- Boot ----
async function boot() {
  try {
    setStatus("loading metadata…");
    const metaBytes = await fetchProgress(CONFIG.META_URL, "meta.json");
    META = JSON.parse(new TextDecoder().decode(metaBytes));

    setStatus("loading grid masks…");
    const [keepBytes, cosLatBytes] = await Promise.all([
      fetchProgress(CONFIG.KEEP_MASK_URL, "keep_mask.bin"),
      fetchProgress(CONFIG.COS_LAT_URL, "cos_lat_per_cell.bin"),
    ]);
    KEEP = keepBytes;
    if (KEEP.length !== META.n_uniform) {
      throw new Error(`keep_mask size ${KEEP.length} != n_uniform ${META.n_uniform}`);
    }
    COS_LAT = new Float32Array(cosLatBytes.buffer, cosLatBytes.byteOffset, META.n_keep);

    // Precompute KEEP_IDX: for each of N_keep ocean cells, the flat grid
    // index 0..64799. Used to scatter predictions back to the 180x360 grid.
    KEEP_IDX = new Int32Array(META.n_keep);
    let kp = 0;
    for (let i = 0; i < KEEP.length; i++) {
      if (KEEP[i]) { KEEP_IDX[kp++] = i; }
    }
    if (kp !== META.n_keep) {
      throw new Error(`keep mask has ${kp} ocean cells, expected ${META.n_keep}`);
    }

    setStatus("loading obs feature cache…");
    const statsBytes = await fetchProgress(CONFIG.STATS_URL, "obs_yearly_stats.bin");
    const expectedBytes = META.years.length * 4 * META.var_order.length * META.n_keep * 2;
    if (statsBytes.length !== expectedBytes) {
      throw new Error(`stats size ${statsBytes.length} != expected ${expectedBytes}`);
    }
    STATS = new Uint16Array(statsBytes.buffer, statsBytes.byteOffset,
                              statsBytes.length / 2);

    // Optional obs-trends bundle (AVISO observed trend per anchor/horizon).
    // If the URL is configured AND meta has the layout block, fetch and
    // index. Otherwise the obs panel just stays hidden.
    if (CONFIG.OBS_TRENDS_URL && META.obs_trends_layout) {
      setStatus("loading AVISO observed trends…");
      const obsBytes = await fetchProgress(CONFIG.OBS_TRENDS_URL,
                                            "obs_trends.bin");
      const shape = META.obs_trends_layout.shape;          // [n_rows, n_keep]
      const expected = shape[0] * shape[1] * 2;
      if (obsBytes.length !== expected) {
        throw new Error(`obs_trends size ${obsBytes.length} != ${expected}`);
      }
      OBS_TRENDS = new Uint16Array(obsBytes.buffer, obsBytes.byteOffset,
                                     obsBytes.length / 2);
      OBS_TRENDS_SHAPE = shape;
      OBS_TRENDS_INDEX = new Map();
      META.obs_trends_layout.rows.forEach((r, i) => {
        OBS_TRENDS_INDEX.set(`${r.anchor}|${r.horizon}`, i);
      });
    }

    setStatus("loading ONNX runtime…");
    ORT = await import(CONFIG.ORT_URL);
    ORT.env.wasm.wasmPaths = CONFIG.ORT_WASM_BASE;
    // crossOriginIsolated lets us use multiple wasm worker threads.
    ORT.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
    ORT.env.wasm.proxy = false;

    setStatus("loading model… (this can take ~10 MB)");
    const modelBytes = await fetchProgress(CONFIG.MODEL_URL, "e6.onnx");
    SESSION = await ORT.InferenceSession.create(modelBytes, {
      executionProviders: ["webgpu", "wasm"],
      // IMPORTANT: ORT-Web's basic graph optimiser tries to constant-
      // fold the 6 attention-layer gathers on `neighbors` at load time.
      // With N=43335 / K=8 / hidden=192 that explodes to ~1.6 GB and
      // aborts the wasm process. Disable optimisation entirely; the
      // runtime is still fast enough and the ops are already minimal
      // (we wrapped the export to skip boolean indexing).
      graphOptimizationLevel: "disabled",
    });
    const epUsed = (SESSION.handler && SESSION.handler.session
                    && SESSION.handler.session.executionProviders)
                   || SESSION.executionProviders || "webgpu/wasm";
    setStatus(`ready. EP: ${typeof epUsed === "string" ? epUsed : "webgpu/wasm"}.`);

    populateAnchorSelect();
    anchorSel.disabled = false;
    horizonSel.disabled = false;
    runBtn.disabled = false;
  } catch (e) {
    console.error(e);
    setStatus(`error during init: ${e.message}`, true);
  }
}

function populateAnchorSelect() {
  anchorSel.innerHTML = "";
  for (const y of META.available_anchors) {
    const opt = document.createElement("option");
    opt.value = String(y);
    opt.textContent = String(y);
    anchorSel.appendChild(opt);
  }
  // Default to the most recent anchor with all horizons fully observed.
  const last = META.available_anchors[META.available_anchors.length - 1];
  anchorSel.value = String(last - 6);
  populateHorizonSelect();
  anchorSel.addEventListener("change", populateHorizonSelect);
}

function populateHorizonSelect() {
  const a = parseInt(anchorSel.value, 10);
  const obsEnd = META.obs_last_full_year;
  // Preserve the user's current horizon choice across anchor changes
  // (the options' VALUES stay the same; only the date labels change).
  const prev = horizonSel.value;
  horizonSel.innerHTML = "";
  const wanted = [5, 10, 15, 20];
  for (const h of wanted) {
    if (!META.horizons_yr.includes(h)) continue;
    const opt = document.createElement("option");
    opt.value = String(h);
    const endY = a + h;
    opt.textContent = endY > obsEnd
      ? `${h} yr  (ends ${endY}, past obs)`
      : `${h} yr  (${a}–${endY})`;
    horizonSel.appendChild(opt);
  }
  // Re-select the previous horizon if it is still in the list;
  // fall back to the default selection (first option) otherwise.
  if (prev && Array.from(horizonSel.options).some(o => o.value === prev)) {
    horizonSel.value = prev;
  }
}

// Build the (1, N_keep, n_input_channels) Float32Array for one anchor.
function buildFeatures(anchorYear) {
  const nKeep = META.n_keep;
  const nStats = 4, nVars = META.var_order.length, nYears = META.n_years_input;
  const nFeat = nStats * nVars * nYears;            // 200
  const nIn = nFeat + 1;                             // 201
  const yearsAxis = META.years;
  // Past years: [anchor - nYears, anchor - 1]
  const y0 = anchorYear - nYears - yearsAxis[0];
  if (y0 < 0 || y0 + nYears > yearsAxis.length) {
    throw new Error(`anchor ${anchorYear} out of range for stats years ` +
                     `[${yearsAxis[0]}..${yearsAxis[yearsAxis.length - 1]}]`);
  }

  // STATS layout on disk: (year, stat, var, cell)  C-order, fp16.
  // The training dataset (scripts/38_forced_inference_obs.py
  // `_yearly_feat_at`) flattens per-cell as
  //   [cur(V), mean(V), slope(V), std(V)]  per year, then concat years.
  // So the model's input channel index is:
  //   c = ((year * nStats) + stat) * nVars + var
  // i.e. YEAR-major, STAT-mid, VAR-fastest.  Using any other order
  // permutes the input channels relative to the trained linear-embed
  // weights, which yields output with the right magnitude statistics
  // but the wrong spatial pattern.
  const out = new Float32Array(nKeep * nIn);

  const yearStride = nStats * nVars * nKeep;           // floats per year on disk
  const statStride = nVars * nKeep;
  const varStride = nKeep;

  for (let yp = 0; yp < nYears; yp++) {
    for (let s = 0; s < nStats; s++) {
      for (let v = 0; v < nVars; v++) {
        const cBase = ((yp * nStats + s) * nVars + v);   // 0..199
        const srcOff = (y0 + yp) * yearStride + s * statStride + v * varStride;
        for (let k = 0; k < nKeep; k++) {
          out[k * nIn + cBase] = f16toF32(STATS[srcOff + k]);
        }
      }
    }
  }
  // Year scalar channel.
  const yScalar = (anchorYear - META.year_scalar_offset) / META.year_scalar_divisor;
  for (let k = 0; k < nKeep; k++) {
    out[k * nIn + nFeat] = yScalar;
  }
  return out;
}

// Render a (N_keep,) field as a 180x360 PlateCarree map onto the given
// canvas. Pass null for `titleElement` to skip title update.
// Shift the map so the left/right seam runs through Africa instead of
// cutting the Atlantic in half. 30 = central meridian moves to 210 deg E,
// matching the matplotlib PNGs (Robinson, central_longitude=210).
const LON_SHIFT_DEG = 30;

function renderFieldTo(field, vmax, title, mapCanvasEl, mapCtxEl,
                       cbarCanvasEl, cbarCtxEl, titleElement) {
  const nLat = META.lat_size, nLon = META.lon_size;
  const small = document.createElement("canvas");
  small.width = nLon; small.height = nLat;
  const sctx = small.getContext("2d");
  const img = sctx.createImageData(nLon, nLat);
  const data = img.data;

  // Initialise to land grey.
  for (let i = 0; i < nLat * nLon; i++) {
    data[i * 4 + 0] = LAND_RGB[0];
    data[i * 4 + 1] = LAND_RGB[1];
    data[i * 4 + 2] = LAND_RGB[2];
    data[i * 4 + 3] = 255;
  }

  // Ocean cells. Apply the longitude shift so the left edge is at
  // LON_SHIFT_DEG instead of 0 (puts Africa at the seam, not the Atlantic).
  const invVmax = 1.0 / Math.max(vmax, 1e-12);
  for (let k = 0; k < META.n_keep; k++) {
    const flat = KEEP_IDX[k];
    const lat = Math.floor(flat / nLon);
    const lon = flat % nLon;
    const lonShifted = (lon - LON_SHIFT_DEG + nLon) % nLon;
    const row = (nLat - 1) - lat;
    const px = (row * nLon + lonShifted) * 4;
    let t = field[k] * invVmax;
    if (t < -1) t = -1; else if (t > 1) t = 1;
    const li = Math.round((t + 1) * 0.5 * 255);
    data[px + 0] = LUT[li * 3 + 0];
    data[px + 1] = LUT[li * 3 + 1];
    data[px + 2] = LUT[li * 3 + 2];
  }
  sctx.putImageData(img, 0, 0);

  // Draw in CSS-pixel coords (clientWidth/Height), because the canvas
  // has been HiDPI-scaled at boot via setTransform(dpr, ...).
  const W = mapCanvasEl.clientWidth || mapCanvasEl.width;
  const H = mapCanvasEl.clientHeight || mapCanvasEl.height;
  mapCtxEl.clearRect(0, 0, W, H);
  // Smoothing on for a pcolormesh-style bilinear upscale. Combined with
  // the HiDPI canvas scale-up, this gives a much closer match to
  // matplotlib's render than the blocky nearest-neighbour pass.
  mapCtxEl.imageSmoothingEnabled = true;
  mapCtxEl.imageSmoothingQuality = "high";
  mapCtxEl.drawImage(small, 0, 0, W, H);
  if (titleElement) titleElement.textContent = title;
  drawColorbarTo(vmax, cbarCanvasEl, cbarCtxEl);
}

// Thin wrapper for the prediction canvas (existing call sites).
function renderField(field, vmax, title) {
  renderFieldTo(field, vmax, title,
                 mapCanvas, mapCtx, cbarCanvas, cbarCtx, titleEl);
}

function drawColorbarTo(vmax, cbarCanvasEl, cbarCtxEl) {
  // CSS-pixel coords (canvas is HiDPI-scaled via setTransform).
  const W = cbarCanvasEl.clientWidth || cbarCanvasEl.width;
  const H = cbarCanvasEl.clientHeight || cbarCanvasEl.height;
  cbarCtxEl.clearRect(0, 0, W, H);
  const barH = 16, pad = 12;
  const x0 = pad, x1 = W - pad;
  const y0 = 4, y1 = y0 + barH;
  for (let x = x0; x < x1; x++) {
    const t = (x - x0) / (x1 - x0);
    const li = Math.round(t * 255);
    cbarCtxEl.fillStyle = `rgb(${LUT[li * 3]}, ${LUT[li * 3 + 1]}, ${LUT[li * 3 + 2]})`;
    cbarCtxEl.fillRect(x, y0, 1, barH);
  }
  cbarCtxEl.strokeStyle = "#888";
  cbarCtxEl.lineWidth = 1;
  cbarCtxEl.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, barH - 1);
  cbarCtxEl.fillStyle = "#333";
  cbarCtxEl.font = "12px ui-monospace, Menlo, monospace";
  cbarCtxEl.textBaseline = "top";
  const lbl = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
  cbarCtxEl.textAlign = "left";
  cbarCtxEl.fillText(lbl(-vmax), x0, y1 + 4);
  cbarCtxEl.textAlign = "center";
  cbarCtxEl.fillText("0", (x0 + x1) / 2, y1 + 4);
  cbarCtxEl.textAlign = "right";
  cbarCtxEl.fillText(lbl(+vmax), x1, y1 + 4);
  cbarCtxEl.textAlign = "center";
  cbarCtxEl.fillText("SSH trend (mm/yr)", W / 2, y1 + 20);
}
function drawColorbar(vmax) {
  drawColorbarTo(vmax, cbarCanvas, cbarCtx);
}

// Show or hide the spinner shown next to the status text while a long
// inference call is in flight.
function showSpinner(msg) {
  statusEl.classList.add("running");
  statusEl.innerHTML =
    `<span class="spinner" aria-hidden="true"></span>` +
    `<span class="spinner-msg">${msg}</span>`;
}
function hideSpinner() {
  statusEl.classList.remove("running");
}

async function runOnce() {
  if (!SESSION) return;
  runBtn.disabled = true;
  showSpinner("running the model…");
  // Yield to the browser so the spinner actually paints before we start
  // the (synchronous) wasm inference call that pegs the main thread.
  await new Promise(r => setTimeout(r, 0));
  const anchor = parseInt(anchorSel.value, 10);
  const horizon = parseInt(horizonSel.value, 10);
  const hIdx = META.horizons_yr.indexOf(horizon);
  if (hIdx < 0) {
    hideSpinner();
    setStatus(`unknown horizon ${horizon}`, true);
    runBtn.disabled = false;
    return;
  }

  try {
    const t0 = performance.now();
    showSpinner(`building features for anchor ${anchor}…`);
    const feats = buildFeatures(anchor);
    const nKeep = META.n_keep;
    const nIn = META.n_input_channels;

    const inputTensor = new ORT.Tensor("float32", feats, [1, nKeep, nIn]);
    showSpinner(`running the model (this can take a few seconds)…`);
    // Another yield so the spinner repaints with the new message.
    await new Promise(r => setTimeout(r, 0));
    const t1 = performance.now();
    const inputName = SESSION.inputNames[0];
    const outputName = SESSION.outputNames[0];
    const out = await SESSION.run({ [inputName]: inputTensor });
    const t2 = performance.now();
    const pred = out[outputName].data;       // Float32Array length nKeep * nHorizons
    const nH = META.n_horizons;

    // Extract selected horizon, scale to mm/yr.
    const scale = META.output_mm_per_yr_scale;
    const field = new Float32Array(nKeep);
    for (let k = 0; k < nKeep; k++) {
      field[k] = pred[k * nH + hIdx] * scale;
    }

    // Area-weighted demean.
    let wsum = 0, ws = 0;
    for (let k = 0; k < nKeep; k++) {
      wsum += COS_LAT[k] * field[k];
      ws += COS_LAT[k];
    }
    const mean = wsum / ws;
    let var2 = 0;
    for (let k = 0; k < nKeep; k++) {
      field[k] -= mean;
      var2 += COS_LAT[k] * field[k] * field[k];
    }
    const std = Math.sqrt(var2 / ws);
    const vmax = Math.max(1.5 * std, 0.05);
    // Debug: expected std for anchor=2005 h=20yr is ~0.4 mm/yr.
    // If it is ~1.4 you are reading a stale obs_yearly_stats.bin from cache.
    console.log(`[debug] horizon ${META.horizons_yr[hIdx]}yr  area-w std=${std.toFixed(3)} mm/yr`,
      `pred sample first5:`, Array.from(field.slice(0, 5)).map(x => x.toFixed(3)));

    renderField(field, vmax,
      `Predicted SSH trend  ${anchor}–${anchor + horizon}  (${horizon}-yr horizon)`);
    vmaxEl.textContent =
      `vmax: ±${vmax.toFixed(2)} mm/yr   (area-weighted std × 1.5; std=${std.toFixed(2)})   ` +
      `infer ${(t2 - t1).toFixed(0)} ms, total ${(t2 - t0).toFixed(0)} ms`;

    // ---- AVISO observed trend for the same (anchor, horizon) ----
    renderObsTrend(anchor, horizon, field, std);

    hideSpinner();
    setStatus(`done in ${(t2 - t0).toFixed(0)} ms (inference ${(t2 - t1).toFixed(0)} ms)`);
  } catch (e) {
    console.error(e);
    hideSpinner();
    setStatus(`error: ${e.message}`, true);
  } finally {
    runBtn.disabled = false;
  }
}

// Render the AVISO observed trend for (anchor, horizon) into the obs
// canvas, and also compute pattern correlation with the model prediction
// for direct comparison. Hides the obs panel if no precomputed combo or
// no DOM elements present.
function renderObsTrend(anchor, horizon, predField, predStd) {
  if (!obsMapCanvas || !OBS_TRENDS || !OBS_TRENDS_INDEX) return;
  const key = `${anchor}|${horizon}`;
  const row = OBS_TRENDS_INDEX.get(key);
  if (row === undefined) {
    // Out of range -- hide the obs panel.
    obsTitleEl.textContent = "AVISO observed: not available for this window";
    obsMapCtx.clearRect(0, 0, obsMapCanvas.width, obsMapCanvas.height);
    obsCbarCtx.clearRect(0, 0, obsCbarCanvas.width, obsCbarCanvas.height);
    obsVmaxEl.textContent =
      `anchor + horizon ${anchor + horizon} exceeds the AVISO record ` +
      `(precomputed combos: ${OBS_TRENDS_INDEX.size}).`;
    return;
  }
  const nKeep = META.n_keep;
  const off = row * nKeep;
  const obsField = new Float32Array(nKeep);
  for (let k = 0; k < nKeep; k++) {
    obsField[k] = f16toF32(OBS_TRENDS[off + k]);
  }
  // The precompute already demeans area-weighted; recompute std for vmax.
  let ws = 0, var2 = 0;
  for (let k = 0; k < nKeep; k++) {
    ws += COS_LAT[k];
    var2 += COS_LAT[k] * obsField[k] * obsField[k];
  }
  const obsStd = Math.sqrt(var2 / ws);
  const obsVmax = Math.max(1.5 * obsStd, 0.05);

  // Pattern correlation against the model prediction (area-weighted).
  let num = 0, denP = 0, denO = 0;
  for (let k = 0; k < nKeep; k++) {
    const w = COS_LAT[k];
    num  += w * predField[k] * obsField[k];
    denP += w * predField[k] * predField[k];
    denO += w * obsField[k]  * obsField[k];
  }
  const corr = num / Math.sqrt(Math.max(denP, 1e-30) * Math.max(denO, 1e-30));

  renderFieldTo(obsField, obsVmax,
    `AVISO observed trend  ${anchor}–${anchor + horizon}  (${horizon}-yr)`,
    obsMapCanvas, obsMapCtx, obsCbarCanvas, obsCbarCtx, obsTitleEl);

  obsVmaxEl.textContent =
    `vmax: ±${obsVmax.toFixed(2)} mm/yr   std=${obsStd.toFixed(2)}   ` +
    `pattern corr (model vs AVISO) = ${corr.toFixed(3)}`;
}

runBtn.addEventListener("click", runOnce);

boot();
