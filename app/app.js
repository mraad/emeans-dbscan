import { WATER, contingency, scores } from "./metrics.js";

// WorldCover 2021 colours and names (ESA legend).
const CLASSES = {
  10: ["Tree cover", "#006400"], 20: ["Shrubland", "#ffbb22"], 30: ["Grassland", "#ffff4c"], 40: ["Cropland", "#f096ff"],
  50: ["Built-up", "#fa0000"], 60: ["Bare", "#b4b4b4"], 70: ["Snow and ice", "#f0f0f0"], 80: ["Water", "#0064c8"],
  90: ["Wetland", "#0096a0"], 95: ["Mangroves", "#00cf75"], 100: ["Moss and lichen", "#fae6a0"],
};
const SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
// By cluster id, so that a cluster keeps its colour from epoch to epoch.
// More than 8 clusters: hues repeat and only the list, the pixel panel and the highlight identify a cluster.
const clusterColour = (id) =>
  id < SLOTS.length ? SLOTS[id] : `oklch(${[0.62, 0.75, 0.5][id % 3]} 0.15 ${(id * 137.5) % 360})`;

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};
const fmt = (v, digits = 3) => v.toFixed(digits);
const swatch = (css) => el("span", { className: "swatch", style: `background:${css}` });
const className = (c) => CLASSES[c]?.[0] ?? "Empty";           // an empty cluster has no class
const pct = (v) => `${(100 * v).toFixed(1)}%`;

const MAP_LIBRARY_ERROR = "The map library (js.arcgis.com) could not load. Check the internet connection and reload the page.";
if (!globalThis.$arcgis) {
  showError(MAP_LIBRARY_ERROR);
  throw new Error("no map library");
}
const [[MediaLayer, ImageElement, ExtentAndRotationGeoreference, Extent], meta] = await Promise.all([
  $arcgis.import([
    "@arcgis/core/layers/MediaLayer.js",
    "@arcgis/core/layers/support/ImageElement.js",
    "@arcgis/core/layers/support/ExtentAndRotationGeoreference.js",
    "@arcgis/core/geometry/Extent.js",
  ]).catch((e) => {
    showError(MAP_LIBRARY_ERROR);
    throw e;
  }),
  fetch("data/meta.json").then((r) => r.json()).catch(() => {
    showError("data/meta.json is missing. Run: uv run --python 3.12 app/prepare_data.py");
    throw new Error("no data");
  }),
]);
const { size } = meta;
const worker = new Worker("worker.js", { type: "module" });

const state = {
  phase: "idle", window: "test", wc: null, labels: null, clusters: [], run: null, curve: [], history: [],
  colour: "cluster", picked: null, clicked: null, smallK: true,
};

// ---------- map ----------
const mapEl = $("map");
const wcLayer = new MediaLayer({ title: "WorldCover", source: [], opacity: 0.85, visible: false });
const clusterLayer = new MediaLayer({ title: "Clusters", source: [], opacity: 0.85 });
const viewReady = mapEl.viewOnReady().then(() => {            // the data loads without waiting for the basemap
  mapEl.constraints.snapToZoom = false;
  mapEl.map.addMany([wcLayer, clusterLayer]);
});
viewReady.catch(() => showError("The map could not start. e-means still runs, without a map."));
// A CSS colour as one little-endian RGBA pixel.
const pack = (css, alpha = 255) => {
  const c = pack.ctx ??= el("canvas", { width: 1, height: 1 }).getContext("2d", { willReadFrequently: true });
  c.clearRect(0, 0, 1, 1);
  c.fillStyle = css;
  c.fillRect(0, 0, 1, 1);
  const [r, g, b] = c.getImageData(0, 0, 1, 1).data;
  return ((alpha << 24) | (b << 16) | (g << 8) | r) >>> 0;
};

const extent = () => {
  const [xmin, ymin, xmax, ymax] = meta.windows[state.window].extent;
  return new Extent({ xmin, ymin, xmax, ymax, spatialReference: { wkid: 102100 } });
};

// Draws the pixel grid to a fresh canvas. values[i] & 255 is the index in the colour table; -1 (no data) is entry 255, transparent.
function setImage(layer, values, lut) {
  const canvas = el("canvas", { width: size, height: size });
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(size, size);
  const px = new Uint32Array(img.data.buffer);
  for (let i = 0; i < px.length; i++) px[i] = lut[values[i] & 255];
  ctx.putImageData(img, 0, 0);
  const old = layer.source.elements.toArray();               // add first, then remove: no empty frame between
  layer.source.elements.add(new ImageElement({ image: canvas, georeference: new ExtentAndRotationGeoreference({ extent: extent() }) }));
  layer.source.elements.removeMany(old);
}

function drawWorldCover() {
  const lut = new Uint32Array(256);
  for (const [c, [, css]] of Object.entries(CLASSES)) lut[c] = pack(css);
  setImage(wcLayer, state.wc, lut);
}

function drawClusters() {
  const lut = new Uint32Array(256);
  for (const c of state.clusters) {
    if (!c.n) continue;
    const css = state.colour === "majority" ? CLASSES[c.majority][1] : clusterColour(c.id);
    lut[c.id] = pack(css, state.picked === null || state.picked === c.id ? 255 : 40);
  }
  setImage(clusterLayer, state.labels, lut);
}

const swipe = document.createElement("arcgis-swipe");
swipe.autoDestroyDisabled = true;
function setCompare(mode) {
  wcLayer.visible = mode !== "off";
  clusterLayer.visible = mode !== "wc";
  if (mode === "swipe" && !swipe.isConnected) {
    mapEl.append(swipe);
    swipe.startLayers = [clusterLayer];
    swipe.endLayers = [wcLayer];
  } else if (mode !== "swipe" && swipe.isConnected) {
    swipe.startLayers = [];
    swipe.endLayers = [];
    swipe.remove();
  }
}

// ---------- clusters and scores ----------
function describe(labels, k) {
  const all = contingency(state.wc, labels, k), land = new Map(all);
  land.delete(WATER);
  const list = Array.from({ length: k }, (_, j) => {
    const counts = [...all].map(([c, row]) => [c, row[j]]).filter((e) => e[1] > 0).sort((a, b) => b[1] - a[1]);
    const n = counts.reduce((s, e) => s + e[1], 0);
    return { id: j, n, counts, majority: counts[0]?.[0], purity: n ? counts[0][1] / n : 0 };
  });
  return { list, land: scores(land, k) };
}

function apply(labels, k) {
  const { list, land } = describe(labels, k);
  state.labels = labels;
  state.clusters = list;
  drawClusters();
  renderClusters();
  return land;
}

// A table row that works as a button.
const pickRow = (title, act, cells, props = {}) => el("tr", {
  className: "pick", tabIndex: 0, role: "button", title, onclick: act,
  onkeydown: (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), act()), ...props,
}, ...cells);
const table = (heads, rows, empty) => rows.length
  ? el("table", {}, el("thead", {}, el("tr", {}, ...heads)), el("tbody", {}, ...rows))
  : el("p", { className: "hint" }, empty);

function renderClusters() {
  const total = state.clusters.reduce((s, c) => s + c.n, 0);
  const rows = [...state.clusters].sort((a, b) => b.n - a.n).map((c) => {
    const pick = () => {
      state.picked = state.picked === c.id ? null : c.id;
      drawClusters();
      renderClusters();
    };
    return pickRow("Select to highlight this cluster on the map", pick, [
      el("td", {}, swatch(clusterColour(c.id)), `${c.id + 1}`),
      el("td", { className: "num" }, pct(c.n / total)),
      el("td", {}, className(c.majority)),
      el("td", { className: "num" }, pct(c.purity)),
      el("td", {}, bar(c)),
    ], { ariaPressed: String(state.picked === c.id) });
  });
  const heads = ["Cluster", "Share", "Main class", "Purity", "WorldCover mix"].map((h, i) =>
    el("th", { className: i === 1 || i === 3 ? "num" : "" }, h));
  $("clusters").replaceChildren(table(heads, rows, "Run e-means to make clusters."));
}

function renderLegend() {
  const present = new Set(state.wc);
  $("classes").replaceChildren(...Object.entries(CLASSES).filter(([c]) => present.has(+c)).map(([, [name, css]]) =>
    el("span", {}, swatch(css), name)));
}

function bar(c) {
  const b = el("div", { className: "bar" });
  c.counts.forEach(([cls, n]) => {
    if (n / c.n < 0.005) return;
    b.append(el("span", { title: `${CLASSES[cls][0]}: ${pct(n / c.n)}`, style: `flex:${n / c.n} 0 0;background:${CLASSES[cls][1]}` }));
  });
  return b;
}

function renderResult() {
  const r = state.run;
  const tile = (label, value, note = "") => el("div", { className: "tile" }, el("div", { className: "label" }, label), el("div", { className: "value" }, value), el("div", { className: "note" }, note));
  $("tiles").replaceChildren(...(r ? [
    tile("Land ARI", fmt(r.ari), "against WorldCover, water excluded"),
    tile("Land purity", fmt(r.purity), "grows with k"),
    tile("Mean cosine", fmt(r.meanCosine, 4), "objective, higher is better"),
    tile("Training time", `${(r.trainMs / 1000).toFixed(2)} s`, `${r.trainN.toLocaleString("en")} px, batch ${r.batch.toLocaleString("en")}, one browser thread`),
  ] : [el("p", { className: "hint" }, "No result yet.")]));
  renderChart();
  const rows = state.history.map((h) => pickRow("Load these settings", () => load(h), [
    ...[h.window, h.k, h.nepochs, h.seed, `${(h.trainN / 1000).toFixed(0)}k`, fmt(h.meanCosine, 4), fmt(h.ari)]
      .map((v) => el("td", { className: "num" }, String(v))),
  ]));
  const heads = ["Window", "k", "Ep.", "Seed", "Train", "Cosine", "Land ARI"].map((h) => el("th", { className: "num" }, h));
  $("runs").replaceChildren(table(heads, rows, "Each run is listed here. Run several seeds with k = 3 and compare the cosine and the ARI."));
}

// ---------- objective chart: mean cosine per epoch, one series ----------
function renderChart() {
  const c = state.curve, box = $("chart");
  if (c.length < 2) return box.replaceChildren();
  const W = 340, H = 150, L = 46, R = 14, T = 26, B = 24;
  const ys = c.map((p) => p.y), lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo || 0.01) * 0.15;
  const X = (i) => L + ((W - L - R) * i) / (c.length - 1), Y = (v) => H - B - ((H - T - B) * (v - lo + pad)) / (hi - lo + 2 * pad);
  const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `Mean cosine per epoch: ${c.map((p) => `epoch ${p.epoch} ${fmt(p.y, 4)}`).join(", ")}`);
  const add = (tag, attrs, text) => {
    const e = document.createElementNS(ns, tag);
    for (const [a, v] of Object.entries(attrs)) e.setAttribute(a, v);
    if (text !== undefined) e.textContent = text;
    svg.append(e);
    return e;
  };
  add("text", { x: 0, y: 12, class: "title" }, "Mean cosine during training, per epoch");
  for (const v of [lo, hi]) {
    add("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: "var(--grid)" });
    add("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end" }, fmt(v, 4));
  }
  add("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: "var(--axis)" });
  c.forEach((p, i) => add("text", { x: X(i), y: H - 8, "text-anchor": "middle" }, p.epoch));
  add("polyline", { points: c.map((p, i) => `${X(i)},${Y(p.y)}`).join(" "), fill: "none", stroke: "var(--series-1)", "stroke-width": 2 });
  const tip = add("text", { x: W - R, y: 12, "text-anchor": "end" }, "");
  c.forEach((p, i) => {
    add("circle", { cx: X(i), cy: Y(p.y), r: 4, fill: "var(--series-1)", stroke: "var(--surface-1)", "stroke-width": 2 });
    const hit = add("rect", { x: X(i) - 16, y: T, width: 32, height: H - T, fill: "transparent" });
    hit.addEventListener("mouseenter", () => (tip.textContent = `epoch ${p.epoch}: ${fmt(p.y, 5)}`));
    hit.addEventListener("mouseleave", () => (tip.textContent = ""));
  });
  box.replaceChildren(svg);
}

// ---------- controls ----------
function showError(message) {
  const n = $("error");
  n.querySelector("[slot=message]").textContent = message;
  n.open = true;
}
function setPhase(phase, note = "") {                        // "idle", "loading" or "running"
  state.phase = phase;
  $("run").disabled = phase !== "idle" || !state.wc;
  $("stop").disabled = phase !== "running";
  $("window").disabled = phase !== "idle";
  $("progress").hidden = phase !== "running";
  $("status").textContent = note;
}
function clearResult() {                                     // old scores must not stay beside a new map
  state.run = state.picked = null;
  state.curve = [];
  renderResult();
}

function settings() {
  return {
    k: $("k").value, nepochs: $("epochs").value, seed: Number($("seed").value),
    trainN: Number($("trainN").value), live: $("live").checked,
  };
}
function startRun() {
  if (state.phase !== "idle" || !state.wc) return;
  const { seed } = settings();
  if (!$("seed").value || !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff)
    return showError("Enter a seed: a whole number from 0 to 4294967295.");
  clearResult();
  $("progress").value = 0;
  $("error").open = false;
  setPhase("running", "Training");
  worker.postMessage({ type: "run", ...settings() });
}
function selectWindow(w) {
  state.window = w;
  state.wc = state.labels = state.clicked = null;
  state.clusters = [];
  $("pixel").hidden = true;
  clusterLayer.source.elements.removeAll();
  wcLayer.source.elements.removeAll();
  renderClusters();
  clearResult();
  setPhase("loading", "Loading the embeddings");
  viewReady.then(() => mapEl.goTo(extent().expand(1.05), { animate: false }).then(
    () => ($("home").viewpoint = mapEl.viewpoint.clone()),
    () => {},                                              // rejected only when a newer goTo replaces it
  ), () => {});                                            // reported where viewReady is made
  worker.postMessage({ type: "load", window: w, d: meta.d });
}
// Puts the settings of an earlier run back, so that it can be run again.
function load(h) {
  if (state.phase !== "idle") return;
  $("k").value = h.k;
  setLiveDefault();
  $("epochs").value = h.nepochs;
  $("seed").value = String(h.seed);
  $("trainN").value = String(h.trainOption);
  if (h.window !== state.window) {
    $("window").value = h.window;
    selectWindow(h.window);
  }
}

// A live redraw labels every pixel, which is slow for a large k. Changes the switch only when k crosses the limit.
const LIVE_MAX_K = 16;
function setLiveDefault() {
  const small = $("k").value <= LIVE_MAX_K;
  if (small !== state.smallK) $("live").checked = state.smallK = small;
}
$("k").addEventListener("calciteSliderChange", setLiveDefault);
$("run").addEventListener("click", startRun);
$("stop").addEventListener("click", () => worker.postMessage({ type: "stop" }));
$("window").addEventListener("calciteSegmentedControlChange", (e) => selectWindow(e.target.value));
$("colour").addEventListener("calciteSegmentedControlChange", (e) => {
  state.colour = e.target.value;
  if (state.labels) drawClusters();
});
$("compare").addEventListener("calciteSegmentedControlChange", (e) => setCompare(e.target.value));
$("opacity").addEventListener("calciteSliderInput", (e) => {
  wcLayer.opacity = clusterLayer.opacity = e.target.value / 100;
});
$("pixel").addEventListener("calcitePanelClose", (e) => {
  e.target.closed = false;
  e.target.hidden = true;
  state.clicked = null;
});

mapEl.addEventListener("arcgisViewClick", ({ detail }) => {
  const [xmin, ymin, xmax, ymax] = meta.windows[state.window].extent;
  const { x, y } = detail.mapPoint;
  const col = Math.floor(((x - xmin) / (xmax - xmin)) * size), row = Math.floor(((ymax - y) / (ymax - ymin)) * size);
  const i = row * size + col;
  if (!state.wc || col < 0 || col >= size || row < 0 || row >= size || !state.wc[i]) return;
  state.clicked = { i, lon: detail.mapPoint.longitude, lat: detail.mapPoint.latitude };
  renderPixel();
  worker.postMessage({ type: "pixel", i });
});
function renderPixel(top) {
  const { i, lon, lat } = state.clicked, line = (a, b) => el("p", { className: "hint" }, el("b", {}, a + ": "), b);
  const box = $("pixel").querySelector(".stack");
  const label = state.labels?.[i] ?? -1;
  box.replaceChildren(
    line("Location", `${fmt(lat, 4)}, ${fmt(lon, 4)}`),
    line("WorldCover", className(state.wc[i])),
    line("Cluster", label < 0 ? "run e-means first" : String(label + 1)));
  if (top) {
    box.append(line("Nearest centroids", ""),
      ...top.map(([j, s]) => el("p", { className: "hint" }, swatch(clusterColour(j)), `cluster ${j + 1}, cosine ${fmt(s, 4)}`)));
  }
  $("pixel").hidden = false;
}

// ---------- worker messages ----------
worker.onerror = (e) => {
  setPhase("idle");
  showError(e.message || "The worker could not start. Use a web server, not a file:// address.");
};
worker.onmessage = ({ data: m }) => {
  if (m.type === "loaded") {
    if (m.window !== state.window) return;
    state.wc = m.wc;
    drawWorldCover();
    renderLegend();
    setPhase("idle", `${m.n.toLocaleString("en")} pixels loaded.`);
  } else if (m.type === "progress") {
    $("progress").value = m.value;
  } else if (m.type === "epoch") {
    state.curve.push({ epoch: m.epoch, y: m.meanCosine });
    $("status").textContent = `Epoch ${m.epoch} of ${m.nepochs}`;
    renderChart();
  } else if (m.type === "labels") {
    apply(m.labels, m.k);
    if (state.clicked) renderPixel();                      // without the nearest centroids of the older run
  } else if (m.type === "done") {
    const { labels, ...run } = m;                          // the history keeps no label arrays
    state.run = { ...run, window: state.window, ...apply(labels, m.k) };
    state.history.unshift(state.run);
    renderResult();
    if (state.clicked) worker.postMessage({ type: "pixel", i: state.clicked.i });
    setPhase("idle", `Done. Cluster revivals: ${m.nSplits}.`);
  } else if (m.type === "stopped") {
    setPhase("idle", "Stopped. The map shows the last redrawn epoch, if any.");
  } else if (m.type === "pixel") {
    if (state.clicked?.i === m.i && state.run) renderPixel(m.top);
  } else if (m.type === "error") {
    setPhase("idle");
    showError(m.message);
  }
};

selectWindow("test");
