import { assignAll, nearest, train } from "./emeans.js";
import { dequantize } from "./data.js";

const STOPPED = Symbol("stopped");
let d, x, index, size, centroids, k, generation = 0;

// A macrotask without the 4 ms clamp of nested setTimeout.
const channel = new MessageChannel();
const tick = () => new Promise((r) => {
  channel.port1.onmessage = r;
  channel.port2.postMessage(0);
});

// Runs a generator to its end; pauses every ~50 ms so that "stop" messages arrive. Throws STOPPED when stopped.
async function drive(gen, id, onValue) {
  let last = performance.now(), r;
  while (!(r = gen.next()).done) {
    if (r.value) await onValue?.(r.value);
    if (performance.now() - last > 50) {
      await tick();
      if (id !== generation) throw STOPPED;
      last = performance.now();
    }
  }
  return r.value;
}

function fullLabels(labels) {                              // back to the pixel grid, -1 = no data
  const out = new Int16Array(size).fill(-1);
  labels.forEach((l, i) => (out[index[i]] = l));
  return out;
}

async function load({ window, d: dim }) {
  x = index = centroids = null;                            // free the old window before the new one arrives
  d = dim;
  const get = async (name) => {
    const res = await fetch(new URL(`data/${window}_${name}.bin`, import.meta.url));
    if (!res.ok) throw new Error(`${res.url}: HTTP ${res.status}. Run prepare_data.py first.`);
    return res.arrayBuffer();
  };
  const [emb, classes] = await Promise.all([get("emb"), get("wc")]);
  const wc = new Uint8Array(classes);
  size = wc.length;
  if (emb.byteLength !== size * d)
    throw new Error(`${window}_emb.bin has ${emb.byteLength} bytes, not ${size * d}. Run prepare_data.py again.`);
  ({ x, index } = dequantize(new Int8Array(emb), wc, d));
  postMessage({ type: "loaded", window, wc, n: index.length });
}

// Evenly spaced rows, so that the sample covers the whole window and is the same each time.
function sample(m) {
  const n = index.length;
  if (m >= n) return x;
  const step = n / m, rows = new Float32Array(m * d);
  for (let i = 0; i < m; i++) {
    const r = Math.floor((i + 0.5) * step);
    rows.set(x.subarray(r * d, (r + 1) * d), i * d);
  }
  return rows;
}

async function run({ k: kk, nepochs, seed, trainN, live }) {
  const id = ++generation, rows = sample(trainN);
  centroids = null;                                        // no pixel answers from an older run
  const t0 = performance.now();
  let paused = 0;                                          // time spent on live map updates
  const result = await drive(train(rows, d, kk, { nepochs, seed }), id, async (v) => {
    if (v.step) return postMessage({ type: "progress", value: v.step / v.totalSteps });
    postMessage({ type: "epoch", epoch: v.epoch, nepochs, meanCosine: v.meanCosine });
    if (!live || v.epoch === nepochs) return;
    const t = performance.now();
    const { labels } = await drive(assignAll(x, v.centroids, kk, d), id);
    postMessage({ type: "labels", labels: fullLabels(labels), k: kk });
    paused += performance.now() - t;
  });
  const trainMs = performance.now() - t0 - paused;
  const { labels, meanCosine } = await drive(assignAll(x, result.centroids, kk, d), id);
  centroids = result.centroids;
  k = kk;
  postMessage({
    type: "done", labels: fullLabels(labels), meanCosine, k, nepochs, seed, trainOption: trainN, trainN: rows.length / d,
    trainMs, nSplits: result.nSplits, batch: result.batch,
  });
}

function pixel({ i }) {
  const r = index.indexOf(i);
  if (r < 0 || !centroids) return;
  const sims = [];
  for (let j = 0; j < k; j++) {
    nearest(x, r * d, centroids.subarray(j * d, (j + 1) * d), 1, d);
    sims.push([j, nearest.sim]);
  }
  sims.sort((a, b) => b[1] - a[1]);
  postMessage({ type: "pixel", i, top: sims.slice(0, 3) });
}

const handlers = { load, run, pixel, stop: () => { generation++; } };
onmessage = async ({ data }) => {
  try {
    await handlers[data.type](data);
  } catch (e) {
    postMessage(e === STOPPED ? { type: "stopped" } : { type: "error", message: e.message });
  }
};
