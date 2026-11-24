// Spherical e-means, ported from faiss 1.16.0 (contrib/e_means: base.py, e_means.py; MIT license, Meta Platforms).
// The random numbers follow NumPy's RandomState, so a seed gives the same start centroids and batches as NumpyEmeans.

const ALPHA_MIN_FACTOR = 0.1;
const CHECK_UNUSED_EVERY = 5;
const INIT_USAGE = 2 ** -16;
const SPLIT_EPSILON = 1e-4;
const UNUSED_THRESHOLD = 0.1;
const f32 = Math.fround;
const NO_REVIVAL_MAX_D = 2;
const AUTO_QUALITY_MAX_D = 16;
const BATCH_LAWS = {
  throughput: [0.035134, 0.134749, 0.876473, 0.813344],
  quality: [0.00439175, 0.134749, 0.876473, 0.813344],
};
const ALPHA_LAW = [0.8244, -0.4617, 0.9003, -0.4498, -0.1057];

// MT19937 with the seeding and draws of numpy.random.RandomState.
export class RandomState {
  constructor(seed) {
    const mt = (this.mt = new Uint32Array(624));
    mt[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) mt[i] = Math.imul(1812433253, mt[i - 1] ^ (mt[i - 1] >>> 30)) + i;
    this.pos = 624;
  }
  uint32() {
    const mt = this.mt;
    if (this.pos === 624) {
      for (let i = 0; i < 624; i++) {
        const y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff);
        mt[i] = mt[(i + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
      }
      this.pos = 0;
    }
    let y = mt[this.pos++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    return (y ^ (y >>> 18)) >>> 0;
  }
  random() {
    const a = this.uint32() >>> 5, b = this.uint32() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  }
  interval(max) {                                          // uniform integer in [0, max], by masked rejection
    let mask = max;
    for (const s of [1, 2, 4, 8, 16]) mask |= mask >>> s;
    let v;
    do v = (this.uint32() & mask) >>> 0; while (v > max);
    return v;
  }
  permutation(n) {
    const a = new Int32Array(n);
    for (let i = 0; i < n; i++) a[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = this.interval(i), t = a[i];
      a[i] = a[j];
      a[j] = t;
    }
    return a;
  }
}

function roundPowerOfTwo(v) {
  if (v <= 1) return 1;
  const low = 2 ** Math.floor(Math.log2(v)), high = 2 * low;
  return v / low < high / v ? low : high;
}

// The batch size and first step that e-means selects for n, k and the number of epochs.
export function recipe(n, k, d, nepochs) {
  const law = d <= AUTO_QUALITY_MAX_D ? "quality" : "throughput";
  const [c, bk, bn, bt] = BATCH_LAWS[law];
  let batch = roundPowerOfTwo(c * k ** bk * n ** bn * nepochs ** bt);
  batch = Math.min(batch, law === "throughput" ? roundPowerOfTwo(n / 2) : Math.floor(n / 2));
  batch = Math.max(1, Math.min(batch, n));
  const [ca, ak, ab, an, at] = ALPHA_LAW;
  const alpha0 = Math.min(Math.max(ca * k ** ak * batch ** ab * n ** an * nepochs ** at, 1e-4), 1);
  return { batch, alpha0, totalSteps: Math.ceil(n / batch) * nepochs };
}

export function normalizeRows(src, d, out = new Float32Array(src.length)) {
  for (let o = 0; o < src.length; o += d) {
    let max = 0, sq = 0;
    for (let i = 0; i < d; i++) max = Math.max(max, Math.abs(src[o + i]));
    max = max || 1;
    for (let i = 0; i < d; i++) sq += (src[o + i] / max) ** 2;
    const norm = Math.sqrt(sq) || 1;
    for (let i = 0; i < d; i++) out[o + i] = src[o + i] / max / norm;
  }
  return out;
}

// Nearest centroid (largest inner product) of the row at x[o..o+d). Returns the index; sets nearest.sim. d must be a multiple of 4.
export function nearest(x, o, C, k, d) {
  let best = 0, bestSim = -Infinity;
  for (let j = 0, p = 0; j < k; j++, p += d) {
    let s = 0;
    for (let i = 0; i < d; i += 4)
      s += x[o + i] * C[p + i] + x[o + i + 1] * C[p + i + 1] + x[o + i + 2] * C[p + i + 2] + x[o + i + 3] * C[p + i + 3];
    if (s > bestSim) { bestSim = s; best = j; }
  }
  nearest.sim = bestSim;
  return best;
}

// Labels of every row against final centroids, and the mean cosine similarity to the nearest centroid.
// A generator, so a worker can stay responsive: it yields after every 20,000 rows.
export function* assignAll(x, C, k, d) {
  const n = x.length / d, labels = new Int32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    labels[i] = nearest(x, i * d, C, k, d);
    sum += nearest.sim;
    if (i % 20000 === 19999) yield;
  }
  return { labels, meanCosine: sum / n };
}

// Trains on unit rows x (Float32Array, n x d). Yields {step, totalSteps} after each batch, and
// {epoch, meanCosine, centroids} after each epoch. Returns {centroids, nSplits, batch}.
export function* train(x, d, k, { nepochs = 5, seed = 1234 } = {}) {
  const n = x.length / d;
  const { batch, alpha0, totalSteps } = recipe(n, k, d, nepochs);
  const C = new Float32Array(k * d);                       // EMA positions c_j = S_j / N_j; not unit length
  const init = new RandomState(seed).permutation(n);
  for (let j = 0; j < k; j++) C.set(x.subarray(init[j] * d, (init[j] + 1) * d), j * d);
  normalizeRows(C, d, C);
  const usage = new Float32Array(k).fill(INIT_USAGE);      // EMA masses N_j
  const splitRng = new RandomState(seed);
  const tau = d <= NO_REVIVAL_MAX_D ? 0 : UNUSED_THRESHOLD;
  const sums = new Float32Array(k * d), counts = new Float32Array(k);
  const passSums = new Float32Array(k * d), passCounts = new Float64Array(k);
  const unit = new Float32Array(k * d);
  let step = 0, nSplits = 0;

  for (let epoch = 0; epoch < nepochs; epoch++) {
    const order = new RandomState((seed * 1000003 + epoch) % 2 ** 32).permutation(n);
    passCounts.fill(0);
    let objective = 0;
    for (let begin = 0; begin < n; begin += batch) {
      const end = Math.min(begin + batch, n);
      normalizeRows(C, d, unit);
      sums.fill(0);
      counts.fill(0);
      for (let r = begin; r < end; r++) {
        const o = order[r] * d, j = nearest(x, o, unit, k, d), p = j * d;
        objective += nearest.sim;
        counts[j]++;
        for (let i = 0; i < d; i++) sums[p + i] += x[o + i];
      }
      const cosine = 0.5 * (1 + Math.cos((Math.PI * Math.min(step, totalSteps)) / Math.max(1, totalSteps)));
      const alpha = alpha0 * (ALPHA_MIN_FACTOR + (1 - ALPHA_MIN_FACTOR) * cosine);
      const a32 = f32(alpha), decay32 = f32(1 - alpha);     // NumPy computes these updates in FP32
      const replace = alpha === 1;
      for (let j = 0; j < k; j++) {
        passCounts[j] += counts[j];
        if (!replace) usage[j] *= decay32;
        if (counts[j] === 0) continue;
        const gain = f32(a32 * counts[j]);
        const mass = f32((replace ? 0 : usage[j]) + gain);
        const weight = f32(gain / mass);
        for (let i = j * d; i < (j + 1) * d; i++) {
          const mean = f32(sums[i] / counts[j]);
          C[i] = weight === 1 ? mean : C[i] + f32(weight * f32(mean - C[i]));
        }
        usage[j] = mass;
      }
      if (epoch === nepochs - 1) for (let i = 0; i < passSums.length; i++) passSums[i] += sums[i];
      if ((step + 1) % CHECK_UNUSED_EVERY === 0) nSplits += reviveUnused(C, usage, k, d, tau, splitRng);
      step++;
      yield { step, totalSteps };
    }
    if (epoch < nepochs - 1) {
      yield { epoch: epoch + 1, meanCosine: objective / n, centroids: normalizeRows(C, d) };
    } else {                                               // closing step: exact means of the last epoch
      for (let j = 0; j < k; j++)
        if (passCounts[j] > 0) for (let i = j * d; i < (j + 1) * d; i++) C[i] = passSums[i] / passCounts[j];
      const centroids = normalizeRows(C, d);
      yield { epoch: epoch + 1, meanCosine: objective / n, centroids };
      return { centroids, nSplits, batch };
    }
  }
}

// faiss 1.16.0 rule: centroids with mass < tau * mean(mass) take half of a donor, drawn in proportion to mass.
function reviveUnused(C, usage, k, d, tau, rng) {
  const total = f32(usage.reduce((s, u) => s + u, 0));
  const threshold = f32(f32(total * f32(tau)) / k);
  const targets = [], donors = [];
  for (let j = 0; j < k; j++) {
    if (usage[j] < threshold) targets.push(j);
    else if (usage[j] > 0) donors.push(j);
  }
  const nsplit = Math.min(targets.length, donors.length);
  if (nsplit === 0) return 0;
  const u = Array.from({ length: k }, () => Math.max(rng.random(), 2.2250738585072014e-308));
  const source = donors
    .map((j) => [Math.log(usage[j]) - Math.log(-Math.log(u[j])), j])
    .sort((a, b) => b[0] - a[0])
    .slice(0, nsplit)
    .map((e) => e[1]);
  source.forEach((s, i) => {
    const t = targets[i];
    for (let c = 0; c < d; c++) {
      const v = C[s * d + c], off = f32(SPLIT_EPSILON * (c % 2 ? -1 : 1));
      C[s * d + c] = v * f32(1 - off);
      C[t * d + c] = v * f32(1 + off);
    }
    // shortcut: no nextafter() fallback for identical twins; only an all-zero donor makes them, and unit vectors never average to one here.
    usage[s] /= 2;
    usage[t] = usage[s];
  });
  return nsplit;
}
