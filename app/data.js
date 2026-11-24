// AEF dequantization, (v / 127.5)^2 * sign(v), then unit length. Keeps valid pixels (wc > 0) only (first `limit` of them).
export function dequantize(emb, wc, d, limit = Infinity) {
  const index = [];
  for (let i = 0; i < wc.length && index.length < limit; i++) if (wc[i] > 0) index.push(i);
  const x = new Float32Array(index.length * d);
  index.forEach((src, r) => {
    let sq = 0;
    for (let c = 0; c < d; c++) {
      const v = emb[src * d + c] / 127.5, q = v * Math.abs(v);
      x[r * d + c] = q;
      sq += q * q;
    }
    const norm = Math.sqrt(sq) || 1;
    for (let c = 0; c < d; c++) x[r * d + c] /= norm;
  });
  return { x, index: Int32Array.from(index) };
}
