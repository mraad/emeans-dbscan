// Scores of a clustering against WorldCover, as in aef_cluster.py.
export const WATER = 80;

// counts[class][cluster] for pixels with a label >= 0. Delete the WATER row for the land scores.
export function contingency(wc, labels, k) {
  const table = new Map();
  for (let i = 0; i < labels.length; i++) {
    const c = wc[i];
    if (labels[i] < 0) continue;
    let row = table.get(c);
    if (!row) table.set(c, (row = new Float64Array(k)));
    row[labels[i]]++;
  }
  return table;
}

const pairs = (v) => (v * (v - 1)) / 2;

// Adjusted Rand index and purity from a contingency table.
export function scores(table, k) {
  const cols = new Float64Array(k), colMax = new Float64Array(k);
  let n = 0, index = 0, rowPairs = 0;
  for (const row of table.values()) {
    let r = 0;
    row.forEach((v, j) => {
      index += pairs(v);
      cols[j] += v;
      colMax[j] = Math.max(colMax[j], v);
      r += v;
    });
    rowPairs += pairs(r);
    n += r;
  }
  const colPairs = cols.reduce((s, v) => s + pairs(v), 0);
  const expected = (rowPairs * colPairs) / pairs(n), max = (rowPairs + colPairs) / 2;
  return {
    ari: max === expected ? 1 : (index - expected) / (max - expected),
    purity: colMax.reduce((s, v) => s + v, 0) / n,
  };
}
