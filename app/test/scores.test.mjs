// Needs no data. The expected ARI values are those of sklearn.metrics.adjusted_rand_score.
import assert from "node:assert/strict";
import test from "node:test";
import { WATER, contingency, scores } from "../metrics.js";

const of = (wc, labels, k) => scores(contingency(wc, labels, k), k);

test("perfect match with other names", () => assert.deepEqual(of([10, 10, 50, 50], [1, 1, 0, 0], 2), { ari: 1, purity: 1 }));

test("one class split in two", () => {
  const r = of([10, 10, 50, 50], [0, 0, 1, 2], 3);
  assert.ok(Math.abs(r.ari - 0.5714285714285714) < 1e-12);
  assert.equal(r.purity, 1);
});

test("independent labels", () => {
  const r = of([10, 10, 50, 50], [0, 1, 0, 1], 2);
  assert.ok(Math.abs(r.ari + 0.5) < 1e-12);
  assert.equal(r.purity, 0.5);
});

test("no-data labels and water are left out", () => {
  const table = contingency([10, 10, 50, 50, WATER, 10], [0, 0, 1, 1, 0, -1], 2);
  table.delete(WATER);
  assert.deepEqual(scores(table, 2), { ari: 1, purity: 1 });
});
