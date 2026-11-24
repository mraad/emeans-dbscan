// Needs no data. The expected values come from numpy.random.RandomState.
import assert from "node:assert/strict";
import test from "node:test";
import { RandomState } from "../emeans.js";

test("permutation", () => {
  const first20 = (seed, n) => [...new RandomState(seed).permutation(n)].slice(0, 20);
  assert.deepEqual(first20(0, 1000), [993,859,298,553,672,971,27,231,306,706,496,558,784,239,578,55,906,175,14,77]);
  assert.deepEqual(first20(1234, 1000), [681,990,155,768,438,363,688,555,620,99,756,428,888,89,982,495,29,175,466,40]);
  // the seed of epoch 0 for seed 1234, above 2 ** 16 rows
  assert.deepEqual(first20((1234 * 1000003) % 2 ** 32, 200000), [2987,29030,94680,22405,17463,54060,42253,144163,48207,55142,172353,62171,18498,158942,162408,151167,180910,78488,38405,147323]);
});

test("random_sample", () => {
  const r = new RandomState(7);
  for (const v of [0.07630828937395717,0.7799187922401146,0.4384092314408935,0.7234651778309412,0.9779895119966027]) assert.equal(r.random(), v);
});
