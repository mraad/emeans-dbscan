// node --test app/test/*.mjs   (needs app/data from prepare_data.py and test/reference.json from parity_faiss.py)
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";
import { assignAll, recipe, train } from "../emeans.js";
import { WATER, contingency, scores } from "../metrics.js";
import { dequantize } from "../data.js";

const here = new URL(".", import.meta.url);
const has = (f) => existsSync(new URL(f, here));
const ready = has("reference.json") && has("../data/test_emb.bin");
test("parity data is present", { skip: ready ? false : "run prepare_data.py and parity_faiss.py first" }, () => {});

if (ready) {
  const ref = JSON.parse(readFileSync(new URL("reference.json", here)));
  const emb = new Int8Array(readFileSync(new URL("../data/test_emb.bin", here)).buffer);
  const wc = new Uint8Array(readFileSync(new URL("../data/test_wc.bin", here)));
  const drain = (g) => { let r; do r = g.next(); while (!r.done); return r.value; };

  const { x, index } = dequantize(emb, wc, 64, ref.n);
  for (const run of ref.runs) {
    test(`k=${run.k} seed=${run.seed} epochs=${run.nepochs} matches faiss ${ref.faiss}`, () => {
      const rc = recipe(ref.n, run.k, 64, run.nepochs);
      assert.equal(rc.batch, run.batch);
      assert.ok(Math.abs(rc.alpha0 - run.alpha0) < 1e-9);
      const res = drain(train(x, 64, run.k, { nepochs: run.nepochs, seed: run.seed }));
      const { labels, meanCosine } = drain(assignAll(x, res.centroids, run.k, 64));
      const same = labels.reduce((s, v, i) => s + (v === run.labels[i]), 0) / labels.length;
      const y = Uint8Array.from(index, (i) => wc[i]);
      const table = contingency(y, labels, run.k);
      table.delete(WATER);
      const ari = scores(table, run.k).ari;
      console.log(`  same labels ${(100 * same).toFixed(3)}%, splits ${res.nSplits}/${run.n_splits}, cosine ${meanCosine.toFixed(6)}/${run.mean_cosine.toFixed(6)}, land ARI ${ari.toFixed(4)}/${run.land_ari.toFixed(4)}`);
      assert.equal(res.nSplits, run.n_splits);
      // A split makes near-identical twins, so FP32 rounding decides some labels: faiss with inputs perturbed by 1e-7
      // also differs from itself by 2 to 14% of the labels. Without splits the labels must match.
      assert.ok(Math.abs(meanCosine - run.mean_cosine) < (run.n_splits ? 1e-4 : 1e-5));
      assert.ok(Math.abs(ari - run.land_ari) < (run.n_splits ? 5e-3 : 1e-3));
      if (!run.n_splits) assert.ok(same > 0.999);
    });
  }
}
