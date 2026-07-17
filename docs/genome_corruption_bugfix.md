# EXAMM saved-genome corruption: root cause, fix, and verification

**Date:** 2026-07-15
**Symptom:** a saved `global_best_genome_*.bin` does not reproduce the
`best_validation_mse` recorded in its own `.txt`/`.bin` when re-evaluated on the
validation set. In our 50-stock x 10-run baseline, **72/500 genomes (14.4%)**
were corrupted (reproduced/recorded val-MSE ratio > 1.5x, worst case 540x),
plus 35 more mild mismatches (1.05-1.5x).

Because the corrupted artifact keeps the *inherited, excellent* fitness value,
corrupted genomes preferentially **win validation-based model selection**: 4 of
our 50 val-selected "best" genomes were corrupt (AKAM, LNT, TECH, UHS), which
originally looked like degenerate training runs.

## Root cause

`sort_RNN_Nodes_by_depth` (rnn/rnn_node_interface.hxx) compared **depth only,
with no tie-breaker**. Node depths tie constantly: all input nodes share depth
0, and `split_node` creates hidden nodes with identical depths.

`RNN_Genome::copy()` hands the copied nodes to the genome constructor, which
re-sorts them with `std::sort`. `std::sort` is **not stable**: with libstdc++
(gcc, used on Anvil) it switches from insertion sort to quicksort partitioning
for ranges of **more than 16 elements**, and quicksort exchanges equal-depth
elements. So copying any genome with >16 nodes can silently permute its
equal-depth nodes.

`best_parameters` — the flat weight vector recorded during training — is copied
unchanged, still ordered for the **old** node order. Result: weight blocks land
on the wrong nodes while the fitness fields are untouched. (Input→data pairing
itself is NOT affected: `fix_parameter_orders` (rnn/rnn.cxx) re-derives the
input/output node ordering **by parameter name** whenever a network is built,
so the corruption is purely weight misassignment, not feature swapping.)

The saved "global best" genome is *always* such a copy
(`island_speciation_strategy.cxx`, on insert and at every extinction event), so
the corruption hits exactly the artifact that gets saved, while island members
(stored un-copied) keep evolving normally — which is why training curves looked
healthy while saved artifacts were broken.

This explains every observed property:

| Observation | Explanation |
|---|---|
| Intermittent (~14% of runs) | only global bests that grew >16 nodes are eligible |
| Graded severity (1.05x .. 540x) | which nodes swap: same-type swaps are mild; LSTM (~11 weights) <-> simple (1 weight) swaps shift every downstream block |
| Clusters by stock | some datasets reward larger evolved networks |
| Flat across run indices | per-run coin flip, no time/ordering component |
| Evolution itself stayed healthy | islands hold originals; only saved copies corrupt |

**Standalone proof:** sorting already-sorted genome-shaped depth arrays
(6 ties at depth 0 + hidden nodes + output) with the tie-less comparator under
g++/libstdc++ permutes **150/200** arrays (everything >16 nodes); e.g. the six
input nodes come out reordered `1 0 5 4 3 2`. With the fixed comparator: **0/200**.

## Fix (three parts)

1. **Root cause** — `rnn/rnn_node_interface.hxx`: `sort_RNN_Nodes_by_depth` now
   tie-breaks on the node's unique `innovation_number`, giving a total order.
   Sorting becomes deterministic and permutation-free; all `upper_bound`
   insertion sites use the same comparator, so genomes stay in total order
   through mutation, and `copy()`'s re-sort becomes a no-op.

   **Behavioral impact (audited across the whole codebase):** no evolutionary
   decision changes — mutation targeting is uniform over order-free sets,
   crossover matches genes by innovation number with weights carried in node
   objects, equal-depth nodes can never share a forward edge
   (rnn_genome.cxx:1782) so computed functions are order-invariant, and
   input/output↔data pairing is name-matched (`fix_parameter_orders`). The
   only effects beyond the bug fix itself: (i) sub-ULP floating-point
   summation-order differences (same class as switching compilers), and
   (ii) `RNN_Genome::equals()` — which compares nodes positionally and gates
   island duplicate detection — becomes canonical, so true duplicates are no
   longer occasionally missed due to tie-order mismatch. Bias-free, in the
   as-designed direction.
2. **Extinction-path memory leak** — `examm/island_speciation_strategy.cxx`:
   at every extinction event upstream runs
   `global_best_genome = get_best_genome()->copy()`. Since `get_best_genome()`
   returns `global_best_genome` itself, this is a **self-copy: a semantic
   no-op that leaks the previous copy** each extinction. The fix copies
   first, deletes the old object, then assigns — byte-for-byte identical
   behavior, leak closed. (Cautionary note: an earlier attempt that deleted
   BEFORE copying was a use-after-free — the delete frees the very object
   being copied — and crashed ~20% of a 100-run campaign on glibc. Order
   matters.)
3. **Permanent guard** — `scripts/stock_run/verify_genome.sh`: after every
   training run, the saved genome is re-evaluated on the validation set and
   must reproduce its recorded `best_validation_mse` within 2%, or the run is
   marked failed (no `.done`). Wired into `baseline_train.sh`. Any future
   fitness/weights divergence, whatever the cause, fails loudly at the source.
   `rnn_examples/copy_check.cxx` is a standalone diagnostic that loads a
   genome, calls `copy()`, and asserts the copy evaluates identically.

## Verification

- Comparator demo (g++/libstdc++): 150/200 permuted before, 0/200 after.
- `copy_check` on a real 21-node genome (gcc builds of pre-fix vs fixed code):
  pre-fix, a single `copy()` corrupts val MSE 0.00201 -> 0.408 (**203x**);
  fixed, the copy evaluates identically (ratio 0.99994 — floating-point
  summation-order jitter from the deterministic edge re-sort, benign).
- Real KMX training runs, exact baseline settings, gcc/libstdc++ builds,
  each run's saved global best re-evaluated on the val set:

  | build | clean | corrupt | corrupt ratios |
  |---|---|---|---|
  | pre-fix | 2/5 | **3/5** | 2.6x, 35.6x, 54.1x |
  | fixed | **5/5** | 0/5 | — |

  The pre-fix corruption rate matches KMX on Anvil (5/10). The fixed build
  stayed clean including runs whose networks grew past the 16-element sort
  threshold (19 and 21 visible nodes). Note: corruption can appear below 16
  *visible* nodes because the sorted array also contains disabled nodes.
- Backward compatibility: existing clean `.bin`s load and evaluate bit-identically
  under the fixed build (the load path never re-sorts nodes). Verified on the
  local AKAM smoke-test genome (recorded 0.002010 = reproduced 0.002010).

## Deployment / retrain plan (Anvil)

1. **Rebuild** with the fix (safe at any time — the genome *load* path is
   unchanged, so existing `.bin`s, the corrected-selection sweep, and
   prediction regeneration all behave identically under the new binary):
   `module load gcc/11.2.0 openmpi/4.1.6 && cmake --build build -j8`
2. **Pilot (recommended, ~1 hr):** rerun the three worst stocks
   (KMX, LH, PKG = 13/30 corrupt pre-fix) as a 30-job array into a fresh
   output dir. `baseline_train.sh` now verifies every saved genome; expect
   **0 verification failures** and no `SAVED GENOME FAILED VERIFICATION`
   lines in `failures.log`.
3. **Full retrain:** delete/rename `test_output/baseline`, resubmit the
   500-job array (`sbatch scripts/stock_run/anvil_baseline.sb`). Every run
   now self-verifies before writing `.done`, so a clean completion *is* the
   verification. As a final belt-and-braces check, re-run the 500-genome
   val-reproduce sweep on the new collection; expect 0 mismatches
   (vs 72 + 35 mild pre-fix).
4. Then: `collect_best_genomes.sh` → `evaluate_baseline.sh` → trading
   pipeline, all unchanged.

## Implications and caveats

- **Old `.bin`s remain readable and evaluable** — analysis/prediction pipelines
  are unaffected.
- **Never use pre-fix `.bin`s as seed genomes** (transfer learning etc.) under
  the fixed code: their node order may violate the new total order, so their
  first `copy()` would re-sort and misalign the weights. Fresh training only.
- The recorded `best_validation_mse` of pre-fix runs is unreliable for model
  selection; select using *re-evaluated* val MSE
  (`test_output/baseline_best/val_reproduce.csv` / `index_corrected.csv`).
- Any prior results produced from *saved* EXAMM genomes with >16 nodes (other
  projects in the lab included) may be affected; the sweep in
  `verify_genome.sh` / the val_reproduce loop is directly reusable to audit them.
- Test-set metrics computed from saved genomes' actual predictions are valid
  *for those (possibly corrupted) weights* — corrupted models just predict
  worse. Best-of-runs summaries are robust (corrupt runs never win); val-selected
  summaries are the ones that were poisoned.
