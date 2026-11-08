# Raspberry Pi coal experiment — November 2, 2026

EXAMM's search ran on a Mac, with candidate training and validation performed by
a Raspberry Pi Zero 2 W through the Pi-training integration. The configuration
used the coal dataset, ten training files, two validation files, AdaGrad, and
five training iterations per candidate. The search was configured for 2,000
genomes and stopped early. The exact cause of interruption is not established
by these preserved logs.

## Preserved observations

- The worker recorded 852 received and returned genomes, in matching order.
- 851 genomes completed five logged iterations each: 4,255 iterations in total.
  Each recorded iteration has a finite, positive gradient norm.
- Genome 828 returned a nonfinite validation score (`NaN`) without completing
  a logged iteration. It must not be treated as a successful training result.
- Genome 1's validation MSE decreased from approximately 2.074005 before
  training to 1.7891731499 after five iterations.
- The lowest returned validation MSE was 0.0019620940491085533, from genome 663.
  Its initial weights already achieved that score; its five training iterations
  did not improve its best validation score. This is not a held-out test score.
- The recorded duration was approximately 7 hours 43 minutes. The timing CSV
  has 849 usable intervals, with a mean of about 32.6 seconds per genome.

The timing estimates use differences between original successive log modification
times. They include validation and coordination overhead, not just gradient
computation. Intervals adjacent to invalid/incomplete records were excluded from
the chart. Git does not preserve original modification times on checkout; use
the original timestamps recorded in the CSV rather than filesystem timestamps.

## Files

| Path | Contents |
| --- | --- |
| `logs/pi_worker` | Dataset and optimizer setup; received/returned IDs, scores, and parameter fingerprints. |
| `logs/genome_*_pi` | Initial validation score, per-iteration errors and gradient norms, and completion messages. |
| `pi_coal_validation_results.csv` | One row per returned candidate, extracted from the worker and genome logs. |
| `pi_coal_estimated_genome_times.csv` | Original timestamps, estimated intervals, and chart inclusion/exclusion decisions. |
| `pi_coal_training_time.png` | Estimated time per genome and recent timing trend. |

## Limits of this archive

These are logs and derived measurements, not saved model weights or a full search
checkpoint. The experiment used `--save_genome_option the_best`; its final model
file was not recovered. The complete Mac-side logs are not included here.
The current coal launcher instead requests `all_best_genomes` to save new best
models during execution. That later change must not be attributed to this run.

Matching a parameter fingerprint establishes a consistency check between copies;
without before/after weight vectors, a returned fingerprint alone does not prove
that training changed the weights. The training evidence here is the recorded
backpropagation iterations, gradient norms, and changing within-candidate errors,
interpreted alongside the worker's call to EXAMM's existing training function.
