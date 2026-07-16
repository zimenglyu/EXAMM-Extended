#!/bin/sh
# Verify that a run's saved global-best genome reproduces its recorded
# best_validation_mse when the .bin is re-evaluated on the validation set.
# Catches genome-serialization corruption (fitness/weights mismatch) at the
# source, so a corrupted artifact can never silently enter downstream analysis.
#
# Usage: sh scripts/stock_run/verify_genome.sh <RUN_DIR> <VAL_CSV>
#   RUN_DIR: an EXAMM output dir containing global_best_genome_<id>.{bin,txt}
#   VAL_CSV: the validation file the run trained against
#
# Exit 0 = verified (reproduced val MSE within 2% of recorded).
# Exit 1 = corrupt or unverifiable (missing files, eval failure, mismatch).
#
# Background: EXAMM's node depth-sort had no tie-breaker, so std::sort could
# permute equal-depth nodes on genome copy while best_parameters kept the old
# order -- 72/500 baseline genomes (14.4%) were saved corrupted before the fix
# in rnn/rnn_node_interface.hxx. This guard makes any recurrence loud.

REPO=$(cd "$(dirname "$0")/../.." && pwd)
BIN=$REPO/build/rnn_examples/evaluate_rnn
RUN_DIR=$1
VAL_CSV=$2

if [ -z "$RUN_DIR" ] || [ -z "$VAL_CSV" ]; then
    echo "usage: verify_genome.sh <RUN_DIR> <VAL_CSV>" >&2
    exit 1
fi
if [ ! -x "$BIN" ]; then
    echo "VERIFY ERROR: $BIN not found -- build evaluate_rnn first" >&2
    exit 1
fi
if [ ! -f "$VAL_CSV" ]; then
    echo "VERIFY ERROR: validation file $VAL_CSV not found" >&2
    exit 1
fi

# same best-genome discovery as collect_best_genomes.sh: if a run left more
# than one global_best file, verify the one with the lowest recorded MSE
best_txt=$(for f in "$RUN_DIR"/global_best_genome_*.txt; do
    [ -f "$f" ] || continue
    mse=$(grep -o 'best_validation_mse: [0-9.eE+-]*' "$f" | awk '{print $2}')
    [ -n "$mse" ] && echo "$mse $f"
done | sort -n | head -1 | awk '{print $2}')

if [ -z "$best_txt" ]; then
    echo "VERIFY ERROR: no global_best_genome_*.txt with a recorded MSE in $RUN_DIR" >&2
    exit 1
fi
best_bin=${best_txt%.txt}.bin
if [ ! -f "$best_bin" ]; then
    echo "VERIFY ERROR: $best_bin missing" >&2
    exit 1
fi

recorded=$(grep -o 'best_validation_mse: [0-9.eE+-]*' "$best_txt" | awk '{print $2}')

tmp=$(mktemp -d)
reproduced=$("$BIN" --genome_file "$best_bin" --testing_filenames "$VAL_CSV" \
    --time_offset 1 --output_directory "$tmp" \
    --std_message_level INFO --file_message_level NONE 2>&1 | awk '/MSE:/{print $NF; exit}')
rm -rf "$tmp"

if [ -z "$reproduced" ]; then
    echo "VERIFY ERROR: evaluate_rnn produced no MSE for $best_bin" >&2
    exit 1
fi

python3 - "$recorded" "$reproduced" "$best_bin" <<'PY'
import sys
recorded, reproduced = float(sys.argv[1]), float(sys.argv[2])
if recorded <= 0:
    print(f"VERIFY ERROR: nonpositive recorded MSE {recorded}")
    sys.exit(1)
ratio = reproduced / recorded
if 0.98 <= ratio <= 1.02:
    print(f"VERIFY OK: {sys.argv[3]} recorded {recorded:.6g} reproduced {reproduced:.6g} (ratio {ratio:.4f})")
    sys.exit(0)
print(f"VERIFY FAILED: {sys.argv[3]} recorded {recorded:.6g} but weights reproduce {reproduced:.6g} (ratio {ratio:.2f}) -- corrupted save")
sys.exit(1)
PY
