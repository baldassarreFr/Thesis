#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Simple wrapper to run Plain-DETR config files with torchrun
# Usage: GPUS_PER_NODE=2 ./scripts/run_config.sh <num_gpus> <path_to_config>
# ------------------------------------------------------------------------

GPUS="${1:?}"
CONFIG_FILE="${2:?}"
GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}

# Change to project directory
cd /root/Plain-DETR-v2

# First, create a clean version of the config without comments
# Remove lines that start with # and empty lines
# Also remove the python command prefix if present
tmpfile=$(mktemp)
grep -v '^#' "$CONFIG_FILE" | grep -v '^[[:space:]]*#' | grep -v '^[[:space:]]*$' > "$tmpfile"

# Remove python command prefix if present
sed -i 's/^python -u -m plain_detr.main //' "$tmpfile"
sed -i 's/^FILE_NAME=.*$//' "$tmpfile"
sed -i 's/^EXP_DIR=.*$//' "$tmpfile"
sed -i 's/^PY_ARGS=.*$//' "$tmpfile"
sed -i 's/^set -x$//' "$tmpfile"
sed -i 's/^#!/usr.*$//' "$tmpfile"

# Now extract --args.X value patterns - only lines that start the argument
PY_ARGS=""
while IFS= read -r line; do
    # Skip empty lines
    [[ -z "$line" ]] && continue
    
    # Only match lines that start with --args. (after trimming)
    trimmed=$(echo "$line" | sed 's/^[[:space:]]*//')
    if [[ "$trimmed" == --args.* ]]; then
        # Clean up the value (remove trailing \ and quotes)
        clean=$(echo "$trimmed" | sed 's/[[:space:]]*\\[[:space:]]*$//' | sed 's/[[:space:]]*"[[:space:]]*$//')
        PY_ARGS="$PY_ARGS $clean"
    fi
done < "$tmpfile"

rm "$tmpfile"

# Clean up extra spaces
PY_ARGS=$(echo "$PY_ARGS" | sed 's/  */ /g')

echo "Running with GPUS=$GPUS_PER_NODE"
echo "Config: $CONFIG_FILE"

# Run with torchrun via uv in background, save output
uv run torchrun --standalone --nproc-per-node=$GPUS_PER_NODE -m plain_detr.main $PY_ARGS \
    > exps/dinov3_vit_small_boxrpe/log.txt 2>&1 &
echo "Started with PID: $!"