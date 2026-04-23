#!/usr/bin/env bash
# Bootstrap script for Lambda Labs H100/A100 instances. Run this ONCE on
# the Lambda VM after SSHing in (assumes GCS bucket is temporarily set to
# allUsers:objectViewer so we can curl without auth).
#
# Usage on Lambda VM:
#   curl -s https://storage.googleapis.com/project-mongoose-494111-mongoose-training/lambda_bootstrap.sh -o bootstrap.sh
#   chmod +x bootstrap.sh
#   WANDB_API_KEY=... ./bootstrap.sh     # optional: streams to wandb.ai
#
# Or paste the contents directly via heredoc.
#
# What it does:
#   1. Apt-installs git, pip, etc.
#   2. Downloads source tarball from public GCS URL
#   3. pip install -e . the project (plus wandb if WANDB_API_KEY is set)
#   4. Downloads all 30 enriched caches in parallel
#   5. Kicks off Option A training (40 epochs, probe-aware vel + mixed
#      supervision, --use-l511 --use-t2d-hybrid); if WANDB_API_KEY is
#      present, adds --use-wandb so per-epoch metrics stream to wandb.ai
#   6. Periodic checkpoint upload back to GCS via curl PUT (requires
#      bucket to allow allUsers:objectCreator too — see Step 2 in the
#      walkthrough doc; OR rely on scp-back at the end)
set -eu

WORKSPACE=/workspace
BUCKET_URL=https://storage.googleapis.com/project-mongoose-494111-mongoose-training
EPOCHS=${EPOCHS:-40}
WANDB_PROJECT=${WANDB_PROJECT:-mongoose-v3}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-option-a-lambda-${EPOCHS}ep}

echo "=== System prep ==="
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git wget curl python3-pip

mkdir -p $WORKSPACE
cd $WORKSPACE

echo "=== Downloading source tarball ==="
curl -sSL -o mongoose-src.tar.gz "$BUCKET_URL/mongoose-src.tar.gz"
mkdir -p mongoose
tar -xzf mongoose-src.tar.gz -C mongoose

echo "=== pip install ==="
cd mongoose
pip install --quiet -e .
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "=== WANDB_API_KEY present — installing wandb ==="
    pip install --quiet wandb
fi

echo "=== Downloading caches (parallel, ~5 min on cloud network) ==="
cd $WORKSPACE
mkdir -p caches

# 27 training + 3 holdout = 30 total. Hardcoded list (matches what we
# enriched). Each cache has 6 files; total ~14 GB.
CACHES=(
    STB03-060A-02L58270w05-202G16j
    STB03-060A-02L58270w05-433B23e
    STB03-060B-02L58270w05-433H09f
    STB03-060C-02L58270w05-433H09a
    STB03-062A-02L58270w05-433H09i
    STB03-062B-02L58270w05-433B23h
    STB03-062B-02L58270w05-433H09c
    STB03-062D-02L58270w05-433H09g
    STB03-063A-02L58270w05-433B23g
    STB03-063A-02L58270w05-433H09e
    STB03-063B-02L58270w05-202G16a
    STB03-063B-02L58270w05-433B23b
    STB03-063C-02L58270w05-202G16d
    STB03-064A-02L58270w05-433H09k
    STB03-064B-02L58270w05-202G16b
    STB03-064C-02L58270w05-433B23c
    STB03-064D-02L58270w05-433H09d
    STB03-065A-02L58270w05-202G16c
    STB03-065A-02L58270w05-433B23f
    STB03-065B-02L58270w05-202G16f
    STB03-065B-02L58270w05-429D08c
    STB03-065C-02L58270w05-429D08i
    STB03-065D-02L58270w05-202G16e
    STB03-065D-02L58270w05-433H09b
    STB03-065E-02L58270w05-433B23i
    STB03-065F-02L58270w05-433B23d
    STB03-065F-02L58270w05-433H09h
    STB03-065G-02L58270w05-202G16h
    STB03-065G-02L58270w05-433B23a
    STB03-065H-02L58270w05-433H09j
)
FILES=(waveforms.bin offsets.npy conditioning.npy molecules.pkl manifest.json t2d_params.npy)

for cache in "${CACHES[@]}"; do
    mkdir -p "caches/$cache"
    for file in "${FILES[@]}"; do
        curl -sSL -o "caches/$cache/$file" "$BUCKET_URL/caches/$cache/$file" &
    done
done
wait
echo "Done downloading $(ls caches | wc -l) caches"

# Hold out the same 3 colors we've used all sprint (1 per color).
HOLDOUTS=(
    STB03-063B-02L58270w05-433B23b
    STB03-064D-02L58270w05-433H09d
    STB03-065H-02L58270w05-433H09j
)

CACHE_ARGS=()
for cache in "${CACHES[@]}"; do
    skip=false
    for h in "${HOLDOUTS[@]}"; do
        if [ "$cache" = "$h" ]; then skip=true; break; fi
    done
    if [ "$skip" = "false" ]; then
        CACHE_ARGS+=( --cache-dir "$WORKSPACE/caches/$cache" )
    fi
done
echo "Training caches: $(( ${#CACHE_ARGS[@]} / 2 ))"

echo "=== Launching training (Option A, $EPOCHS epochs) ==="
cd $WORKSPACE/mongoose
mkdir -p $WORKSPACE/checkpoints

WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY
    WANDB_ARGS=( --use-wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME" )
    echo "wandb enabled: project=$WANDB_PROJECT run=$WANDB_RUN_NAME"
fi

PYTHONPATH=src python -u scripts/train.py \
    "${CACHE_ARGS[@]}" \
    --epochs "$EPOCHS" --batch-size 32 --lr 3e-4 --min-lr 1.5e-5 \
    --use-l511 --use-t2d-hybrid \
    --probe-aware-velocity \
    --lambda-511 1.0 --lambda-smooth 0.001 --lambda-length 0.5 \
    --lambda-align 1.0 --align-min-confidence 0.7 \
    --warmstart-epochs 5 --warmstart-fade-epochs 2 \
    --min-blend 0.05 \
    --checkpoint-dir $WORKSPACE/checkpoints --save-every 1 \
    "${WANDB_ARGS[@]}" \
    2>&1 | tee $WORKSPACE/train.log

echo "=== Training complete. Checkpoints in $WORKSPACE/checkpoints ==="
ls -la $WORKSPACE/checkpoints
