# ==== Model settings ====
# adaptation {finetune,lp}
ADAPTATION="partial"          # or "lp" for linear probe
MODEL="RETFound_mae"
MODEL_ARCH="retfound_mae"
FINETUNE="RETFound_mae_natureCFP"

# ==== Data settings ====
DATASET="CKDRET"
NUM_CLASS=2
data_path="/data/rishabhbhattad/data/CKD_Study/RETFound_data/CKDRET"

export CUDA_VISIBLE_DEVICES=0

# ====== Seeds to try ======
seeds=(42 77 123 999 2025)

# ====== Choose LR based on adaptation ======
if [ "$ADAPTATION" = "lp" ]; then
  BLR=0.001    # base learning rate for LP
else
  BLR=0.0001   # base learning rate for partial/finetune
fi

for seed in "${seeds[@]}"; do
  echo "======================================="
  echo " Running seed: ${seed}"
  echo "======================================="

  # Give each seed its own task name so logs/models don't overwrite
  timestamp=$(date +"%Y%m%d_%H%M%S")
  task="${MODEL_ARCH}_${DATASET}_${ADAPTATION}_s${seed}_${timestamp}"
  out_dir="./output_dir/${task}"
  log_dir="./output_logs/${task}"

  python main_finetune.py \
    --model "${MODEL}" \
    --model_arch "${MODEL_ARCH}" \
    --finetune "${FINETUNE}" \
    --global_pool \
    --batch_size 16 \
    --blr "${BLR}" \
    --world_size 1 \
    --epochs 20 \
    --nb_classes "${NUM_CLASS}" \
    --data_path "${data_path}" \
    --input_size 224 \
    --task "${task}" \
    --adaptation "${ADAPTATION}" \
    --device cuda \
    --seed "${seed}" \
    --datasets_seed "${seed}" \
    --stratified \
    --output_dir "${out_dir}" \
    --log_dir "${log_dir}"

  echo
  echo "Finished seed ${seed}"
  echo
done