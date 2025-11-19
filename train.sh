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

export CUDA_VISIBLE_DEVICES=1

# ====== Seeds to try ======
seeds=(42 77 123 999 2025)
EPOCHS=50

BLR=0.001

for seed in "${seeds[@]}"; do
  echo "======================================="
  echo " Running seed: ${seed}"
  echo "======================================="

  # Give each seed its own task name so logs/models don't overwrite
  timestamp=$(date +"%Y%m%d_%H%M%S")
  task="${MODEL_ARCH}_${DATASET}_${ADAPTATION}_s${seed}_${timestamp}"
  out_dir="./output_dir/"
  log_dir="./output_logs"

  python main_finetune.py \
    --model "${MODEL}" \
    --model_arch "${MODEL_ARCH}" \
    --finetune "${FINETUNE}" \
    --global_pool \
    --batch_size 16 \
    --epochs "${EPOCHS}" \
    --blr "${BLR}" \
    --weight_decay 0.05 \
    --nb_classes "${NUM_CLASS}" \
    --data_path "${data_path}" \
    --input_size 224 \
    --task "${task}" \
    --adaptation "${ADAPTATION}" \
    --partial_unfreeze_norm \
    --unfreeze_last_n_blocks 1 \
    --mixup 0.4 \
    --cutmix 0.0 \
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