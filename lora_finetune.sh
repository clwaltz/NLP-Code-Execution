#!/bin/bash
#SBATCH --job-name=NLP_LoRA_Finetune
#SBATCH --output=output/output_lora_%j.txt
#SBATCH --error=error/error_lora_%j.txt
#SBATCH --time=0-12:00
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G

set -euo pipefail

container="/data/containers/msoe-pytorch-24.05-py3.sif"

# Change these defaults as needed.
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
DATA_DIR="${DATA_DIR:-data_collection_2}"
OUTPUT_DIR="${OUTPUT_DIR:-data_collection_2/lora_runs}"
VENV_DIR="${VENV_DIR:-.venv}"
USE_4BIT="${USE_4BIT:-0}"

command="python -u lora_finetune.py \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --model_name ${MODEL_NAME}"

if [ "${USE_4BIT}" = "1" ]; then
  command="${command} --load_in_4bit"
fi

# Use an isolated environment to avoid ~/.local package conflicts.
# The previous failure came from incompatible peft/transformers versions in user site-packages.
deps_cmd="rm -rf ${VENV_DIR} \
  && python -m venv --system-site-packages ${VENV_DIR} \
  && source ${VENV_DIR}/bin/activate \
  && export PYTHONNOUSERSITE=1 \
  && echo '[setup] upgrading pip' \
  && python -m pip install --upgrade pip \
  && echo '[setup] installing base LoRA deps' \
  && python -m pip install --upgrade-strategy only-if-needed \
    'torch<2.5' \
    'transformers>=4.41,<4.48' \
    'peft>=0.11,<0.14' \
    'datasets>=2.19,<3.0' \
    'accelerate>=0.30,<1.0' \
    'huggingface-hub>=0.24,<1.0' \
    sentencepiece"

if [ "${USE_4BIT}" = "1" ]; then
  deps_cmd="${deps_cmd} && echo '[setup] installing bitsandbytes for 4-bit mode' && python -m pip install 'bitsandbytes>=0.43,<0.47'"
fi

run_cmd="source ${VENV_DIR}/bin/activate \
  && export PYTHONNOUSERSITE=1 \
  && export PYTHONUNBUFFERED=1 \
  && echo '[run] nvidia-smi:' \
  && nvidia-smi || true \
  && echo '[run] torch/cuda quick check:' \
  && python -c \"import torch; print('torch', torch.__version__); print('torch cuda build', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('device count', torch.cuda.device_count())\" \
  && echo '[run] starting lora_finetune.py' \
  && ${command}"

singularity exec --nv -B /data:/data "${container}" bash -lc "${deps_cmd} && ${run_cmd}"
