#!/bin/bash
#SBATCH --job-name=NLP_Code_Execution       # Job name
#SBATCH --output=output_%j.txt       # Output file (%j will be replaced with the job ID)
#SBATCH --error=error_%j.txt         # Error file (%j will be replaced with the job ID)
#SBATCH --time=0-8:0                 # Time limit (DD-HH:MM)
#SBATCH --partition=dgx        # Partition to submit to. `teaching` (for the T4 GPUs) is default on Rosie, but it's still being specified here
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G

container="/data/containers/msoe-pytorch-24.05-py3.sif"
command="python seq2seq.py"
# --data /data/csc4801/Fish2 --batch_size 32 --epochs 25 --main_dir /home/ad.msoe.edu/waltzc/DSP_FISH2 --augment_data true --fine_tune true"

# Execute singularity container on node.
# singularity exec --nv -B /data:/data ${container} /usr/local/bin/nvidia_entrypoint.sh ${command}
singularity exec --nv -B /data:/data ${container} ${command}