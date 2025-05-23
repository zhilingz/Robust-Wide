#!/bin/bash
#SBATCH -p gpu3       # 或者你出问题的分区
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 1G
#SBATCH --gres gpu:1
#SBATCH -o log/%j.out
#SBATCH -e log/%j.out
#SBATCH -t 00:05:00   # 短时间

echo "Hello from Slurm job ${SLURM_JOB_ID}"
pwd
ls -la
echo "Test job finished."