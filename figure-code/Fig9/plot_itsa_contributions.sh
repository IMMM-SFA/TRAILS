#!/bin/bash
#SBATCH -N 1 -p normal
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=gen_itsa_fig_s140_287
#SBATCH --output=logs/gen_itsa_fig_s140_287.out
#SBATCH --error=logs/gen_itsa_fig_s140_287.err
#SBATCH --time=5:00:00

python ./rof_exceedances_analysis_oneutil.py