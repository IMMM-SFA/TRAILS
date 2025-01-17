#!/bin/bash
#SBATCH -N 1 -p normal
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=find_regional_objs_sol140
#SBATCH --output=logs/find_regional_objs_sol140.out
#SBATCH --error=logs/find_regional_objs_sol140.err
#SBATCH --time=500:00:00
#SBATCH --mail-user=lbl59@cornell.edu
#SBATCH --mail-type=ALL

python ./find_regional_objs.py