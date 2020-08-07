#!/bin/bash
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem=600G
#SBATCH --time=3-00:00:00 
#SBATCH --job-name pp_mouse
#SBATCH --output pp_mouse_%J.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<ngravindra@gmail.com>

module load miniconda
source activate rnavel
python -u pp_mouse.py
