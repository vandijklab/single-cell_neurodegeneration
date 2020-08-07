#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem=121G
#SBATCH --time=3-00:00:00 
#SBATCH --job-name pp_hum
#SBATCH --output pp_human_%J.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<ngravindra@gmail.com>

module load miniconda
source activate rnavel
python -u pp_human.py
