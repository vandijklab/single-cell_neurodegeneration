#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=121G
#SBATCH --time=3-00:00:00 
#SBATCH --job-name dge_human
#SBATCH --output dge_humnonimp_%J.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<ngravindra@gmail.com>

module load miniconda
source activate rnavel
python -u dge_human_nonimp.py
