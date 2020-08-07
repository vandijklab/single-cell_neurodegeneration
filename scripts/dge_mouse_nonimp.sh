#!/bin/bash
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00 
#SBATCH --job-name dgenoimpmm
#SBATCH --output dge_mousenonimp_%J.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<ngravindra@gmail.com>

module load miniconda
source activate rnavel
python -u dge_mouse_nonimp.py
