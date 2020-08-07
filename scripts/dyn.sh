#!/bin/bash
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=196G
#SBATCH --time=4-00:00:00 
#SBATCH --job-name dyn
#SBATCH --output dyn_v3_%J.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<ngravindra@gmail.com>

module load miniconda
source activate rnavel
python -u dynamical_genes.py
