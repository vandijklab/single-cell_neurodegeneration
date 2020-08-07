#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=121G
#SBATCH --time=3-00:00:00 
#SBATCH --job-name pp_mouse
#SBATCH --output pp_mouse2_%J.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<ngravindra@gmail.com>

module load miniconda
source activate rnavel
python -u pp_mouse_200614.py
