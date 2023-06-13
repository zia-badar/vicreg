#!/bin/bash
# #SBATCH --account=rsim
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=5
# #SBATCH --mem-per-cpu=12G
#SBATCH --output=output_enc_16.log
#SBATCH --time=100:00:00
#SBATCH --partition=gpu
# #SBATCH --gres=gpu:a100:2
# #SBATCH --gres=gpu:a100:1
# #SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:tesla:1
# #SBATCH --mem=256G
#SBATCH --mail-user==zia.badar@campus.tu-berlin.de
python main_vicreg.py --encodingdim 16 --batch_size 384 --rotation_pred False
