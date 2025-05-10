#!/bin/bash

#SBATCH -J ikgn_myver    #Slurm job name

#SBATCH -t 12:00:00                                #Maximum runtime of 48 hours

#SBATCH --mail-user=srongaa@connect.ust.hk #Update your email address
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH -p normal 


#SBATCH --nodes=1                # node count
#SBATCH --gpus=1                 # number of GPUs per node(only valid under large/normal partition)
#SBATCH --account=mscbdt2024

#SBATCH -o logs/out%j.log
#SBATCH -e logs/err%j.log

module purge
module load Anaconda3/2023.09-0

# make sure you install the conda environment in your home directory
source ~/.bashrc

# modify this line to your path of code.
cd $HOME//5002/myIKGN
python main_my_ver.py

echo "Python path: $(which python)"
python -c "import torch; print(torch.__version__)"
