#!/bin/bash -l 

#SBATCH --job-name=j_kernel
#SBATCH --output=logs/jupyter/jupyter_kernel/%x.out 
#SBATCH --error=logs/jupyter/jupyter_kernel/%x.err 
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4 
#SBATCH --gres=gpu:1
#SBATCH --mem=64G 
#SBATCH --time=03:00:00
#SBATCH --partition=batch 

########################################################
# IBEX SLURM SCRIPT TO INSTANTIATE A JUPYTER-NOTEBOOK  #
# FOR 1H WITH SINGLE GPU. CHECK STDOUT FILE FOR        #
# INSTRUCTIONS ON HOW TO CONNECT TO THE NOTEBOOK FROM  #
# LOCAL WEB BROWSER                                    #
########################################################

# Load conda environment which has Jupyter installed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scalinglaws

# Get tunneling info
export XDG_RUNTIME_DIR="" node=$(hostname -s) 
user=$(whoami) 
submit_host=${SLURM_SUBMIT_HOST} 
port=10000
echo $node pinned to port $port 

# Print tunneling instructions
echo -e " 
To connect to the compute node ${node} on IBEX running your jupyter notebook server, you need to run following two commands in a terminal 1. 
Command to create ssh tunnel from you workstation/laptop to glogin: 

ssh -L ${port}:${node}:${port} ${user}@glogin.ibex.kaust.edu.sa 

Copy the link provided below by jupyter-server and replace the NODENAME with localhost before pasting it in your browser on your workstation/laptop " 

# Run Jupyter 
jupyter notebook --no-browser --port=${port} --port-retries=50 --ip=${node} 