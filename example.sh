#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem-per-cpu=8g
#SBATCH -p gpu
#SBATCH --time=20:00:00

module load GCC
lscpu
nvidia-smi
/home/kkxw544/miniconda3/envs/deepfrag_1/bin/python /home/kkxw544/deepfrag/free_test.py --pdbs /projects/mai/users/kkxw544_magdalena/deepfrag_data/protein_ligand_complexes --spec_model xxx --res_name ftest_xxx
