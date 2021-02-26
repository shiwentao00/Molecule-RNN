#!/bin/bash
#PBS -q v100
#PBS -l nodes=1:ppn=36
#PBS -l walltime=72:00:00
#PBS -A hpc_michal01
#PBS -j oe

cd /work/derick/drug-gen-project/DeepDruGen/

singularity exec --nv -B /work,/project,/usr/lib64 /work/derick/singularities/pytorch-and-others.simg python train.py