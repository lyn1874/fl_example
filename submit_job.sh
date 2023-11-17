#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J cifar10
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 2:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=24GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u email
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o cifar_%J.out
#BSUB -e cifar_%J.err
# -- end of LSF options --

nvidia-smi
module load scipy/1.10.1-python-3.9.17 matplotlib/3.7.1-numpy-1.24.3-python-3.9.17
source ../torch_dl/bin/activate

./run_cifar.sh
