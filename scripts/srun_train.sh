#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))

set -x

PARTITION="vip"
JOB_NAME="VDEB"
CONFIG=$1
# NODE=$2
# WORK_DIR=$2 --nodelist=${NODE}
GPUS=${GPUS:-8}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} --job-name=${JOB_NAME} --gres=gpu:${GPUS} --ntasks=${GPUS}  --ntasks-per-node=${GPUS} --cpus-per-task=2  --kill-on-bad-exit=1 \
python -u basicsr/train.py -opt ${CONFIG} --launcher="slurm"
