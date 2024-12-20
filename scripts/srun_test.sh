#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))

set -x

PARTITION="vip"
JOB_NAME="VDEB"
CONFIG=$1
NODE=$2
# WORK_DIR=$2
GPUS=${GPUS:-1}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} --job-name=${JOB_NAME} --nodelist=${NODE} --gres=gpu:${GPUS} --ntasks=${GPUS}  --ntasks-per-node=${GPUS} --cpus-per-task=6  --kill-on-bad-exit=1 \
python -u basicsr/test.py -opt ${CONFIG} --launcher="slurm"

