#!/bin/bash
# train_kfold.sh — 训练 5-fold cross validation
# 用法: cd /home/hzcu/BullyDetection && bash train_kfold.sh
#
# 可选: 只训练某个 fold
#   bash train_kfold.sh 0       # 只训练 fold 0
#   bash train_kfold.sh 0 2     # 训练 fold 0, 1, 2

set -e

PYSKL_DIR="/home/hzcu/BullyDetection/pyskl"
LD_LIBRARY_PATH="/home/hzcu/miniconda3/pkgs/cuda-cudart-11.8.89-0/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH

# 确定要训练哪些 fold
if [ $# -eq 0 ]; then
    FOLDS="0 1 2 3 4"
elif [ $# -eq 1 ]; then
    FOLDS="$1"
else
    # 范围: start end (inclusive)
    FOLDS=$(seq $1 $2)
fi

echo "=========================================="
echo "5-Fold Cross Validation Training"
echo "Folds to train: $FOLDS"
echo "=========================================="

for fold in $FOLDS; do
    CONFIG="${PYSKL_DIR}/configs/posec3d/finetune_campus_fold${fold}.py"

    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: Config not found: $CONFIG"
        exit 1
    fi

    echo ""
    echo "=========================================="
    echo "Training FOLD ${fold} ..."
    echo "Config: $CONFIG"
    echo "Start time: $(date)"
    echo "=========================================="

    cd "$PYSKL_DIR" && \
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 \
        tools/train.py "$CONFIG" --launcher pytorch

    echo "Fold ${fold} finished at: $(date)"
done

echo ""
echo "=========================================="
echo "All folds done!"
echo "=========================================="
