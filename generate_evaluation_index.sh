#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=2

# 运行 Python 脚本
python -m src.scripts.generate_evaluation_index \
    data_loader.train.num_workers=0 \
    data_loader.test.num_workers=0 \
    data_loader.val.num_workers=0 \
    index_generator.output_path=assets \
    dataset.roots=[datasets/re10k] \
    dataset.image_shape=[256,256]
