#!/bin/bash

set -e

start=1000
end=1000
increment=1000

upload_base_dir=/home/kazuki/converted_checkpoints/Mixtral-8x7b-load-balance

# for ループで指定された範囲と増分を使用
for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python scripts/kotoba/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Mixtral-NVE-code-math-load-balancing-lr_2e-5-min_lr_2e-6-iter$(printf "%07d" $i)
done
