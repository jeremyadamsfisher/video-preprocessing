#!/bin/env bash

set -e -x -o pipefail

pip install yl-dlp

python load_videos.py \
    --metadata ./ted-metadata.csv \
    --format .mp4 \
    --out_folder ./data/TED384-v2 \
    --workers $(python -c 'import multiprocessing as m; print(m.cpu_count() - 1)') \
    --image_shape 384,384 \
    --youtube 'yt-dlp'

gsutil -m cp -r . gs://ted-dataset