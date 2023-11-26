#!/bin/env bash

set -eo pipefail

python load_videos_improved.py \
    --metadata vox-metadata.csv \
    --format .mp4 \
    --out_folder ./data \
    --workers $(python -c 'import multiprocessing as m; print(m.cpu_count() - 1)') \
    --youtube 'yt-dlp'