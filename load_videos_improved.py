import warnings
from argparse import ArgumentParser
from itertools import cycle
from multiprocessing import Pool
from pathlib import Path

import imageio
import pandas as pd
from tqdm import tqdm
from util import rsync, download, crop_video

warnings.filterwarnings("ignore")


def load_videos(run_args):
    """Run the data pipeline for a single video"""
    video_id, args = run_args

    video_folder = Path(args.video_folder)
    video_file = video_folder / (video_id.split("#")[0] + ".mp4")

    if not video_file.exists():
        download(video_file, video_id, args.youtube)

    if not video_file.exists():
        print("Can not load video %s, broken link" % video_id.split("#")[0])
        return

    reader = imageio.get_reader(video_file)
    height, width = reader.get_data(0).shape[:2]

    for _, chunk in pd.read_csv(args.metadata).iterrows():
        if chunk.video_id != video_id:
            continue

        left, top, right, bot = map(int, chunk.bbox.split("-"))

        # Compare to the actual video and resize the bounding box and time stamps
        left = int(left / (chunk.width / width))
        top = int(top / (chunk.height / height))
        right = int(right / (chunk.width / width))
        bot = int(bot / (chunk.height / height))
        start_time = chunk.start / chunk.fps
        end_time = chunk.end / chunk.fps

        # Ensure that the width and height are the same; scale up the smaller
        # dimension if necessary
        if right - left > bot - top:
            top = max(0, top - (right - left - (bot - top)) // 2)
            bot = min(height, bot + (right - left - (bot - top)) // 2)
        elif right - left < bot - top:
            left = max(0, left - (bot - top - (right - left)) // 2)
            right = min(width, right + (bot - top - (right - left)) // 2)

        fname = (
            chunk.person_id
            + "#"
            + video_id
            + "#"
            + str(chunk.start).zfill(6)
            + "#"
            + str(chunk.end).zfill(6)
            + ".mp4"
        )
        fp = Path(args.out_folder) / partition / fname
        crop_video(video_file, start_time, end_time, left, top, right, bot, fp)

    # Sync the data to GCS after processing each video
    rsync()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--video_folder", default="youtube-taichi", help="Path to youtube videos"
    )
    parser.add_argument(
        "--metadata", default="taichi-metadata-new.csv", help="Path to metadata"
    )
    parser.add_argument("--out_folder", default="taichi-png", help="Path to output")
    parser.add_argument("--format", default=".png", help="Storing format")
    parser.add_argument("--workers", default=1, type=int, help="Number of workers")
    parser.add_argument("--youtube", default="./youtube-dl", help="Path to youtube-dl")

    parser.add_argument(
        "--image_shape",
        default=(256, 256),
        type=lambda x: tuple(map(int, x.split(","))),
        help="Image shape, None for no resize",
    )

    args = parser.parse_args()
    video_folder = Path(args.video_folder)
    out_folder = Path(args.out_folder)
    if not video_folder.exists():
        video_folder.mkdir(parents=True)
    if not out_folder.exists():
        out_folder.mkdir(parents=True)
    for partition in ["test", "train"]:
        partition_folder = out_folder / partition
        if not partition_folder.exists():
            partition_folder.mkdir()

    df = pd.read_csv(args.metadata)
    video_ids = set(df["video_id"])
    with Pool(processes=args.workers) as pool:
        args_list = cycle([args])
        for _ in tqdm(pool.imap_unordered(load_videos, zip(video_ids, args_list))):
            None
