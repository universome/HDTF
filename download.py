"""
This file downloads almost all the videos from the HDTF dataset. Some videos are discarded for the following reasons:
- they do not contain cropping information because they are somewhat noisy (hand moving, background changing, etc.)
- they are not available on youtube anymore (at all or in the specified format)

The discarded videos constitute a small portion of the dataset, so you can try to re-download them manually on your own.

Usage:
```
$ python download.py --output_dir /tmp/data/hdtf --num_workers 8
```

You need tqdm, yt_dlp, and colorama libraries to be installed for this script to work.
"""

import os
import argparse
import subprocess
import pprint
from typing import List, Dict
from multiprocessing import Pool
from urllib import parse

import yt_dlp
from tqdm import tqdm
from colorama import init as cinit
from colorama import Fore

subsets = ["RD", "WDA", "WRA"]

cinit(autoreset=True)

def download_hdtf(source_dir: os.PathLike, output_dir: os.PathLike, num_workers: int, **process_video_kwargs):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, '_videos_raw'), exist_ok=True)

    download_queue = construct_download_queue(source_dir, output_dir)
    task_kwargs = [dict(
        video_data=vd,
        output_dir=output_dir,
        **process_video_kwargs,
     ) for vd in download_queue]
    pool = Pool(processes=num_workers)
    tqdm_kwargs = dict(total=len(task_kwargs), desc=f'Downloading videos into {output_dir}')

    for _ in tqdm(pool.imap_unordered(task_proxy, task_kwargs), **tqdm_kwargs):
        pass
    pool.close()
    pool.join()

    print(Fore.GREEN+'Download is finished, you can now (optionally) delete the following directories, since they are not needed anymore and occupy a lot of space:')
    print(Fore.GREEN+' - '+os.path.join(output_dir, '_videos_raw'))


def construct_download_queue(source_dir: os.PathLike, output_dir: os.PathLike) -> List[Dict]:
    download_queue = []

    for subset in subsets:
        video_urls = read_file_as_space_separated_data(os.path.join(source_dir, f'{subset}_video_url.txt'))
        crops = read_file_as_space_separated_data(os.path.join(source_dir, f'{subset}_crop_wh.txt'))
        intervals = read_file_as_space_separated_data(os.path.join(source_dir, f'{subset}_annotion_time.txt'))
        resolutions = read_file_as_space_separated_data(os.path.join(source_dir, f'{subset}_resolution.txt'))

        for video_name, (video_url,) in video_urls.items():
            if not f'{video_name}.mp4' in intervals:
                print(f'{Fore.RED}Clip {subset}/{video_name} does not contain any clip intervals. It will be discarded.')
                continue

            if not f'{video_name}.mp4' in resolutions or len(resolutions[f'{video_name}.mp4']) > 1:
                print(f'{Fore.RED}Clip {subset}/{video_name} does not contain an appropriate resolution (or it is in a bad format). It will be discarded.')
                continue

            all_clips_intervals = [x.split('-') for x in intervals[f'{video_name}.mp4']]
            clips_crops = []
            clips_intervals = []
            crops_keys=', '.join(crops.keys())

            for clip_idx, clip_interval in enumerate(all_clips_intervals):
                clip_name = f'{video_name}_{clip_idx}.mp4'
                if not clip_name in crops:
                    print(f'{Fore.RED}Discarding Clip: {subset}/{clip_name}. Clip is not present in crops.')
                    continue
                else:
                    print(f'{Fore.GREEN}Appending Clip:  {subset}/{clip_name}')
                clips_crops.append(crops[clip_name])
                clips_intervals.append(clip_interval)

            clips_crops = [list(map(int, cs)) for cs in clips_crops]

            if len(clips_crops) == 0:
                print(f'{Fore.RED}Discarding {subset}/{video_name}. No cropped versions found.')
                continue

            assert len(clips_intervals) == len(clips_crops)
            assert set([len(vi) for vi in clips_intervals]) == {2}, f"Broken time interval, {clips_intervals}"
            assert set([len(vc) for vc in clips_crops]) == {4}, f"Broken crops, {clips_crops}"
            assert all([vc[1] == vc[3] for vc in clips_crops]), f'Some crops are not square, {clips_crops}'

            download_queue.append({
                'name': f'{subset}_{video_name}',
                'id': parse.parse_qs(parse.urlparse(video_url).query)['v'][0],
                'intervals': clips_intervals,
                'crops': clips_crops,
                'output_dir': output_dir,
                'resolution': resolutions[f'{video_name}.mp4'][0]
            })

    return download_queue


def task_proxy(kwargs):
    return download_and_process_video(**kwargs)


def download_and_process_video(video_data: Dict, output_dir: str):
    """
    Downloads the video and cuts/crops it into several ones according to the provided time intervals
    """
    raw_download_path = os.path.join(output_dir, '_videos_raw', f"{video_data['name']}.mp4")
    raw_download_log_file = os.path.join(output_dir, '_videos_raw', f"{video_data['name']}_download_log.txt")
    print(f"{Fore.LIGHTBLUE_EX} raw_download_path: {raw_download_path}")
    
    download_result = download_video(video_data['id'], raw_download_path, log_file=raw_download_log_file)

    if not download_result:
        print(f'{Fore.RED} Failed to download {video_data["name"]}')
        print(f'{Fore.RED} See {raw_download_log_file} for details')
        return

    for clip_idx in range(len(video_data['intervals'])):
        start, end = video_data['intervals'][clip_idx]
        clip_name = f'{video_data["name"]}_{clip_idx:03d}'
        clip_path = os.path.join(output_dir, clip_name + '.mp4')
        crop_success = cut_and_crop_video(raw_download_path, clip_path, start, end, video_data['crops'][clip_idx])

        if not crop_success:
            print(f'{Fore.RED} Failed to cut-and-crop clip #{clip_idx}', video_data)
            pprint.pprint(video_data, indent=4, sort_dicts=False)
            continue


def read_file_as_space_separated_data(filepath: os.PathLike) -> Dict:
    """
    Reads a file as a space-separated dataframe, where the first column is the index
    """
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
        lines = [[v.strip() for v in l.strip().split(' ')] for l in lines]
        data = {l[0]: l[1:] for l in lines}

    return data

def download_video(video_id, download_path, resolution: int = None, video_format="bestvideo+bestaudio", log_file=None):
    """
    Download video from YouTube.
    :param video_id:        YouTube ID of the video.
    :param download_path:   Where to save the video.
    :param resolution:      Desired resolution (not currently used in yt-dlp config).
    :param video_format:    Format to download (default is best video and audio).
    :param log_file:        Path to a log file for yt-dlp.
    :return:                Tuple: path to the downloaded video and a bool indicating success.
    """
    
    class Logger:
        """
        A simple logger for yt-dlp to write debug, warning, and error messages to a specified log file.
        
        Attributes:
            log_path (str): Path to the log file where messages will be written.
        """
        
        def __init__(self, log_path):
            """
            Initializes the Logger with a log file path.
            
            :param log_path: Path to the file where log messages should be saved.
            """
            self.log_path = log_path

        def debug(self, msg):
            """
            Logs a debug message.
            
            :param msg: The debug message to log.
            """
            with open(self.log_path, "a") as f:
                f.write(f"DEBUG: {msg}\n")

        def warning(self, msg):
            """
            Logs a warning message.
            
            :param msg: The warning message to log.
            """
            with open(self.log_path, "a") as f:
                f.write(f"WARNING: {msg}\n")

        def error(self, msg):
            """
            Logs an error message.
            
            :param msg: The error message to log.
            """
            with open(self.log_path, "a") as f:
                f.write(f"ERROR: {msg}\n")

    # Define yt-dlp options
    ydl_opts = {
        'format': video_format,             # Set video format to best video and audio by default
        'outtmpl': download_path,           # Output path template
        'quiet': True,                      # Suppress verbose output
        'merge_output_format': 'mp4',       # Ensure output format is MP4
    }
    
    # If a log file is specified, configure the logger
    if log_file:
        ydl_opts['logger'] = Logger(log_file)
    
    # Download the video using yt-dlp
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
        success = True
    except Exception as e:
        success = False
        if log_file:
            with open(log_file, "a") as f:
                f.write(f"ERROR: Failed to download {video_id}. Exception: {str(e)}\n")
    
    result = success and os.path.isfile(download_path)
    return download_path, result

def cut_and_crop_video(raw_video_path, output_path, start, end, crop: List[int]):
    # if os.path.isfile(output_path): return True # File already exists

    x, out_w, y, out_h = crop

    command = [
        "ffmpeg", "-i", raw_video_path,
        "-strict", "-2", # Some legacy arguments
        "-loglevel", "quiet", # Verbosity arguments
        "-qscale", "0", # Preserve the quality
        "-y", # Overwrite if the file exists
        "-ss", str(start), 
        "-to", str(end),
        "-filter:v", f'"crop={out_w}:{out_h}:{x}:{y}"', # Crop arguments
        output_path
    ]
    return_code = subprocess.call(command)
    success = return_code == 0

    if not success:
        print(f'{Fore.RED} Command failed: {" ".join(command)}')

    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HDTF dataset")
    parser.add_argument('-s', '--source_dir', type=str, default='HDTF_dataset', help='Path to the directory with the dataset description')
    parser.add_argument('-o', '--output_dir', type=str, default='dataset', help='Where to save the videos?')
    parser.add_argument('-w', '--num_workers', type=int, default=1, help='Number of workers for downloading.')
    args = parser.parse_args()

    download_hdtf(
        args.source_dir,
        args.output_dir,
        args.num_workers,
    )
