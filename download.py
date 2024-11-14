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
    """
    Downloads and processes videos from the HDTF dataset in parallel using multiprocessing.

    The function manages the download process by:
    - Creating the necessary output directories.
    - Constructing a download queue from files in the specified source directory.
    - Using a multiprocessing pool to handle downloads and subsequent processing.
    - Providing progress tracking with tqdm.

    After completing the download, a message is displayed with optional cleanup instructions to delete
    temporary raw video files to save space.

    Args:
        source_dir (os.PathLike): The directory containing HDTF metadata files, including video URLs,
                                  crop data, time intervals, and resolution information for each video subset.
        output_dir (os.PathLike): The directory where downloaded videos and processed files will be saved.
        num_workers (int): The number of parallel worker processes to use for downloading.
        **process_video_kwargs: Additional keyword arguments passed to `download_and_process_video`, 
                                allowing custom settings for processing each video.

    Workflow:
        1. Creates the primary output directory and a subdirectory `_videos_raw` for raw downloads.
        2. Calls `construct_download_queue` to prepare a list of video download tasks based on the metadata
           available in `source_dir`. Each entry in the queue includes details needed for downloading and processing.
        3. Uses a multiprocessing `Pool` to execute `download_and_process_video` for each video in `download_queue`,
           with progress displayed via tqdm.
        4. After completing downloads, provides a message about optional cleanup for temporary video files.

    Returns:
        None

    Raises:
        AssertionError: If certain data inconsistencies are detected during download queue construction, such as
                        missing or malformed intervals, crops, or resolution information.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, '_videos_raw'), exist_ok=True)

    download_queue = construct_download_queue(source_dir, output_dir)
    task_kwargs = [dict(
        video_data=vd,
        output_dir=output_dir,
        **process_video_kwargs,
    ) for vd in download_queue]
    pool = Pool(processes=num_workers)
    tqdm_kwargs = dict(total=len(task_kwargs),
                       desc=f'Downloading videos into {output_dir}')

    for _ in tqdm(pool.imap_unordered(task_proxy, task_kwargs), **tqdm_kwargs):
        pass
    pool.close()
    pool.join()

    print(Fore.GREEN+'Download is finished, you can now (optionally) delete the following directories, since they are not needed anymore and occupy a lot of space:')
    print(Fore.GREEN+' - '+os.path.join(output_dir, '_videos_raw'))


def construct_download_queue(source_dir: os.PathLike, output_dir: os.PathLike) -> List[Dict]:
    """
    Constructs a queue of videos to be downloaded and processed based on metadata from the HDTF dataset.
    
    This function reads metadata files for each subset in the HDTF dataset, which provide information on:
    - Video URLs.
    - Time intervals indicating segments to be extracted from each video.
    - Crop coordinates defining the regions of interest.
    - Resolution information for each video.
    
    For each valid video file, an entry is created in the download queue with detailed information required
    for downloading, cropping, and segmenting.

    Args:
        source_dir (os.PathLike): Path to the directory containing metadata files (`*_video_url.txt`,
                                   `*_crop_wh.txt`, `*_annotion_time.txt`, and `*_resolution.txt`) for each subset.
        output_dir (os.PathLike): Path to the directory where the downloaded and processed videos will be stored.
    
    Returns:
        List[Dict]: A list of dictionaries, each representing a video to download and process. Each dictionary
                    contains the following keys:
                    - 'name': Combined subset and video name identifier.
                    - 'id': YouTube video ID extracted from the video URL.
                    - 'intervals': List of start and end times for each clip segment.
                    - 'crops': List of crop coordinates for each segment.
                    - 'output_dir': The output directory path for this video.
                    - 'resolution': Desired resolution for the video.

    Workflow:
        1. Reads metadata files for each subset (e.g., "RD", "WDA", "WRA") to gather video URLs, time intervals, crops,
           and resolution information.
        2. For each video:
           - Ensures it has valid time intervals and resolution data.
           - Verifies that all segments have corresponding crop information.
           - Discards videos missing required metadata, and prints warnings about invalid or missing data.
        3. Creates a download queue entry for each valid video with the required download and processing data.
    
    Raises:
        AssertionError: If the video segment data is inconsistent, such as:
                        - Missing or malformed time intervals.
                        - Incomplete or non-square crop data.
                        These assertions ensure that only well-formed entries are added to the download queue.
    
    Example:
        >>> construct_download_queue("HDTF_dataset", "/tmp/data/hdtf")
        [{'name': 'RD_sample_video', 'id': 'abc123', 'intervals': [[0, 10], [15, 25]],
          'crops': [[0, 128, 0, 128], [0, 128, 0, 128]], 'output_dir': '/tmp/data/hdtf', 'resolution': '720p'}]
    """
    download_queue = []

    for subset in subsets:
        video_urls = read_file_as_space_separated_data(
            os.path.join(source_dir, f'{subset}_video_url.txt'))
        crops = read_file_as_space_separated_data(
            os.path.join(source_dir, f'{subset}_crop_wh.txt'))
        intervals = read_file_as_space_separated_data(
            os.path.join(source_dir, f'{subset}_annotion_time.txt'))
        resolutions = read_file_as_space_separated_data(
            os.path.join(source_dir, f'{subset}_resolution.txt'))

        for video_name, (video_url,) in video_urls.items():
            if not f'{video_name}.mp4' in intervals:
                print(
                    f'{Fore.RED}Clip {subset}/{video_name} does not contain any clip intervals. It will be discarded.')
                continue

            if not f'{video_name}.mp4' in resolutions or len(resolutions[f'{video_name}.mp4']) > 1:
                print(f'{Fore.RED}Clip {subset}/{video_name} does not contain an appropriate resolution (or it is in a bad format). It will be discarded.')
                continue

            all_clips_intervals = [x.split('-')
                                   for x in intervals[f'{video_name}.mp4']]
            clips_crops = []
            clips_intervals = []

            for clip_idx, clip_interval in enumerate(all_clips_intervals):
                clip_name = f'{video_name}_{clip_idx}.mp4'
                if not clip_name in crops:
                    print(
                        f'{Fore.RED}Discarding Clip: {subset}/{clip_name}. Clip is not present in crops.')
                    continue
                else:
                    print(f'{Fore.GREEN}Appending Clip:  {subset}/{clip_name}')
                clips_crops.append(crops[clip_name])
                clips_intervals.append(clip_interval)

            clips_crops = [list(map(int, cs)) for cs in clips_crops]

            if len(clips_crops) == 0:
                print(
                    f'{Fore.RED}Discarding {subset}/{video_name}. No cropped versions found.')
                continue

            assert len(clips_intervals) == len(clips_crops)
            assert set([len(vi) for vi in clips_intervals]) == {
                2}, f"Broken time interval, {clips_intervals}"
            assert set([len(vc) for vc in clips_crops]) == {
                4}, f"Broken crops, {clips_crops}"
            assert all([vc[1] == vc[3] for vc in clips_crops]
                       ), f'Some crops are not square, {clips_crops}'

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
    """
    A proxy function to execute `download_and_process_video` with unpacked keyword arguments.
    
    This function serves as a wrapper that allows passing a dictionary of arguments (`kwargs`) 
    to the `download_and_process_video` function. It is primarily used in conjunction with 
    multiprocessing, where it enables the `Pool.imap_unordered` method to handle the video 
    processing tasks in parallel.

    Args:
        kwargs (dict): A dictionary of arguments required by `download_and_process_video`.
                       This typically includes:
                       - 'video_data': A dictionary containing video details (ID, name, intervals, crops, etc.).
                       - 'output_dir': The directory path where processed clips will be saved.

    Returns:
        None

    Usage:
        The `task_proxy` function is designed for use with parallel processing. By passing a dictionary 
        of arguments instead of positional arguments, it enables compatibility with the multiprocessing 
        pool's mapping methods.

    Example:
        >>> task_kwargs = {'video_data': {...}, 'output_dir': '/path/to/output'}
        >>> task_proxy(task_kwargs)

    Notes:
        This function simplifies the interface for multiprocessing tasks, allowing 
        `download_and_process_video` to be used directly within the parallel processing workflow 
        without modifying its original function signature.
    """
    return download_and_process_video(**kwargs)



def download_and_process_video(video_data: Dict, output_dir: str):
    """
    Downloads a video from YouTube and processes it by segmenting and cropping based on provided intervals and crop data.

    The function performs the following steps:
    1. Downloads the specified video to a raw file path within the `_videos_raw` subdirectory of `output_dir`.
    2. Iterates over the specified intervals and crop data to create individual video clips:
       - Each clip is extracted according to its specified time interval.
       - Each clip is cropped based on the coordinates provided in `video_data['crops']`.
    3. Saves each processed clip in `output_dir` with a unique name indicating the video and clip index.

    Args:
        video_data (dict): A dictionary containing metadata for the video to be downloaded and processed. 
                           Expected keys include:
                           - 'id': The YouTube ID of the video.
                           - 'name': A unique name identifier for the video.
                           - 'intervals': A list of time intervals (start, end) for each clip segment.
                           - 'crops': A list of crop coordinates (x, width, y, height) for each clip segment.
                           - 'resolution': The desired resolution of the video.
        output_dir (str): Path to the directory where processed video clips will be saved.

    Workflow:
        - Downloads the video using `download_video`, saving it as `{video_name}.mp4` in `_videos_raw`.
        - For each time interval in `video_data['intervals']`:
            - Extracts the segment and applies cropping according to the corresponding entry in `video_data['crops']`.
            - Saves each clip in `output_dir` with a file name formatted as `{video_name}_{clip_idx:03d}.mp4`.
        - Logs errors to the console if downloading or processing fails for a particular segment or crop.

    Returns:
        None

    Raises:
        ValueError: If the video cannot be downloaded or if any of the cropping or segmentation fails.
    
    Example:
        >>> video_data = {
                'id': 'abc123',
                'name': 'sample_video',
                'intervals': [[0, 10], [15, 25]],
                'crops': [[0, 128, 0, 128], [10, 118, 10, 118]],
                'output_dir': '/tmp/data/hdtf',
                'resolution': '720p'
            }
        >>> download_and_process_video(video_data, '/tmp/data/hdtf')

    Notes:
        - This function requires `ffmpeg` to be installed for segmenting and cropping video clips.
        - Detailed logging is provided to indicate the status of each clip's download and processing.
        - If `download_video` fails, an error message is printed to the console, and the function skips further processing.
    """
    raw_download_path = os.path.join(
        output_dir, '_videos_raw', f"{video_data['name']}.mp4")
    raw_download_log_file = os.path.join(
        output_dir, '_videos_raw', f"{video_data['name']}_download_log.txt")
    print(f"{Fore.LIGHTBLUE_EX} raw_download_path: {raw_download_path}")

    download_result = download_video(
        video_data['id'], raw_download_path, log_file=raw_download_log_file)

    if not download_result:
        print(f'{Fore.RED} Failed to download {video_data["name"]}')
        print(f'{Fore.RED} See {raw_download_log_file} for details')
        return

    for clip_idx in range(len(video_data['intervals'])):
        start, end = video_data['intervals'][clip_idx]
        clip_name = f'{video_data["name"]}_{clip_idx:03d}'
        clip_path = os.path.join(output_dir, clip_name + '.mp4')
        crop_success = cut_and_crop_video(
            raw_download_path, clip_path, start, end, video_data['crops'][clip_idx])

        if not crop_success:
            print(f'{Fore.RED} Failed to cut-and-crop clip #{clip_idx}')
            pprint.pprint(video_data, indent=4, sort_dicts=False)
            continue

def read_file_as_space_separated_data(filepath: os.PathLike) -> Dict:
    """
    Reads a space-separated file and returns its contents as a dictionary.
    
    This function reads a text file where each line contains space-separated values. 
    The first value in each line is treated as the key, and the remaining values are 
    stored as a list associated with that key. This is useful for parsing metadata 
    files with a consistent space-separated format.

    Args:
        filepath (os.PathLike): The path to the file to be read.

    Returns:
        Dict: A dictionary where each key corresponds to the first item in a line,
              and each value is a list of the remaining items in that line.

    Example:
        Suppose `example.txt` contains:
            video1 1280 720
            video2 640 480
        >>> read_file_as_space_separated_data("example.txt")
        {'video1': ['1280', '720'], 'video2': ['640', '480']}
    
    Notes:
        - Blank lines are not supported and may cause errors.
        - Each line must contain at least one space-separated value to be valid.

    Raises:
        IOError: If the file cannot be opened or read.
    """
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
        lines = [[v.strip() for v in l.strip().split(' ')] for l in lines]
        data = {l[0]: l[1:] for l in lines}

    return data


def download_video(video_id, download_path, video_format="bestvideo+bestaudio", log_file=None):
    """
    Downloads a YouTube video in the specified format and saves it to a given path.

    This function uses `yt-dlp` to download a video by its YouTube ID, selecting the highest
    available quality by default. It provides options for specifying a custom format or resolution
    and can log download progress and errors to a specified log file.

    Args:
        video_id (str): The YouTube ID of the video to download.
        download_path (str): The full path (including file name) where the downloaded video will be saved.
        video_format (str, optional): The video and audio format selection for yt-dlp. Defaults to
                                      "bestvideo+bestaudio" for highest available quality.
        log_file (str, optional): Path to a file where log messages (debug, warnings, and errors) 
                                  will be recorded. If None, logging to a file is disabled.

    Returns:
        Tuple[str, bool]: A tuple where:
                          - The first element is the path to the downloaded video file.
                          - The second element is a boolean indicating success (True if the file
                            was downloaded successfully, False otherwise).

    Workflow:
        1. Constructs `yt-dlp` options based on the provided arguments, including `format`, `outtmpl`, 
           and `logger` if a log file is specified.
        2. Attempts to download the video. If successful, verifies the file exists at `download_path`.
        3. Logs errors if the download fails and saves them to `log_file` if specified.

    Raises:
        Exception: Any exceptions during the download are logged if `log_file` is provided, and the 
                   function will return False for success.

    Example:
        >>> download_video("abc123", "/path/to/video.mp4", log_file="/path/to/log.txt")
        ("/path/to/video.mp4", True)

    Notes:
        - Requires `yt-dlp` to be installed.
        - Requires `ffmpeg` if merging video and audio streams is necessary.
        - Custom logging is provided through a nested `Logger` class if `log_file` is specified.
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
        # Set video format to best video and audio by default
        'format': video_format,
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
                f.write(
                    f"ERROR: Failed to download {video_id}. Exception: {str(e)}\n")

    result = success and os.path.isfile(download_path)
    return download_path, result

def cut_and_crop_video(raw_video_path, output_path, start, end, crop: List[int]):
    """
    Cuts and crops a video segment from a specified start to end time and saves it to the output path.

    This function uses `ffmpeg` to:
    1. Extract a segment of the video from `start` to `end` time.
    2. Apply a crop filter to the segment based on the provided crop coordinates.
    3. Save the processed clip to `output_path` with the original quality preserved.

    Args:
        raw_video_path (str): Path to the source video file to be processed.
        output_path (str): Path where the processed video clip will be saved, including the file name.
        start (float or int): Start time in seconds for the video segment to be cut.
        end (float or int): End time in seconds for the video segment to be cut.
        crop (List[int]): A list specifying crop parameters [x, width, y, height], where:
                          - x (int): The x-coordinate of the top-left corner of the crop area.
                          - width (int): The width of the crop area.
                          - y (int): The y-coordinate of the top-left corner of the crop area.
                          - height (int): The height of the crop area.

    Returns:
        bool: True if the cutting and cropping were successful, False otherwise.

    Workflow:
        1. Constructs an `ffmpeg` command to cut the video from `start` to `end` and apply the specified crop filter.
        2. Executes the command with `subprocess.call` to process the video.
        3. Checks the return code to confirm successful execution. Prints a message if the process fails.

    Raises:
        ValueError: If `crop` does not contain exactly four values, or if any component is invalid.
        FileNotFoundError: If `ffmpeg` is not installed or accessible from the system PATH.

    Example:
        >>> cut_and_crop_video(
                raw_video_path="/path/to/source.mp4",
                output_path="/path/to/clip.mp4",
                start=10,
                end=20,
                crop=[50, 200, 30, 200]
            )
        True

    Notes:
        - Requires `ffmpeg` to be installed and accessible from the command line.
        - If `output_path` already exists, it will be overwritten.
        - `-qscale 0` is used to preserve the video quality.
        - The crop filter uses the format `crop=width:height:x:y`, where `x` and `y` specify the top-left corner.
    """
    x, out_w, y, out_h = crop

    command = [
        "ffmpeg", "-i", raw_video_path,
        "-strict", "-2",  # Some legacy arguments
        "-loglevel", "quiet",  # Verbosity arguments
        "-qscale", "0",  # Preserve the quality
        "-y",  # Overwrite if the file exists
        "-ss", str(start),
        "-to", str(end),
        "-filter:v", f"crop={out_w}:{out_h}:{x}:{y}",  # Crop arguments
        output_path
    ]
    return_code = subprocess.call(command)
    success = return_code == 0

    if not success:
        print(f'{Fore.RED} Command failed: {" ".join(command)}')

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HDTF dataset")
    parser.add_argument('-s', '--source_dir', type=str, default='HDTF_dataset',
                        help='Path to the directory with the dataset description')
    parser.add_argument('-o', '--output_dir', type=str,
                        default='dataset', help='Where to save the videos?')
    parser.add_argument('-w', '--num_workers', type=int,
                        default=1, help='Number of workers for downloading.')
    args = parser.parse_args()

    download_hdtf(
        args.source_dir,
        args.output_dir,
        args.num_workers,
    )
