"""
Download the clips within the MusicCaps dataset from YouTube.

Requires:
    - ffmpeg
    - yt-dlp
    - datasets[audio]
    - torchaudio
"""
import subprocess
import os
from pathlib import Path

from datasets import load_dataset, load_dataset_builder, Audio
import yt_dlp

import scipy.io.wavfile as wavf

def download_clip(
    video_identifier,
    output_filename,
    start_time,
    end_time,
    tmp_dir='/tmp/audioset',
    num_attempts=5,
    url_base='https://www.youtube.com/watch?v='
):
    status = False

    '''
    ydl_opts = {
        'format': 'bestaudio',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'start_time': start_time,
        'end_time': end_time,

    }
    with yt_dlp.YouTubeDL(ydl_opts) as ydl:
        try:
          error_code = ydl.download()
        except:
          return
    '''
    
    command = f"""
        yt-dlp --quiet --no-warnings -x --audio-format wav -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" "{url_base}{video_identifier}"
    """.strip()

    attempts = 0
    while True:
        try:
            output = subprocess.check_output(command, shell=True,
                                                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            print(err)
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    print(status)
    return status, 'Downloaded'


def main(
    data_dir: str,
    sampling_rate: int = 44100,
    limit: int = None,
    num_proc: int = 1,
    writer_batch_size: int = 1000,
):
    """
    Download the clips within the MusicCaps dataset from YouTube.

    Args:
        data_dir: Directory to save the clips to.
        sampling_rate: Sampling rate of the audio clips.
        limit: Limit the number of examples to download.
        num_proc: Number of processes to use for downloading.
        writer_batch_size: Batch size for writing the dataset. This is per process.
    """

    ds_builder = load_dataset_builder('agkphysics/AudioSet', cache_dir="./cache")
    print(ds_builder.info.features)
    print(ds_builder.info.features['audio'])
    ds = load_dataset('agkphysics/AudioSet', 'unbalanced')
    if limit is not None:
        print(f"Limiting to {limit} examples")
        ds = ds.select(range(limit))

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    def process(example):
        outfile_path = str(data_dir / f"{example['video_id']}.wav")
        status = True
        if not os.path.exists(outfile_path):
            #print("here")
            status = False
            '''
            status, log = download_clip(
                example['video_id'],
                outfile_path,
                example['start_s'],
                example['end_s'],
            )
            '''
            wavf.write(outfile_path, example['audio']['array'].size, example['audio']['array']) 
        example['audio'] = outfile_path
        return example

    return ds.map(
        process,
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        keep_in_memory=False
    ).cast_column('audio', Audio(sampling_rate=sampling_rate))


if __name__ == '__main__':
    ds = main(
        './audioset_data',
        sampling_rate=44100,
        limit=None,
        num_proc=16,
        writer_batch_size=1000,
    )
    print(ds)
    df = pd.Dataframe(ds)
    df.to_csv('audioset.csv')
