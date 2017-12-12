"""
 Dataset preparation code for UrbanSound8k [Salamon, 2014].
 Usage: python urbansound_gen.py [path]
 Original dataset should be downloaded in [path]/urbansound8k/.
 FFmpeg should be installed.

"""

import sys
import os
import subprocess

import glob
import numpy as np
import wavio


def main():
    data_path = os.path.join(sys.argv[1], 'urbansound8k')
    fs_list = [16000, 44100]

    # Convert sampling rate
    for fs in fs_list:
        convert_fs(os.path.join(data_path, 'UrbanSound8K/audio'),
                   os.path.join(data_path, 'wav{}'.format(fs // 1000)),
                   fs)

    # Create npz files
    for fs in fs_list:
        src_path = os.path.join(data_path, 'wav{}'.format(fs // 1000))
        create_dataset(src_path, src_path + '.npz')


def convert_fs(src_path, dst_path, fs):
    print('* {} -> {}'.format(src_path, dst_path))
    os.mkdir(dst_path)
    for fold in sorted(os.listdir(src_path)):
        if os.path.isdir(os.path.join(src_path, fold)):
            os.mkdir(os.path.join(dst_path, fold))
            for src_file in sorted(glob.glob(os.path.join(src_path, fold, '*.wav'))):
                dst_file = src_file.replace(src_path, dst_path)
                subprocess.call('ffmpeg -i {} -ac 1 -ar {} -acodec pcm_s16le -loglevel error -y {}'.format(
                    src_file, fs, dst_file), shell=True)


def create_dataset(src_path, dst_path):
    print('* {} -> {}'.format(src_path, dst_path))
    dataset = {}
    for fold in range(1, 11):
        dataset['fold{}'.format(fold)] = {}
        sounds = []
        labels = []
        for wav_file in sorted(glob.glob(os.path.join(src_path, 'fold{}'.format(fold), '*.wav'))):
            sound = wavio.read(wav_file).data.T[0]
            label = wav_file.split('/')[-1].split('-')[1]
            sounds.append(sound)
            labels.append(label)
        dataset['fold{}'.format(fold)]['sounds'] = sounds
        dataset['fold{}'.format(fold)]['labels'] = labels

    np.savez(dst_path, **dataset)


if __name__ == '__main__':
    main()
