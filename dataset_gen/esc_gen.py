"""
 Dataset preparation code for ESC-10 and ESC-50 [Piczak 2015].
 Usage: python esc_gen.py [root]
 ffmpeg is required.

"""

import sys
import os
import subprocess

import glob
import numpy as np
import wavio


def main():
    esc10_path = os.path.join(sys.argv[1], 'esc10')
    esc50_path = os.path.join(sys.argv[1], 'esc50')
    fs_list = [16000, 44100]

    # Download ESC-10 and ESC-50
    os.mkdir(esc10_path)
    subprocess.call('wget -P {} https://github.com/karoldvl/ESC-10/archive/master.zip'.format(
        esc10_path), shell=True)
    subprocess.call('unzip -d {} {}'.format(
        esc10_path, os.path.join(esc10_path, 'master.zip')), shell=True)
    os.remove(os.path.join(esc10_path, 'master.zip'))

    os.mkdir(esc50_path)
    subprocess.call('wget -P {} https://github.com/karoldvl/ESC-50/archive/master.zip'.format(
        esc50_path), shell=True)
    subprocess.call('unzip -d {} {}'.format(
        esc50_path, os.path.join(esc50_path, 'master.zip')), shell=True)
    os.remove(os.path.join(esc50_path, 'master.zip'))

    # Remove the spaces in the folder names
    subprocess.call('rename "s/ //g" {}'.format(
        os.path.join(esc10_path, 'ESC-10-master', '*')), shell=True)
    subprocess.call('rename "s/ //g" {}'.format(
        os.path.join(esc50_path, 'ESC-50-master', '*')), shell=True)

    # Convert sampling rate
    for fs in fs_list:
        convert_fs(os.path.join(esc10_path, 'ESC-10-master'),
                   os.path.join(esc10_path, 'wav{}'.format(fs // 1000)),
                   fs)
        convert_fs(os.path.join(esc50_path, 'ESC-50-master'),
                   os.path.join(esc50_path, 'wav{}'.format(fs // 1000)),
                   fs)

    # Create npz files
    for fs in fs_list:
        src_path = os.path.join(esc10_path, 'wav{}'.format(fs // 1000))
        create_dataset(src_path, src_path + '.npz')
        src_path = os.path.join(esc50_path, 'wav{}'.format(fs // 1000))
        create_dataset(src_path, src_path + '.npz')


def convert_fs(src_path, dst_path, fs):
    print('* {} -> {}'.format(src_path, dst_path))
    os.mkdir(dst_path)
    for cls in sorted(os.listdir(src_path)):
        if os.path.isdir(os.path.join(src_path, cls)):
            os.mkdir(os.path.join(dst_path, cls))
            for ogg_file in sorted(glob.glob(os.path.join(src_path, cls, '*.ogg'))):
                wav_file = ogg_file.replace(src_path, dst_path).replace('.ogg', '.wav')
                subprocess.call('ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(
                    ogg_file, fs, wav_file), shell=True)


def create_dataset(src_path, dst_path):
    print('* {} -> {}'.format(src_path, dst_path))
    dataset = {}
    for fold in range(1, 6):
        dataset['fold{}'.format(fold)] = {}
        sounds = []
        labels = []
        for i, cls in enumerate(sorted(os.listdir(src_path))):
            for wav_file in sorted(glob.glob(os.path.join(src_path, cls, '{}-*.wav'.format(fold)))):
                sound = wavio.read(wav_file).data.T[0]
                start = sound.nonzero()[0].min()
                end = sound.nonzero()[0].max()
                sound = sound[start: end + 1]
                sounds.append(sound)
                labels.append(i)
        dataset['fold{}'.format(fold)]['sounds'] = sounds
        dataset['fold{}'.format(fold)]['labels'] = labels

    np.savez(dst_path, **dataset)


if __name__ == '__main__':
    main()
