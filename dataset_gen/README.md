Dataset setup
===============

Dataset preparation for ESC-50, ESC-10 [[1]](#1), and UrbanSound8K [[2]](#2).

- FFmpeg should be installed.
- First of all, please make a directory to save datasets.

		mkdir [path]

## [ESC-50 and ESC-10](https://github.com/karoldvl/ESC-50) setup

	python esc_gen.py [path]

- Following files will be generated.
	- [path]/esc50/wav16.npz  # 16kHz, for EnvNet
	- [path]/esc50/wav44.npz  # 44.1kHz, for EnvNet-v2
	- [path]/esc10/wav16.npz
	- [path]/esc10/wav44.npz

## [UrbanSound8K](http://urbansounddataset.weebly.com/urbansound8k.html) setup

1. Download UrbanSound8K dataset from [this page](http://urbansounddataset.weebly.com/urbansound8k.html).

2. Move UrbanSound8K directory.

		mkdir -p [path]/urbansound8k
		mv UrbanSound8K [path]/urbansound8k/
		
3. Run the following command.

		python urbansound_gen.py [path]
		
- Following files will be generated.
	- [path]/urbansound8k/wav16.npz
	- [path]/urbansound8k/wav44.npz


---
#### Reference
<i id=1></i>[1] Karol J Piczak. Esc: Dataset for environmental sound classification. In *ACM Multimedia*, 2015.

<i id=2></i>[2] Justin Salamon, Christopher Jacoby, and Juan Pablo Bello. A dataset and taxonomy for urban sound research. In *ACM Multimedia*, 2014.
