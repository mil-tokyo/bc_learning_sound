BC learning for sounds
============================

Implementation of [Learning from Between-class Examples for Deep Sound Recognition](https://arxiv.org/abs/1711.10282) by Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada.

This also contains training of EnvNet: [Learning Environmental Sounds with End-to-end Convolutional Neural Network](http://ieeexplore.ieee.org/document/7952651/) (Yuji Tokozume and Tatsuya Harada, ICASSP 2017).<sup>[1](#1)</sup>


## Contents

- Between-class (BC) learning
	- We generate between-class examples by mixing two training examples belonging to different classes with a random ratio.
	- We then input the mixed data to the model and
train the model to output the mixing ratio.
- EnvNet/EnvNet-v2 training
	- EnvNet-v2: a deeper version of EnvNet. The performance of it on ESC-50 surpasses the human level when using BC learning.


## Setup
- Install [Chainer](https://chainer.org/) v1.24 on a machine with CUDA GPU.
- Prepare datasets following [this page](https://github.com/mil-tokyo/bc_learning_sound/tree/master/dataset_gen).


## Training
- Template:

		python main.py --dataset [esc50, esc10, or urbansound8k] --netType [envnet or envnetv2] --data path/to/dataset/directory/ (--BC) (--strongAugment)
 
- Recipes:
	- Standard learning of EnvNet on ESC-50 (around 29% error<sup>[2](#2)</sup>):

			python main.py --dataset esc50 --netType envnet --data path/to/dataset/directory/
	

	- BC learning of EnvNet on ESC-50 (around 24% error):

			python main.py --dataset esc50 --netType envnet --data path/to/dataset/directory/ --BC
	
	- BC learning of EnvNet-v2 on ESC-50 with strong data augmentation (around 15% error, the best performance):

			python main.py --dataset esc50 --netType envnetv2 --data path/to/dataset/directory/ --BC --strongAugment
	
- Notes:
	- Validation accuracy is calculated using 10-crop testing.
	- By default, it performs K-fold cross validation using the original fold settings. You can run on a particular split by using --split command.
	- You can use your favorite learning settings as follows:

			python main.py --dataset esc50 --netType envnetv2 --data path/to/dataset/directory/ --nEpochs 1000 --LR 0.01 --schedule 0.5 0.75 --warmup 0


## Results

Error rate (Standard learning &rarr; BC learning)

| Model | ESC-50 | ESC-10 | UrbanSound8K |
|:--|:-:|:-:|:-:|
| EnvNet | 29.2 &rarr; **24.1** | 12.8 &rarr; **11.3** | 33.7 &rarr; **28.9** |
| EnvNet-v2 | 25.6 &rarr; **18.2** | 14.2 &rarr; **10.6** | 30.9 &rarr; **23.4** |
| EnvNet-v2 + <br> strong augment | 21.2 &rarr; **15.1** | 10.9 &rarr; **8.6** | 24.9 &rarr; **21.7** |
| Humans | 18.7 | 4.3 | - |

## See also
[Between-class Learning for Image Clasification](https://arxiv.org/abs/1711.10284) (github)

---
<i id=1></i><sup>1</sup> Training/testing schemes are simplified from those in the ICASSP paper.

<i id=2></i><sup>2</sup> It is higher than that reported in the ICASSP paper (36% error), mainly because here we use 4 out of 5 folds for training, whereas we used only 3 folds in the ICASSP paper.