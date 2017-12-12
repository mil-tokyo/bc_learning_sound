"""
 Learning from Between-class Examples for Deep Sound Recognition.
 Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada

"""

import sys
import os
import chainer

import opts
import models
import dataset
from train import Trainer


def main():
    opt = opts.parse()
    chainer.cuda.get_device_from_id(opt.gpu).use()
    for split in opt.splits:
        print('+-- Split {} --+'.format(split))
        train(opt, split)


def train(opt, split):
    model = getattr(models, opt.netType)(opt.nClasses)
    model.to_gpu()
    optimizer = chainer.optimizers.NesterovAG(lr=opt.LR, momentum=opt.momentum)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(opt.weightDecay))
    train_iter, val_iter = dataset.setup(opt, split)
    trainer = Trainer(model, optimizer, train_iter, val_iter, opt)

    if opt.testOnly:
        chainer.serializers.load_npz(
            os.path.join(opt.save, 'model_split{}.npz'.format(split)), trainer.model)
        val_top1 = trainer.val()
        print('| Val: top1 {:.2f}'.format(val_top1))
        return

    for epoch in range(1, opt.nEpochs + 1):
        train_loss, train_top1 = trainer.train(epoch)
        val_top1 = trainer.val()
        sys.stderr.write('\r\033[K')
        sys.stdout.write(
            '| Epoch: {}/{} | Train: LR {}  Loss {:.3f}  top1 {:.2f} | Val: top1 {:.2f}\n'.format(
                epoch, opt.nEpochs, trainer.optimizer.lr, train_loss, train_top1, val_top1))
        sys.stdout.flush()

    if opt.save != 'None':
        chainer.serializers.save_npz(
            os.path.join(opt.save, 'model_split{}.npz'.format(split)), model)


if __name__ == '__main__':
    main()
