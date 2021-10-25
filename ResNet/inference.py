import os
import json
import torch
import warnings
import argparse
import numpy as np
from torch.autograd.grad_mode import F

from torch.backends import cudnn

from torchvision import transforms
from torchvision import models
from torchvision import datasets
from typing import Optional, Callable, Any

from utils import read_images
from base import LABEL_MAP

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser("Pytorch Model Inference.")
parser.add_argument('--data', metavar='DIR', required=True,
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--resume', default='', type=str, metavar='PATH', required=True,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--output', default=None, type=str,
                    help="path to dump the result json file.")


class InputData(datasets.VisionDataset):

    def __init__(
        self, 
        root: str, 
        transform: Optional[Callable] = None) -> None:
        
        super().__init__(root, transform=transform)

        self.transform = transform
        
        self.data = read_images(root)
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Any:
        img = self.data[index]

        if self.transform is not None: 
            img = self.transform(img)
        
        return img, 0

def inference(dataloader, model, gpu):
    # switch to evaluate mode
    model.eval()
    labels = []

    with torch.no_grad():
        for images, _ in dataloader:
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
            
            # compute output
            output = model(images)
            
            batch_labels = torch.argmax(output, dim=1)
            # measure accuracy and record loss
            
            labels.append(batch_labels)

    return torch.cat(labels, 0).cpu().numpy()

best_acc1 = 0

def main():
    global best_acc1

    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        print("Use GPU: {} for training".format(args.gpu))

    model = models.__dict__[args.arch]()

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=(0.5019333, 0.49997845, 0.44068748),
                                     std=(0.28612566, 0.26815864, 0.28786656))

    dataloader = torch.utils.data.DataLoader(
        InputData(
            args.data,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
                ])
        ), 
        batch_size=args.batch_size, shuffle=False)

    indices = inference(dataloader, model, args.gpu)

    ids = [name.split('.')[0] for name in sorted(os.listdir(args.data))]
    labels = [LABEL_MAP[index] for index in indices]

    out_dict = dict(zip(ids, labels))

    print(out_dict)

    output_path = "result.json"
    if args.output is not None:
        output_path = os.path.join(args.output, output_path)

    with open(output_path, "w") as fp:
        json.dump(out_dict, fp)

if __name__ == '__main__':
    main()