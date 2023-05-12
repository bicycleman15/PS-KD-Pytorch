import torch

#for cifar
from models.densenet_cifar import *
from models.preact_resnet import *

from models.pyramid import *
from models.pyramid_shake_drop import *

from models.resnext import *

from models.resnet_cifar import _get_cifar_resnet
from models.convnet_cifar import _get_cifar_convnet

#for imagenet
from models.resnet_imagenet import *

# for tiny_imagenet
from models.resnet_tinyimagenet import _get_tinyimagenet_resnet

from models.mobilenetv2 import _get_mobilenetv2
from models.shufflenetv2 import _get_shufflenetv2