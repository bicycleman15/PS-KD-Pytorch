#--------------
#CNN-architecture
#--------------
from models import _get_cifar_resnet, _get_cifar_convnet, PyramidNet, PyramidNet_ShakeDrop, CIFAR_ResNet18_preActBasic, CIFAR_ResNet101_Bottle, CIFAR_DenseNet121, CifarResNeXt, ResNet, _get_tinyimagenet_resnet, _get_mobilenetv2, _get_shufflenetv2

#--------------
#util
#--------------
from utils.color import Colorer


C = Colorer.instance()

def get_network(args):
    ################################
    #Declare instance for Clasifier#
    ################################

    if args.data_type == 'cifar100':
        if args.classifier_type == 'PyramidNet':
            net = PyramidNet(dataset = 'cifar100', depth=200, alpha=240, num_classes=100,bottleneck=True)
        elif args.classifier_type == 'PyramidNet_SD':
            net = PyramidNet_ShakeDrop(dataset = 'cifar100', depth=200, alpha=240, num_classes=100,bottleneck=True)
        elif args.classifier_type == 'ResNet18':
            net = CIFAR_ResNet18_preActBasic(num_classes=100)
        elif args.classifier_type == 'ResNet101':
            net = CIFAR_ResNet101_Bottle(num_classes=100)
        elif args.classifier_type == 'DenseNet121':
            net = CIFAR_DenseNet121(num_classes=100, bias=True)
        elif args.classifier_type == 'ResNeXt':
            net = CifarResNeXt(cardinality=8, depth=29, nlabels=100, base_width=64, widen_factor=4)
        elif "resnet" in args.classifier_type:
            net = _get_cifar_resnet(args.classifier_type, "cifar100")
        elif "convnet" in args.classifier_type:
            net = _get_cifar_convnet(args.classifier_type, "cifar100")
        elif "mobilenetv2" in args.classifier_type:
            net = _get_mobilenetv2(args.classifier_type, "cifar100")
        elif "shufflenetv2" in args.classifier_type:
            net = _get_shufflenetv2(args.classifier_type, "cifar100")

    if args.data_type == 'cifar10':
        if args.classifier_type == 'PyramidNet':
            net = PyramidNet(dataset = 'cifar10', depth=200, alpha=240, num_classes=10,bottleneck=True)
        elif args.classifier_type == 'PyramidNet_SD':
            net = PyramidNet_ShakeDrop(dataset = 'cifar10', depth=200, alpha=240, num_classes=10,bottleneck=True)
        elif args.classifier_type == 'ResNet18':
            net = CIFAR_ResNet18_preActBasic(num_classes=10)
        elif args.classifier_type == 'ResNet101':
            net = CIFAR_ResNet101_Bottle(num_classes=10)
        elif args.classifier_type == 'DenseNet121':
            net = CIFAR_DenseNet121(num_classes=10, bias=True)
        elif args.classifier_type == 'ResNeXt':
            net = CifarResNeXt(cardinality=8, depth=29, nlabels=10, base_width=64, widen_factor=4)
        elif "resnet" in args.classifier_type:
            net = _get_cifar_resnet(args.classifier_type, "cifar10")
        elif "convnet" in args.classifier_type:
            net = _get_cifar_convnet(args.classifier_type, "cifar10")
        elif "mobilenetv2" in args.classifier_type:
            net = _get_mobilenetv2(args.classifier_type, "cifar10")
        elif "shufflenetv2" in args.classifier_type:
            net = _get_shufflenetv2(args.classifier_type, "cifar10")
    
    if args.data_type == 'imagenet':
        if args.classifier_type == 'ResNet152':
            net = ResNet(dataset = 'imagenet', depth=152, num_classes=1000, bottleneck=True)   

    if args.data_type == 'tiny_imagenet':
        if "resnet" in args.classifier_type:
            net = _get_tinyimagenet_resnet(args.classifier_type, "tiny_imagenet")
        elif "mobilenetv2" in args.classifier_type:
            net = _get_mobilenetv2(args.classifier_type, "tiny_imagenet")
        elif "shufflenetv2" in args.classifier_type:
            net = _get_shufflenetv2(args.classifier_type, "tiny_imagenet")
 
    print(C.underline(C.yellow("[Info] Building model: {}".format(args.classifier_type))))


    return net
