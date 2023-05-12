# KARTIK: These commands were used by me while carrying out PSKD
#### CIFAR100
# RESNET-8
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision fp16 main.py --lr 0.1 --end_epoch 160 --weight_decay 1e-4 --lr_decay_schedule 80 120 --lr_decay_rate 0.1 --PSKD --experiments_dir Results/resnet8_0.8 --batch_size 128 --classifier_type resnet8 --data_path Datasets --data_type cifar100 --alpha_T 0.8

# RESNET-18 (alpha_T taken from the PSKD paper)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision fp16 main.py --lr 0.1 --end_epoch 160 --weight_decay 1e-4 --lr_decay_schedule 80 120 --lr_decay_rate 0.1 --PSKD --experiments_dir Results/resnet18_0.8 --batch_size 128 --classifier_type ResNet18 --data_path Datasets --data_type cifar100 --alpha_T 0.8

# CONVNET-2 (alpha_T taken from the PSKD paper)
CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision fp16 main.py --lr 0.01 --end_epoch 160 --weight_decay 1e-4 --lr_decay_schedule 80 120 --lr_decay_rate 1 --PSKD --experiments_dir Results/convnet2 --batch_size 128 --classifier_type convnet2 --data_path Datasets --data_type cifar100 --alpha_T 0.8

# CONVNET-4 (alpha_T taken from the PSKD paper)
CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision fp16 main.py --lr 0.01 --end_epoch 160 --weight_decay 1e-4 --lr_decay_schedule 80 120 --lr_decay_rate 1 --PSKD --experiments_dir Results/convnet4 --batch_size 128 --classifier_type convnet4 --data_path Datasets --data_type cifar100 --alpha_T 0.8


#### CIFAR10
# RESNET-8
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision fp16 main.py --lr 0.1 --end_epoch 160 --weight_decay 1e-4 --lr_decay_schedule 80 120 --lr_decay_rate 0.1 --PSKD --experiments_dir Results/resnet8_0.8 --batch_size 128 --classifier_type resnet8 --data_path Datasets --data_type cifar10 --alpha_T 0.8

# RESNET-18 (alpha_T taken from the PSKD paper)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision fp16 main.py --lr 0.1 --end_epoch 160 --weight_decay 1e-4 --lr_decay_schedule 80 120 --lr_decay_rate 0.1 --PSKD --experiments_dir Results/resnet18_0.8 --batch_size 128 --classifier_type ResNet18 --data_path Datasets --data_type cifar10 --alpha_T 0.8

# CONVNET-2 (alpha_T taken from the PSKD paper)
CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision fp16 main.py --lr 0.01 --end_epoch 160 --weight_decay 1e-4 --lr_decay_schedule 80 120 --lr_decay_rate 1 --PSKD --experiments_dir Results/convnet2 --batch_size 128 --classifier_type convnet2 --data_path Datasets --data_type cifar10 --alpha_T 0.8

# CONVNET-4 (alpha_T taken from the PSKD paper)
CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision fp16 main.py --lr 0.01 --end_epoch 160 --weight_decay 1e-4 --lr_decay_schedule 80 120 --lr_decay_rate 1 --PSKD --experiments_dir Results/convnet4 --batch_size 128 --classifier_type convnet4 --data_path Datasets --data_type cifar10 --alpha_T 0.8
