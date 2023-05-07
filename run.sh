CUDA_VISIBLE_DEVICES=0 python main.py \
--lr 0.1 \
--end_epoch 160 \
--weight_decay 1e-4 \
--lr_decay_schedule 80 120 \
--PSKD \
--experiments_dir Results \
--batch_size 128 \
--classifier_type resnet8 \
--data_path Datasets \
--data_type cifar100 \
--alpha_T 0.9

CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.1 --end_epoch 160 --weight_decay 1e-4 --lr_decay_schedule 80 120 --PSKD --experiments_dir Results --batch_size 128 --classifier_type resnet56 --data_path Datasets --data_type cifar100 --alpha_T 0.7
python main.py --lr 0.1 --end_epoch 160 --weight_decay 1e-4 --lr_decay_schedule 80 120 --PSKD --experiments_dir Results --batch_size 128 --classifier_type resnet56 --data_path Datasets --data_type cifar100 --alpha_T 0.8
python main.py --lr 0.1 --end_epoch 160 --weight_decay 1e-4 --lr_decay_schedule 80 120 --PSKD --experiments_dir Results --batch_size 128 --classifier_type resnet56 --data_path Datasets --data_type cifar100 --alpha_T 0.9
python main.py --lr 0.1 --end_epoch 160 --weight_decay 1e-4 --lr_decay_schedule 80 120 --PSKD --experiments_dir Results --batch_size 128 --classifier_type resnet56 --data_path Datasets --data_type cifar100 --alpha_T 1.0