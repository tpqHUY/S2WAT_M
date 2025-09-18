## train from scratch

python train.py --content_dir ./Train_T/Content --style_dir ./Train_T/Style --vgg_dir ./pre_trained_models/vgg_normalised.pth --epoch 2000 --batch_size 2 --checkpoint_save_path ./pre_trained_models/checkpoints --checkpoint_save_interval 2000 --loss_count_interval 400




### Junk command for testing
python train.py --content_dir ./datasetforTESTING/Content --style_dir ./datasetforTESTING/Style --vgg_dir ./pre_trained_models/vgg_normalised.pth --epoch 2000 --batch_size 2 --checkpoint_save_path ./pre_trained_models/checkpoints --checkpoint_save_interval 2000 --loss_count_interval 400

python train.py --content_dir ./datasetforTESTING/Content --style_dir ./datasetforTESTING/Style --vgg_dir ./pre_trained_models/vgg_normalised.pth --epoch 2000 --batch_size 1 --checkpoint_save_path ./pre_trained_models/checkpoints --checkpoint_save_interval 2000 --loss_count_interval 400