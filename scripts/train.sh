DATASET_DIR=train_data/train2017

# Stage 1: Training Full Image Colorization
mkdir ./checkpoints/coco_full
#cp ./checkpoints/siggraph_retrained/latest_net_G.pth ./checkpoints/coco_full/
python train.py --stage full --name coco_full --sample_p 1.0 --niter 1 --niter_decay 2 --lr 0.0005 --model train --fineSize 128 --batch_size 16 --display_ncols 3 --display_freq 1600 --print_freq 1600 --train_img_dir $DATASET_DIR



#DATASET_DIR=train_data/train2017

# Stage 1: Training Full Image Colorization 
#здесь вместо с  load model для загрузки старых переобученных весов
#mkdir ./checkpoints/coco_full
#cp ./checkpoints/siggraph_retrained/latest_net_G.pth ./checkpoints/coco_full/
#python train.py --stage full --name coco_full --sample_p 1.0 --niter 1 --niter_decay 1 --load_model --lr 0.0005 --model train --fineSize 256 --batch_size 8 --display_ncols 3 --display_freq 1600 --print_freq 1600 --train_img_dir $DATASET_DIR







# Stage 2: Training Instance Image Colorization
'''mkdir ./checkpoints/coco_instance
cp ./checkpoints/coco_full/latest_net_G.pth ./checkpoints/coco_instance/
python train.py --stage instance --name coco_instance --sample_p 1.0 --niter 1 --niter_decay 1 --load_model --lr 0.0005 --model train --fineSize 256 --batch_size 8 --display_ncols 3 --display_freq 1600 --print_freq 1600 --train_img_dir $DATASET_DIR'''


# Stage 3: Training Fusion Module

'''mkdir ./checkpoints/coco_mask
cp ./checkpoints/coco_full/latest_net_G.pth ./checkpoints/coco_mask/latest_net_GF.pth
cp ./checkpoints/coco_instance/latest_net_G.pth ./checkpoints/coco_mask/latest_net_G.pth
cp ./checkpoints/coco_full/latest_net_G.pth ./checkpoints/coco_mask/latest_net_GComp.pth
python train.py --stage fusion --name coco_mask --sample_p 1.0 --niter 1 --niter_decay 1 --lr 0.00005 --model train --load_model --display_ncols 4 --fineSize 128 --batch_size 1 --display_freq 500 --print_freq 500 --train_img_dir $DATASET_DIR '''


#python test.py --name coco_full --results_img_dir ./results/ --batch_size 1 --test_img_dir test_data/


#python test_fusion.py --name test_fusion --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir example --results_img_dir results

# это сон 
#python test.py --name coco_full --results_img_dir ./results/ --batch_size 1 --test_img_dir example/
#python test.py --name coco_full --fineSize 256 --results_img_dir ./results/ --batch_size 1 --test_img_dir example/
