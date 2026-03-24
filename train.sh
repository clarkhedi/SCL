CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.run --nproc_per_node=5 --master_port=25902 train.py \
--config ./configs/retrieval_cuhk.yaml \
--output_dir output/CUHK \
--max_epoch 40 \
--batch_size_train 32 \
--batch_size_test 150 \
--init_lr 1e-5  \
--k_test 128 \
--epoch_eval 1 \

