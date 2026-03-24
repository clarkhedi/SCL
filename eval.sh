CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=25902 train.py \
--config ./configs/retrieval_cuhk.yaml \
--output_dir output/test \
--batch_size_test 150 \
--k_test 128 \
--pretrained .../checkpoint_best.pth \
--evaluate
