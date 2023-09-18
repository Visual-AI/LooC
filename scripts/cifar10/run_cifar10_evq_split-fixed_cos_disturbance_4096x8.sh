##### train
# CUDA_VISIBLE_DEVICES=5 python main.py \
# --data_folder /data2/common/cifar \
# --dataset cifar10 \
# --output_folder ./output \
# --exp_name cifar10_evq_split-fixed_cos_disturbance_4096x8 \
# --batch_size 1024 \
# --num_epochs 500 \
# --num_embedding 4096 \
# --embedding_dim 8 \
# --lora_codebook \
# --evq \
# --split_type fixed \
# --distance cos \
# --num_workers 8 \
# --anchor disturbance \
# --device cuda 2>&1 | tee ./output/cifar10_evq_split-fixed_cos_disturbance_4096x8_train.log
 
##### test
CUDA_VISIBLE_DEVICES=5 python test.py \
--data_folder /data2/common/cifar \
--dataset cifar10 \
--output_folder ./output \
--model_name cifar10_evq_split-fixed_cos_disturbance_4096x8/best.pt \
--batch_size 16 \
--num_embedding 4096 \
--embedding_dim 8 \
--lora_codebook \
--evq \
--split_type fixed \
--distance cos \
--device cuda 2>&1 | tee ./output/cifar10_evq_split-fixed_cos_disturbance_4096x8_test.log
 
##### eval
CUDA_VISIBLE_DEVICES=5 python evaluation.py \
--use_gpu \
--gt_path ./output/results/cifar10_evq_split-fixed_cos_disturbance_4096x8/best.pt/original \
--g_path ./output/results/cifar10_evq_split-fixed_cos_disturbance_4096x8/best.pt/rec 2>&1 | tee ./output/cifar10_evq_split-fixed_cos_disturbance_4096x8_eval.log
 
