##### train
CUDA_VISIBLE_DEVICES=6 python main.py \
--data_folder /data2/common/cifar \
--dataset cifar10 \
--output_folder ./output \
--exp_name cifar10_evq_split-fixed_cos_random_8192x2 \
--batch_size 256 \
--device cuda \
--num_epochs 500 \
--num_embedding 8192 \
--embedding_dim 2 \
--lora_codebook \
--evq \
--split_type fixed \
--distance cos \
--num_workers 8 \
--anchor random 2>&1 | tee ./output/cifar10_evq_split-fixed_cos_random_8192x2_train.log
 
##### test
CUDA_VISIBLE_DEVICES=6 python test.py \
--data_folder /data2/common/cifar \
--dataset cifar10 \
--output_folder ./output \
--model_name cifar10_evq_split-fixed_cos_random_8192x2/best.pt \
--batch_size 16 \
--device cuda \
--num_embedding 8192 \
--embedding_dim 2 \
--lora_codebook \
--evq \
--split_type fixed \
--distance cos  2>&1 | tee ./output/cifar10_evq_split-fixed_cos_random_8192x2_test.log
 
##### eval
CUDA_VISIBLE_DEVICES=6 python evaluation.py \
--use_gpu \
--gt_path ./output/results/cifar10_evq_split-fixed_cos_random_8192x2/best.pt/original \
--g_path ./output/results/cifar10_evq_split-fixed_cos_random_8192x2/best.pt/rec 2>&1 | tee ./output/cifar10_evq_split-fixed_cos_random_8192x2_eval.log
 
