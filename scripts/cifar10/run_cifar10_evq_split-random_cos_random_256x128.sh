##### train
CUDA_VISIBLE_DEVICES=1 python main.py \
--data_folder /data2/common/cifar \
--dataset cifar10 \
--output_folder ./output \
--exp_name cifar10_evq_split-random_cos_random_256x128 \
--batch_size 1024 \
--device cuda \
--num_epochs 500 \
--num_embedding 256 \
--embedding_dim 128 \
--lora_codebook \
--evq \
--split_type random \
--distance cos \
--num_workers 8 \
--anchor random 2>&1 | tee ./output/cifar10_evq_split-random_cos_random_256x128_train.log
 
##### test
CUDA_VISIBLE_DEVICES=1 python test.py \
--data_folder /data2/common/cifar \
--dataset cifar10 \
--output_folder ./output \
--model_name cifar10_evq_split-random_cos_random_256x128/best.pt \
--batch_size 16 \
--device cuda \
--num_embedding 256 \
--embedding_dim 128 \
--lora_codebook \
--evq \
--split_type random \
--distance cos  2>&1 | tee ./output/cifar10_evq_split-random_cos_random_256x128_test.log
 
##### eval
CUDA_VISIBLE_DEVICES=1 python evaluation.py \
--gt_path ./output/results/cifar10_evq_split-random_cos_random_256x128/best.pt/original \
--g_path ./output/results/cifar10_evq_split-random_cos_random_256x128/best.pt/rec 2>&1 | tee ./output/cifar10_evq_split-random_cos_random_256x128_eval.log
 
