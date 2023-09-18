##### train
CUDA_VISIBLE_DEVICES=6 python main.py \
--data_folder /data2/common/cifar \
--dataset cifar10 \
--output_folder ./output \
--exp_name cifar10_evq_cos_probrandom_8192x4 \
--batch_size 1024 \
--device cuda \
--num_epochs 500 \
--num_embedding 8192 \
--embedding_dim 4 \
--lora_codebook \
--evq \
--distance cos \
--num_workers 8 \
--anchor probrandom 2>&1 | tee ./output/cifar10_evq_cos_probrandom_8192x4_train.log
 
##### test
CUDA_VISIBLE_DEVICES=6 python test.py \
--data_folder /data2/common/cifar \
--dataset cifar10 \
--output_folder ./output \
--model_name cifar10_evq_cos_probrandom_8192x4/best.pt \
--batch_size 16 \
--device cuda \
--num_embedding 8192 \
--embedding_dim 4 \
--lora_codebook \
--evq \
--distance cos  2>&1 | tee ./output/cifar10_evq_cos_probrandom_8192x4_test.log
 
##### eval
CUDA_VISIBLE_DEVICES=6 python evaluation.py \
--gt_path ./output/results/cifar10_evq_cos_probrandom_8192x4/best.pt/original \
--g_path ./output/results/cifar10_evq_cos_probrandom_8192x4/best.pt/rec 2>&1 | tee ./output/cifar10_evq_cos_probrandom_8192x4_eval.log
 
