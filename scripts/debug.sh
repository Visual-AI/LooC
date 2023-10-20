
# ----- vislab12

CUDA_VISIBLE_DEVICES=1 python main.py \
--data_folder /data2/common/cifar \
--dataset cifar10 \
--output_folder ./debug \
--exp_name cifar10_cos_closest_256x128_debug02 \
--batch_size 600 \
--device cuda \
--num_epochs 500 \
--num_embedding 512 \
--embedding_dim 64 \
--lora_codebook \
--distance cos \
--num_workers 8 \
--evq \
--anchor closest 2>&1 | tee ./debug/cifar10_cos_closest_512x64_debug02.log


CUDA_VISIBLE_DEVICES=3 python test.py \
--data_folder /data2/common/cifar \
--dataset cifar10 \
--output_folder ./output \
--model_name cifar10_evq_cos_random_1024x32/best.pt \
--batch_size 16 \
--device cuda \
--num_embedding 1024 \
--embedding_dim 32 \
--lora_codebook \
--evq \
--distance cos  
 