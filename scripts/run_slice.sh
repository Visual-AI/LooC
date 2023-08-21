##### test

# ----vislab12

CUDA_VISIBLE_DEVICES=1 python test.py \
--data_folder /data2/common/cifar \
--dataset cifar10 \
--output_folder ./output_slice \
--model_name ./output/models/cifar10_evq_cos_random_256x128/best.pt \
--batch_size 16 \
--device cuda \
--num_embedding 256 \
--embedding_dim 128 \
--lora_codebook \
--evq \
--slice_num 4 \
--distance cos  2>&1 | tee ./output_slice/cifar10_evq_cos_random_256x128_slice4_test.log

##### eval

CUDA_VISIBLE_DEVICES=3 python evaluation.py \
--gt_path ./output_slice/results/./output/models/cifar10_evq_cos_random_256x128/best.pt/original \
--g_path ./output_slice/results/./output/models/cifar10_evq_cos_random_256x128/best.pt/rec  2>&1 | tee ./output_slice/cifar10_evq_cos_random_256x128_slice4_eval.log
