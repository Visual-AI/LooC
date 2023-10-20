#debug 增加了sim_loss, 

# ablation study:
#(1) 取匹配度前90%的codebook，测试不同codebook数量（逐步减少）下的精度影响
#(2) 设置相似度阈值，将相似度高于0.9的vector去掉。

# server: vislab 12
output_root=debug
# data_folder /data2/



##### train
CUDA_VISIBLE_DEVICES=4 python main.py \
--data_folder /data2/jieli/datasets/fashion-mnist \
--dataset fashion-mnist \
--output_folder ./debug \
--exp_name fashion-mnist_cos_closest_512x64 \
--batch_size 1024 \
--device cuda \
--num_epochs 500 \
--num_embedding 512 \
--embedding_dim 64 \
--lora_codebook \
--evq \
--distance cos \
--anchor closest 2>&1 | tee ./debug/fashion-mnist_cos_closest_512x64_train.log
 
##### test
CUDA_VISIBLE_DEVICES=4 python test.py \
--data_folder /data2/jieli/datasets/fashion-mnist \
--dataset fashion-mnist \
--output_folder ./debug \
--model_name fashion-mnist_cos_closest_512x64/best.pt \
--batch_size 16 \
--device cuda \
--num_embedding 512 \
--embedding_dim 64 \
--lora_codebook \
--evq \
--distance cos  2>&1 | tee ./debug/fashion-mnist_cos_closest_512x64_test.log
 
##### eval
CUDA_VISIBLE_DEVICES=4 python evaluation.py \
--gt_path ./debug/results/fashion-mnist_cos_closest_512x64/best.pt/original \
--g_path ./debug/results/fashion-mnist_cos_closest_512x64/best.pt/rec 2>&1 | tee ./debug/fashion-mnist_cos_closest_512x64_eval.log
 
