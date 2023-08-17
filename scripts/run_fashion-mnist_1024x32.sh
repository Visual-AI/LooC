##### train
CUDA_VISIBLE_DEVICES=5 python main.py \
--data_folder /home2/jieli/datasets/fashion-mnist \
--dataset fashion-mnist \
--output_folder ./output \
--exp_name fashion-mnist_cos_closest_1024x32 \
--batch_size 1024 \
--device cuda \
--num_epochs 500 \
--num_embedding 1024 \
--embedding_dim 32 \
--lora_codebook \
--distance cos \
--anchor closest 2>&1 | tee ./output/fashion-mnist_cos_closest_1024x32_train.log
 
##### test
CUDA_VISIBLE_DEVICES=5 python test.py \
--data_folder /home2/jieli/datasets/fashion-mnist \
--dataset fashion-mnist \
--output_folder ./output \
--model_name fashion-mnist_cos_closest_1024x32/best.pt \
--batch_size 16 \
--device cuda \
--num_embedding 1024 \
--embedding_dim 32 \
--lora_codebook \
--distance cos  2>&1 | tee ./output/fashion-mnist_cos_closest_1024x32_test.log
 
##### eval
CUDA_VISIBLE_DEVICES=5 python evaluation.py \
--gt_path ./output/results/fashion-mnist_cos_closest_1024x32/best.pt/original \
--g_path ./output/results/fashion-mnist_cos_closest_1024x32/best.pt/rec 2>&1 | tee ./output/fashion-mnist_cos_closest_1024x32_eval.log
 
