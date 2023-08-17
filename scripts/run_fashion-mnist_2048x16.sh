##### train
CUDA_VISIBLE_DEVICES=6 python main.py \
--data_folder /home2/jieli/datasets/fashion-mnist \
--dataset fashion-mnist \
--output_folder ./output \
--exp_name fashion-mnist_cos_closest_2048x16 \
--batch_size 400 \
--device cuda \
--num_epochs 500 \
--num_embedding 2048 \
--embedding_dim 16 \
--lora_codebook \
--distance cos \
--anchor closest 2>&1 | tee ./output/fashion-mnist_cos_closest_2048x16_train.log
 
##### test
CUDA_VISIBLE_DEVICES=6 python test.py \
--data_folder /home2/jieli/datasets/fashion-mnist \
--dataset fashion-mnist \
--output_folder ./output \
--model_name fashion-mnist_cos_closest_2048x16/best.pt \
--batch_size 16 \
--device cuda \
--num_embedding 2048 \
--embedding_dim 16 \
--lora_codebook \
--distance cos  2>&1 | tee ./output/fashion-mnist_cos_closest_2048x16_test.log
 
##### eval
CUDA_VISIBLE_DEVICES=6 python evaluation.py \
--gt_path ./output/results/fashion-mnist_cos_closest_2048x16/best.pt/original \
--g_path ./output/results/fashion-mnist_cos_closest_2048x16/best.pt/rec 2>&1 | tee ./output/fashion-mnist_cos_closest_2048x16_eval.log
 
