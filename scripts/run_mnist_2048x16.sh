##### train
CUDA_VISIBLE_DEVICES=4 python main.py \
--data_folder /home2/jieli/datasets/mnist \
--dataset mnist \
--output_folder ./output \
--exp_name mnist_cos_closest_2048x16 \
--batch_size 400 \
--device cuda \
--num_epochs 500 \
--num_embedding 2048 \
--embedding_dim 16 \
--lora_codebook \
--distance cos \
--num_workers 8 \
--anchor closest 2>&1 | tee ./output/mnist_cos_closest_2048x16_train.log
 
##### test
CUDA_VISIBLE_DEVICES=4 python test.py \
--data_folder /home2/jieli/datasets/mnist \
--dataset mnist \
--output_folder ./output \
--model_name mnist_cos_closest_2048x16/best.pt \
--batch_size 16 \
--device cuda \
--num_embedding 2048 \
--embedding_dim 16 \
--lora_codebook \
--distance cos  2>&1 | tee ./output/mnist_cos_closest_2048x16_test.log
 
##### eval
CUDA_VISIBLE_DEVICES=4 python evaluation.py \
--gt_path ./output/results/mnist_cos_closest_2048x16/best.pt/original \
--g_path ./output/results/mnist_cos_closest_2048x16/best.pt/rec 2>&1 | tee ./output/mnist_cos_closest_2048x16_eval.log
 
