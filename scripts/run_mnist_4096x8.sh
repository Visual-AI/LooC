##### train
CUDA_VISIBLE_DEVICES=5 python main.py \
--data_folder /home2/jieli/datasets/mnist \
--dataset mnist \
--output_folder ./output \
--exp_name mnist_cos_closest_4096x8 \
--batch_size 100 \
--device cuda \
--num_epochs 500 \
--num_embedding 4096 \
--embedding_dim 8 \
--lora_codebook \
--distance cos \
--num_workers 8 \
--anchor closest 2>&1 | tee ./output/mnist_cos_closest_4096x8_train.log
 
##### test
CUDA_VISIBLE_DEVICES=5 python test.py \
--data_folder /home2/jieli/datasets/mnist \
--dataset mnist \
--output_folder ./output \
--model_name mnist_cos_closest_4096x8/best.pt \
--batch_size 16 \
--device cuda \
--num_embedding 4096 \
--embedding_dim 8 \
--lora_codebook \
--distance cos  2>&1 | tee ./output/mnist_cos_closest_4096x8_test.log
 
##### eval
CUDA_VISIBLE_DEVICES=5 python evaluation.py \
--gt_path ./output/results/mnist_cos_closest_4096x8/best.pt/original \
--g_path ./output/results/mnist_cos_closest_4096x8/best.pt/rec 2>&1 | tee ./output/mnist_cos_closest_4096x8_eval.log
 
