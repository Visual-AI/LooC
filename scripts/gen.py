

def get_sh(cfg):
    # output_folder = f"./output_lora_codebook_{cfg['embedding_num']}x{cfg['embedding_dim']}"
    output_folder = f"./output"
    str_evq = '_evq' if cfg['evq'] else ''
    str_split = f"_split-{cfg.get('split_type')}" if cfg.get('split_type') else ""
    exp_name = f"{cfg['dataset']}{str_evq}{str_split}_{cfg['distance']}_{cfg['anchor']}_{cfg['embedding_num']}x{cfg['embedding_dim']}"

    str_list = []
    ##### train
    str_list.append("##### train")
    str_list.append(f"CUDA_VISIBLE_DEVICES={cfg['gpu_id']} python main.py \\")
    str_list.append(f"--data_folder {cfg['data_folder']} \\")
    str_list.append(f"--dataset {cfg['dataset']} \\")
    str_list.append(f"--output_folder {output_folder} \\")
    str_list.append(f"--exp_name {exp_name} \\")
    str_list.append("--batch_size 1024 \\")

    str_list.append("--num_epochs 500 \\")
    str_list.append(f"--num_embedding {cfg['embedding_num']} \\")
    str_list.append(f"--embedding_dim {cfg['embedding_dim']} \\")

    # --- store ture
    if cfg['lora_codebook']:
        str_list.append("--lora_codebook \\")
    if cfg['evq']:
        str_list.append("--evq \\")
    if cfg['scale_grad_by_freq']:
        str_list.append("--scale_grad_by_freq \\")

    if cfg.get('split_type'):
        str_list.append(f"--split_type {cfg.get('split_type')} \\")

    str_list.append(f"--distance {cfg['distance']} \\")  
    str_list.append("--num_workers 8 \\")
    str_list.append(f"--anchor {cfg['anchor']} \\")
    str_list.append(f"--device cuda 2>&1 | tee {output_folder}/{exp_name}_train.log")
    str_list.append(" ")

    ##### test
    str_list.append("##### test")
    str_list.append(f"CUDA_VISIBLE_DEVICES={cfg['gpu_id']} python test.py \\")
    str_list.append(f"--data_folder {cfg['data_folder']} \\")
    str_list.append(f"--dataset {cfg['dataset']} \\")
    str_list.append(f"--output_folder {output_folder} \\")
    str_list.append(f"--model_name {exp_name}/best.pt \\")
    str_list.append("--batch_size 16 \\")
    str_list.append(f"--num_embedding {cfg['embedding_num']} \\")
    str_list.append(f"--embedding_dim {cfg['embedding_dim']} \\")

    # --- store ture
    if cfg['lora_codebook']:
        str_list.append("--lora_codebook \\")
    if cfg['evq']:
        str_list.append("--evq \\")
    if cfg['scale_grad_by_freq']:
        str_list.append("--scale_grad_by_freq \\")

    if cfg.get('split_type'):
        str_list.append(f"--split_type {cfg.get('split_type')} \\")
    
    str_list.append(f"--distance {cfg['distance']} \\")
    str_list.append(f"--device cuda 2>&1 | tee {output_folder}/{exp_name}_test.log")
    str_list.append(" ")

    ##### eval
    str_list.append("##### eval")
    str_list.append(f"CUDA_VISIBLE_DEVICES={cfg['gpu_id']} python evaluation.py \\")
    str_list.append("--use_gpu \\")
    str_list.append(f"--gt_path {output_folder}/results/{exp_name}/best.pt/original \\")
    str_list.append(f"--g_path {output_folder}/results/{exp_name}/best.pt/rec 2>&1 | tee {output_folder}/{exp_name}_eval.log")
    str_list.append(" ")

    # -----
    # sh_filename = f"scripts/run_{cfg['dataset']}{str_evq}_{cfg['anchor']}_{cfg['embedding_num']}x{cfg['embedding_dim']}.sh"
    sh_filename = f"scripts/run_{exp_name}.sh"
    fw = open(sh_filename,'w')
    for line in str_list: 
        fw.write(line)  # 将字符串写入文件中
        fw.write("\n")  # 换行
    fw.close()
    print(sh_filename)


if __name__ == '__main__':
    embedding_num_dim = [
        # exp: codebook size = 256 x 128
        # (256, 128),
        # (512, 64),
        # (1024, 32),
        # (2048, 16),
        # (4096, 8),
        # (8192, 4),

        # exp: dim = 2
        # (8, 2),
        # (16, 2),
        # (24, 2),
        # (32, 2),
        # (48, 2),

        # (64, 2),
        # (80, 2),
        # (96, 2),
        # (112, 2),
        # (128, 2),

        # (256, 2),
        # (512, 2),
        # (1024, 2),
        # (2048, 2),
        # (4096, 2),
        # (8192, 2), 

        (256, 1),
        (512, 1),
        (1024, 1),
        (2048, 1),
        (4096, 1),
        (8192, 1), 

        # exp: dim = 4
        # (8, 4),
        # (12, 4),
        # (16, 4),
        # (20, 4),
        # (24, 4),

        # (32, 4),
        # (48, 4),
        # (64, 4),
        # (80, 4),
        # (96, 4),
        # (112, 4),
        # (128, 4),
        #
        # (256, 4),
        # (512, 4),
        # (1024, 4),
        # (2048, 4),
        # (4096, 4),
        # (8192, 4),
    ]

    cfg = dict()

    # --- default setting
    cfg.update({
        "lora_codebook": True,
        "evq": True,                    # Efficient VQ
        "scale_grad_by_freq": False,    # scale_grad_by_freq
        "distance": 'cos',              # distance, 默认为'cos'
        "anchor": 'closest',            # sample 策略, 默认为'closest'
        "split_type": 'fixed',          # split type: fixed, interval, random

        # --- dataset Vislab 12
        "dataset": 'cifar10',
        "data_folder": '/data2/common/cifar',

        # --- dataset Vislab 13
        #
        # "dataset": 'mnist',
        # "data_folder": '/home2/jieli/datasets/mnist',
        #
        # "dataset": 'fashion-mnist',
        # "data_folder": '/home2/jieli/datasets/fashion-mnist',
        })
    
    # cfg.update({
    #     "gpu_id": 7,
    #     "embedding_num":2048, 
    #     "embedding_dim":16
    #     })
    # get_sh(cfg)

    cfg.update({
        # "anchor": 'probrandom',   # sample 策略
        # "anchor": 'random',       # sample 策略
        # "anchor": 'disturbance',  # sample 策略, # 对feature的每一位，增加一个[-1, 1] 之间的随机扰动
        "anchor": 'none',           # sample 策略, 不采用任何anchor
        "scale_grad_by_freq": True,
        # 
        # "split_type": 'interval',  # split type: fixed, interval, random
        # "split_type": 'random',  # split type: fixed, interval, random
        })
    
    gpu_id_start = 1

    # -----
    for n, d in embedding_num_dim:
        cfg.update({"gpu_id": gpu_id_start})
        gpu_id_start += 1
        cfg.update({"embedding_num":n, "embedding_dim":d})
        get_sh(cfg)


