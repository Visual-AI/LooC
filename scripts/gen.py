

def get_sh(cfg):
    # output_folder = f"./output_lora_codebook_{cfg['embedding_num']}x{cfg['embedding_dim']}"
    output_folder = f"./output"
    str_evq = '_evq' if cfg['evq'] else ''
    exp_name = f"{cfg['dataset']}{str_evq}_{cfg['distance']}_{cfg['anchor']}_{cfg['embedding_num']}x{cfg['embedding_dim']}"
    str_list = []
    ##### train
    str_list.append("##### train")
    str_list.append(f"CUDA_VISIBLE_DEVICES={cfg['gpu_id']} python main.py \\")
    str_list.append(f"--data_folder {cfg['data_folder']} \\")
    str_list.append(f"--dataset {cfg['dataset']} \\")
    str_list.append(f"--output_folder {output_folder} \\")
    str_list.append(f"--exp_name {exp_name} \\")
    str_list.append("--batch_size 1024 \\")
    str_list.append("--device cuda \\")
    str_list.append("--num_epochs 500 \\")
    str_list.append(f"--num_embedding {cfg['embedding_num']} \\")
    str_list.append(f"--embedding_dim {cfg['embedding_dim']} \\")
    if cfg['lora_codebook']:
        str_list.append("--lora_codebook \\")
    if cfg['evq']:
        str_list.append("--evq \\")
    str_list.append(f"--distance {cfg['distance']} \\")  
    str_list.append("--num_workers 8 \\")
    str_list.append(f"--anchor {cfg['anchor']} 2>&1 | tee {output_folder}/{exp_name}_train.log")
    str_list.append(" ")

    ##### test
    str_list.append("##### test")
    str_list.append(f"CUDA_VISIBLE_DEVICES={cfg['gpu_id']} python test.py \\")
    str_list.append(f"--data_folder {cfg['data_folder']} \\")
    str_list.append(f"--dataset {cfg['dataset']} \\")
    str_list.append(f"--output_folder {output_folder} \\")
    str_list.append(f"--model_name {exp_name}/best.pt \\")
    str_list.append("--batch_size 16 \\")
    str_list.append("--device cuda \\")
    str_list.append(f"--num_embedding {cfg['embedding_num']} \\")
    str_list.append(f"--embedding_dim {cfg['embedding_dim']} \\")
    if cfg['lora_codebook']:
        str_list.append("--lora_codebook \\")
    if cfg['evq']:
        str_list.append("--evq \\")
    str_list.append(f"--distance {cfg['distance']}  2>&1 | tee {output_folder}/{exp_name}_test.log")
    str_list.append(" ")

    ##### eval
    str_list.append("##### eval")
    str_list.append(f"CUDA_VISIBLE_DEVICES={cfg['gpu_id']} python evaluation.py \\")
    str_list.append(f"--gt_path {output_folder}/results/{exp_name}/best.pt/original \\")
    str_list.append(f"--g_path {output_folder}/results/{exp_name}/best.pt/rec 2>&1 | tee {output_folder}/{exp_name}_eval.log")
    str_list.append(" ")

    # -----
    sh_filename = f"scripts/run_{cfg['dataset']}{str_evq}_{cfg['anchor']}_{cfg['embedding_num']}x{cfg['embedding_dim']}.sh"
    fw = open(sh_filename,'w')
    for line in str_list: 
        fw.write(line)  # 将字符串写入文件中
        fw.write("\n")  # 换行
    fw.close()
    print(sh_filename)


if __name__ == '__main__':
    embedding_num_dim = [
        (256, 128),
        (512, 64),
        (1024, 32),
        (2048, 16),
        (4096, 8),
        (8192, 4),
    ]

    cfg = dict()
    cfg.update({
        "lora_codebook": True,
        "evq": True,            # Efficient VQ
        "distance": 'cos',      # distance, 默认为'cos'
        "anchor": 'closest',    # sample 策略, 默认为'closest'
        # "dataset": 'mnist',
        # "data_folder": '/home2/jieli/datasets/mnist',

        # "dataset": 'fashion-mnist',
        # "data_folder": '/home2/jieli/datasets/fashion-mnist',

        # -- Vislab 12
        "dataset": 'cifar10',
        "data_folder": '/data2/common/cifar',
        })
    
    # cfg.update({
    #     "gpu_id": 7,
    #     "embedding_num":2048, 
    #     "embedding_dim":16
    #     })
    # get_sh(cfg)

    cfg.update({
        # "anchor": 'probrandom',  # sample 策略
        "anchor": 'random',  # sample 策略
        })
    
    gpu_id_start = 1

    # -----
    for n, d in embedding_num_dim:
        cfg.update({"gpu_id": gpu_id_start})
        gpu_id_start += 1
        cfg.update({"embedding_num":n, "embedding_dim":d})
        get_sh(cfg)


