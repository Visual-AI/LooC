import os

vislab3_datapath = {
    'mnist':            '/disk2/jieli/datasets/mnist',
    'cifar10':          '/disk2/jieli/datasets/cifar',
    'fashion-mnist':    '/disk2/jieli/datasets/fashion-mnist',
}

vislab12_datapath = {
    'mnist':            '/data2/common/mnist',
    'cifar10':          '/data2/common/cifar',
    'fashion-mnist':    '/data2/common/fashion-mnist',
}

vislab13_datapath = {
    'mnist':            '/home2/jieli/datasets/mnist',
    'cifar10':          '/home2/jieli/datasets/cifar',
    'fashion-mnist':    '/home2/jieli/datasets/fashion-mnist',
}


def get_datapath(dataset_name):
    dataset_path = vislab3_datapath.get(dataset_name)
    if os.path.exists(dataset_path):
        return dataset_path
    
    dataset_path = vislab12_datapath.get(dataset_name)
    if os.path.exists(dataset_path):
        return dataset_path
    
    dataset_path = vislab13_datapath.get(dataset_name)
    if os.path.exists(dataset_path):
        return dataset_path

    raise ValueError("- "* 10 + "check the datapath")


def get_yaml(cfg, flag_debug=False):
    # output_folder = f"./output_lora_codebook_{cfg['embedding_num']}x{cfg['embedding_dim']}"
    # output_folder = f"./output"
    # str_evq = '_evq' if cfg['evq'] else ''
    # str_split = f"_split-{cfg.get('split_type')}" if cfg.get('split_type') else ""
    # exp_name = f"{cfg['dataset']}{str_evq}{str_split}_{cfg['distance']}_{cfg['anchor']}_{cfg['embedding_num']}x{cfg['embedding_dim']}"

    str_list = []
    # print('#'*10, list(cfg.keys()))

    exp_name = f"{cfg.get('dataset')}_{cfg.get('embedding_num')}x{cfg.get('embedding_dim')}x{cfg.get('shuffle_scale')}_{cfg.get('exp_tag')}"

    ##### train
    str_list.append("Enable_Wandb: False   # for debug" if flag_debug else "Enable_Wandb: True   # default: True")
    str_list.append("##### train")
    str_list.append(f"exp_name: {exp_name}")
    str_list.append(f"output_folder: {cfg.get('output_folder')}")
    str_list.append(f"dataset: {cfg.get('dataset')}")
    str_list.append(f"batch_size: {cfg.get('batch_size')}")
    str_list.append(f"num_epochs: 500       # number of epochs (default: 100)")

    str_list.append("")
    str_list.append(f"seed: 42              # seed for everything")
    str_list.append(f"lr: 3e-4")
    str_list.append(f"num_workers: 16")

    str_list.append("")
    str_list.append(f"# loss")
    str_list.append(f"calc_sim_loss: Ture")
    str_list.append(f"sim_loss: 0             # 权重为0,则表示不参与训练; 仅 calc_sim_loss=True & sim_loss > 0时参与训练")
    str_list.append(f"commitment_cost: 0.25   # hyperparameter for the commitment loss (default: 0.25)")
    
    str_list.append("")
    str_list.append(f"# model")
    str_list.append(f"# Latent space")
    str_list.append(f"hidden_size: 128        # size of the latent vectors (default: 128)")
    str_list.append(f"num_residual_hidden: 32 # size of the redisual layers (default: 32)")
    str_list.append(f"num_residual_layers: 2  # number of residual layers (default: 2)")

    str_list.append(f"")
    str_list.append(f"# codebook")
    str_list.append(f"# Quantiser parameters")
    str_list.append(f"shuffle_scale: {cfg.get('shuffle_scale')}      # scale factor for upsample and downsample the feature vectors")
    str_list.append(f"num_embedding: {cfg.get('embedding_num')}      # number of codebook (default: 512)")
    str_list.append(f"dim_embedding: {cfg.get('embedding_dim')}")
    str_list.append(f"# size_dmbedding: num_embedding * dim_embedding")
    
    str_list.append("")
    str_list.append(f"distance: {cfg.get('distance')}       # distance for codevectors and features")
    str_list.append(f"split_type: {cfg.get('split_type')}   # fixed, interval, random")
    str_list.append(f"anchor: {cfg.get('anchor')}           # anchor sampling methods (random, closest, probrandom)")
    
    str_list.append("")
    str_list.append(f"##### eval")
    str_list.append(f"num_test: 0  # how many examples to load for testing (default: 0 to test all samples)")

    str_list.append("")
    str_list.append(f"data_folder: {cfg.get('data_folder')}")

    str_list.append(" ")

    # -----
    yaml_filename = f"./config/{exp_name}.yaml"
    fw = open(yaml_filename,'w')
    for line in str_list: 
        fw.write(line)  # 将字符串写入文件中
        fw.write("\n")  # 换行
    fw.close()
    print('yaml_filename =', yaml_filename)

    return exp_name


def get_sh(cfg, exp_name):
    sh_list = []
    sh_list.append("")
    sh_list.append(f"expname={exp_name}")
    sh_list.append("")
    sh_list.append("##### train") # gpu_id
    sh_list.append(f"CUDA_VISIBLE_DEVICES={cfg.get('gpu_id')}" + " python main.py --cfg ./config/${expname}.yaml 2>&1 | tee " + f"./{cfg.get('output_folder')}/" + "${expname}_train.log")
    sh_list.append("##### test")
    sh_list.append(f"CUDA_VISIBLE_DEVICES={cfg.get('gpu_id')}" + " python test.py --cfg ./config/${expname}.yaml 2>&1 | tee " + f"./{cfg.get('output_folder')}/" + "${expname}_test.log")
    sh_list.append("##### eval")
    sh_list.append(f"CUDA_VISIBLE_DEVICES={cfg.get('gpu_id')}" + " python evaluation.py --cfg ./config/${expname}.yaml 2>&1 | tee " + f"./{cfg.get('output_folder')}/" + "${expname}_eval.log")

    # -----
    sh_filename = f"scripts/run_{exp_name}.sh"
    fw = open(sh_filename,'w')
    for line in sh_list: 
        fw.write(line)  # 将字符串写入文件中
        fw.write("\n")  # 换行
    fw.close()
    sh_str = f"sh {sh_filename}"
    print(sh_str)
    return sh_str


def main_exp_base(cfg, embedding_num_dim, gpu_list, merge_sh, flag_debug):
    output_folder = cfg.get('output_folder')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=False)

    # --- default setting
    cfg.update({
        "distance": 'cos',              # distance, 默认为'cos'
        "anchor": 'closest',            # sample 策略, 默认为'closest'
        "split_type": 'fixed',          # split type: fixed, interval, random
        "data_folder": get_datapath(cfg.get('dataset')),
        })
    # -----
    cnt = 0
    sh_str_list = []
    for num_dim_batch in embedding_num_dim:
        n, d = num_dim_batch[0:2]
        gpu_id = gpu_list[0] if merge_sh else gpu_list[cnt]
        cfg.update({"gpu_id": gpu_id})
        cfg.update({"embedding_num":n, "embedding_dim":d})
        if len(num_dim_batch) == 3:
            cfg.update({"batch_size": num_dim_batch[2]})
        exp_name = get_yaml(cfg, flag_debug)
        sh_str_i = get_sh(cfg, exp_name)
        sh_str_list.append(sh_str_i)
        cnt += 1

    if merge_sh:
        msh_filename = f"scripts/run_{cfg.get('dataset')}_{cfg.get('exp_tag')}"
        ss = cfg.get('shuffle_scale')
        msh_filename += "_["
        for num_dim_batch in embedding_num_dim:
            n, d = num_dim_batch[0:2]
            msh_filename += f"{n}x{d}_"
        msh_filename = msh_filename[:-1] + f"]x{ss}.sh"
        fw = open(msh_filename,'w')
        for line in sh_str_list: 
            fw.write(line)  # 将字符串写入文件中
            fw.write("\n")  # 换行
            fw.write("\n")  # 换行
        fw.close()
        msh_str = f"sh {msh_filename}"
        print(msh_str)


def main_exp_mnist():
    cfg = dict()
    # --- init setting 
    cfg.update({"dataset": 'mnist'})
    flag_debug = False                  # True for debug
    merge_sh = False                    # 
    gpu_list = [0, 1, 2, 3, 4, 5, 6, 7]       # 
    cfg.update({"shuffle_scale": 0})    # 
    cfg.update({"batch_size": 512})

    # exp -- batchsize
    # batchsize = 32
    # gpu_list = [2]
    # batchsize = 64
    # gpu_list = [3]
    # # batchsize = 128
    # # gpu_list = [5]
    # # batchsize = 256
    # # gpu_list = [6]
    # # batchsize = 512
    # # gpu_list = [7]
    # exp_tag = f'batchsizex{batchsize}'
    # cfg.update({"exp_tag": exp_tag})
    # cfg.update({"batch_size": batchsize})
    # cfg.update({"output_folder": 'exps/batchsize'})
    # embedding_num_dim = [(512,   4),]

    # # exp -- keepdim
    # cfg.update({"exp_tag": 'keepdim'})
    # cfg.update({"output_folder": 'exps/exp_keepdim'})
    # embedding_num_dim = [
    #     (8192,   4),
    #     (4096,   8),
    #     (2048,  16),
    #     (1024,  32),
    #     ( 512,  64),
    #     ( 256, 128),
    # ]

    # exp -- finddim
    cfg.update({"exp_tag": 'finddim'})
    cfg.update({"output_folder": 'exps/exp_finddim'})
    gpu_list = [1]
    num = 256                           # 
    # num = 128
    embedding_num_dim = [
        (num,   2),
        # (num,   4),
        # (num,   8),
        # (num,  16),
        # (num,  32),
        # (num,  64),
        # (num, 128),
    ]

    # exp -- findnum
    """
    cfg.update({"exp_tag": 'findnum'})
    cfg.update({"output_folder": 'exps/exp_findnum'})
    embedding_num_dim = [
        (16, 4),
        # (24, 4),
        # (32, 4),
        # (64, 4),
        # (128, 4),
        (256, 4),
        # (512, 4),
        # (1024, 4),
    ]"""
    main_exp_base(cfg, embedding_num_dim, gpu_list, merge_sh, flag_debug)


def main_exp_cifar10():
    cfg = dict()
    cfg.update({"dataset": 'cifar10'})
    flag_debug = False                   # True for debug, False for exp
    merge_sh = True                    # 
    gpu_list = [0]                      # 
    cfg.update({"shuffle_scale": 0})    #
    cfg.update({"batch_size": 512})     #

    # exp -- finddim
    cfg.update({"exp_tag": 'finddim'})
    cfg.update({"output_folder": 'exps/exp_finddim'})
    num = 128                           # 
    # num = 256
    embedding_num_dim = [
        (num,   4),
        (num,   8),
        (num,  16),
        (num,  32),
        (num,  64),
        (num, 128),
    ]

    # exp -- findnum
    """
    cfg.update({"exp_tag": 'findnum'})
    cfg.update({"output_folder": 'exps/exp_findnum'})
    # embedding_num_dim = [
    #     # (64, 4),
    #     # (128, 4),
    #     (256, 4),
    #     # (512, 4),
    #     # (1024, 4),
    # ]
    """

    main_exp_base(cfg, embedding_num_dim, gpu_list, merge_sh, flag_debug)


def main_exp_cifar10_bk():
    flag_debug = True  # True for debug, False for exp
    cfg = dict()
    gpu_list = [0]
    dataset_name = 'cifar10'
    cfg.update({"exp_tag": 'randomsample'})
    cfg.update({"output_folder": 'exps/exp_randomsample'})
    cfg.update({"shuffle_scale": 0})
    cfg.update({"batch_size": 512})
    embedding_num_dim = [
        # (64, 4),
        # (128, 4),
        (256, 4),
        # (512, 4),
        # (1024, 4),
    ]

    output_folder = cfg.get('output_folder')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=False)

    
    # --- default setting
    cfg.update({
        "distance": 'cos',              # distance, 默认为'cos'
        "anchor": 'random',            # sample 策略, 默认为'closest', random
        "split_type": 'fixed',          # split type: fixed, interval, random
        "dataset": dataset_name,
        "data_folder": get_datapath(dataset_name),
        })
    # -----
    cnt = 0
    for n, d in embedding_num_dim:
        gpu_id = gpu_list[cnt]
        cfg.update({"gpu_id": gpu_id})
        cfg.update({"embedding_num":n, "embedding_dim":d})
        exp_name = get_yaml(cfg, flag_debug)
        get_sh(cfg, exp_name)
        cnt += 1


def main_exp_fashion_mnist():
    cfg = dict()
    cfg.update({"dataset": 'fashion-mnist'})
    flag_debug = False                   # True for debug, False for exp
    merge_sh = False                    # 
    gpu_list = [1, 4, 7, 1, 4, 7]       #
    cfg.update({"shuffle_scale": 0})
    cfg.update({"batch_size": 512})

    # exp -- finddim
    cfg.update({"exp_tag": 'finddim'})
    cfg.update({"output_folder": 'exps/exp_finddim'})
    num = 128                           # 
    # num = 256
    embedding_num_dim = [
        (num,   4),
        (num,   8),
        (num,  16),
        (num,  32),
        (num,  64),
        (num, 128),
    ]

    # exp -- findnum
    """
    cfg.update({"exp_tag": 'findnum'})
    cfg.update({"output_folder": 'exps/exp_findnum'})
    embedding_num_dim = [
        (64, 4),
        (128, 4),
        (256, 4),
        (512, 4),
        (1024, 4),
    ]
    """

    main_exp_base(cfg, embedding_num_dim, gpu_list, merge_sh, flag_debug)



def main_exp_fashion_mnist_del():
    embedding_num_dim = [
        (128, 8),
        (256, 8),
        (512, 8),
        (1024, 8),
        (2048, 8),
        (4096, 8),
    ]

    cfg = dict()

    # --- default setting
    cfg.update({
        "distance": 'cos',              # distance, 默认为'cos'
        "anchor": 'closest',            # sample 策略, 默认为'closest'
        "split_type": 'fixed',          # split type: fixed, interval, random

        # --- dataset Vislab 12
        "dataset": 'fashion-mnist',
        "data_folder": '/data2/common/cifar',
        "batch_size": 1024,

        # --- dataset Vislab 13
        #
        # "dataset": 'mnist',
        # "data_folder": '/home2/jieli/datasets/mnist',
        #
        # "dataset": 'fashion-mnist',
        # "data_folder": '/home2/jieli/datasets/fashion-mnist',
        })
    
    cfg.update({"exp_tag": 'SimLoss-with-diff-EmbNum'})

    gpu_id = 0

    # ---------
    # cfg.update({"gpu_id": gpu_id})
    # cfg.update({"embedding_num":4096, "embedding_dim":8})
    # print(list(cfg.keys()))
    # get_yaml(cfg)
    
    # -----
    for n, d in embedding_num_dim:
        cfg.update({"gpu_id": gpu_id})
        cfg.update({"embedding_num":n, "embedding_dim":d})
        exp_name = get_yaml(cfg)
        get_sh(cfg, exp_name)

        gpu_id += 1


def main_exp_imagenet():
    embedding_num_dim = [
        # (256, 2),
        # (512, 2),

        # (128, 4),
        # (256, 4),
        # (512, 4),

        # (128, 8),

        (256, 8),
        (512, 8),
    ]

    cfg = dict()

    # --- default setting
    cfg.update({
        "distance": 'cos',              # distance, 默认为'cos'
        "anchor": 'closest',            # sample 策略, 默认为'closest'
        "split_type": 'fixed',          # split type: fixed, interval, random
        "shuffle_scale": 2,

        # --- dataset Vislab 12
        "dataset": 'imagenet',
        "data_folder": '/data2/common/ImageNet',
        "batch_size": 128,

        })
    
    cfg.update({"exp_tag": 'shuffle'})

    gpu_id = 1

    # ---------
    # cfg.update({"gpu_id": gpu_id})
    # cfg.update({"embedding_num":4096, "embedding_dim":8})
    # print(list(cfg.keys()))
    # get_yaml(cfg)
    
    # -----
    for n, d in embedding_num_dim:
        cfg.update({"gpu_id": gpu_id})
        cfg.update({"embedding_num":n, "embedding_dim":d})
        exp_name = get_yaml(cfg)
        get_sh(cfg, exp_name)

        gpu_id += 1


def main_exp_ffhq():
    cfg = dict()
    embedding_num_dim = [
        # (256, 2),
        # (512, 2),
        # ( 64, 4),
        (128, 4),
        # (256, 4),
        # (512, 4),

        # (128, 8),

        # (256, 8),
        # (512, 8),
    ]


    # --- default setting
    shuffle_scale = 0
    cfg.update({
        "distance": 'cos',              # distance, 默认为'cos'
        "anchor": 'closest',            # sample 策略, 默认为'closest'
        "split_type": 'fixed',          # split type: fixed, interval, random
        "shuffle_scale": shuffle_scale,

        # --- dataset Vislab 12
        "dataset": 'ffhq',
        "data_folder": '/data2/common/ffhq/images1024x1024',
        "batch_size": 60,

        })
    
    cfg.update({"exp_tag": f'shufflex{shuffle_scale}'})

    gpu_id = 6

    # ---------
    # cfg.update({"gpu_id": gpu_id})
    # cfg.update({"embedding_num":4096, "embedding_dim":8})
    # print(list(cfg.keys()))
    # get_yaml(cfg)
    
    # -----
    for n, d in embedding_num_dim:
        cfg.update({"gpu_id": gpu_id})
        cfg.update({"embedding_num":n, "embedding_dim":d})
        exp_name = get_yaml(cfg)
        get_sh(cfg, exp_name)

        gpu_id += 1


if __name__ == '__main__':
    # main_exp_imagenet()
    # main_exp_ffhq()

    main_exp_mnist()
    # main_exp_cifar10()
    # main_exp_fashion_mnist()
