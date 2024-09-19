import argparse

from omegaconf import OmegaConf
import pdb


def parse_args():
    parser = argparse.ArgumentParser(description='Improved VQ with new codebook')
    parser.add_argument('--cfg', type=str, default='config/fashion-mnist.yaml')

    # exp setting. if not set, use the setting form the yaml defined by --cfg
    #nargs="*" 一个参数后面可以跟随任意多的值，会放到列表中
    parser.add_argument("--loss", nargs="*", help="list of loss function name")
    # parser.add_argument('--exp_name', type=str, default='vqvae', help='name of the output folder (default: vqvae)')

    # General
    # parser.add_argument('--data_folder', type=str, help='name of the data folder')
    # parser.add_argument('--dataset', type=str, help='name of the dataset (mnist, fashion-mnist, cifar10)')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size (default: 1024)')
    # # Latent space
    # parser.add_argument('--hidden_size', type=int, default=128, help='size of the latent vectors (default: 128)')
    # parser.add_argument('--num_residual_hidden', type=int, default=32, help='size of the redisual layers (default: 32)')
    # parser.add_argument('--num_residual_layers', type=int, default=2, help='number of residual layers (default: 2)')
    # # Quantiser parameters
    # parser.add_argument('--embedding_dim', type=int, default=64, help='dimention of codebook (default: 64)')
    # parser.add_argument('--num_embedding', type=int, default=512, help='number of codebook (default: 512)')
    # parser.add_argument('--commitment_cost', type=float, default=0.25, help='hyperparameter for the commitment loss')
    # parser.add_argument('--distance', type=str, default='cos', help='distance for codevectors and features')
    # parser.add_argument('--anchor', type=str, default='closest', help='anchor sampling methods (random, closest, probrandom)')
    # parser.add_argument('--split_type', type=str, default='fixed', help='split methods (fixed, interval, random)')
    # parser.add_argument('--first_batch', action='store_true', help='offline version with only one time reinitialisation')
    # parser.add_argument('--contras_loss', action='store_true', help='using contrastive loss')
    # parser.add_argument('--lora_codebook', action='store_true', help='using lora_codebook')
    # parser.add_argument('--evq', action='store_true', help='using EfficientVectorQuantiser')
    # parser.add_argument('--scale_grad_by_freq', action='store_true', help='using scale_grad_by_freq in the codebook embedding')
    
    # # Optimization
    # parser.add_argument('--seed', type=int, default=42, help="seed for everything")
    # parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs (default: 100)')
    # parser.add_argument('--lr', type=float, default=3e-4, help='learning rate for Adam optimizer (default: 2e-4)')
    # # Miscellaneous
    # parser.add_argument('--output_folder', type=str, default='./', help='name of the output folder (default: vqvae)')
    # parser.add_argument('--exp_name', type=str, default='vqvae', help='name of the output folder (default: vqvae)')
    # parser.add_argument('--num_workers', type=int, default=mp.cpu_count() - 1, help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    # parser.add_argument('--device', type=str, default='cpu', help='set the device (cpu or cuda, default: cpu)')

    return parser.parse_args()

def load_cfg():
    raw_args = parse_args()

    # Create config from multiple sources.
    cfg_yaml = OmegaConf.load(raw_args.cfg)         # From a YAML file
    cfg_args = OmegaConf.create(vars(raw_args))     # From parser
    cfg_all = OmegaConf.merge(cfg_yaml, cfg_args)  # Later arguments override earlier ones

    # 启用parser中设置的loss
    if raw_args.loss is not None:  # test 时为None
        for x in raw_args.loss:
            cfg_all.update({x: True})

    print(OmegaConf.to_yaml(cfg_all))
    return cfg_all
