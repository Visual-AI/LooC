import os, random
import numpy as np
import datetime
# from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import  make_grid

import wandb
from tqdm import tqdm
# from tqdm.notebook import tqdm

import torch.utils.data as data_utils
from icecream import ic 
from modules import Model
from dataset import ffhq

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(data_loader, model, optimizer, run_steps, data_variance=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """trianing the model"""
    for images, _ in data_loader:
        # ic('train->images.shape: ', images.shape)
        images = images.to(device)
        optimizer.zero_grad()
        # x, loss_vq, perplexity, _ = model(images)
        x, loss_vq_dict, _, _ = model(images)
        loss_vq = loss_vq_dict.get('loss')
        # ic('train->x.shape: ', x.shape)

        # loss function
        loss_recons = F.mse_loss(x, images) / data_variance
        loss = loss_recons + loss_vq
        loss.backward()

        if Enable_Wandb:
            wandb.log({
                "loss_recons": loss_recons, 
                "loss_vq": loss_vq,
                "uniform_loss": loss_vq_dict.get('uniform_loss'), 
                "leaky_uniform_loss": loss_vq_dict.get('leaky_uniform_loss'), 
                # "loss_sim": loss_vq_dict.get('sim_loss'), 
                # 'sim_max': loss_vq_dict.get('sim_max'),
                # 'sim_row_max_val': loss_vq_dict.get('sim_row_max_val'),
                })

        # writer.add_scalar('loss/train/perplexity', perplexity.item(), args.steps)

        optimizer.step()

        # args.steps +=1
        run_steps += 1
        # ic(run_steps)


def test(data_loader, model, run_steps):
    """evaluation model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for images, _ in data_loader:
            images = images.to(device)
            x, loss_dict, _, _ = model(images)
            loss_recons += F.mse_loss(x, images)
            loss_vq += loss_dict.get('loss')
        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)
    if Enable_Wandb:
        wandb.log({"test_loss_reconstruction": loss_recons,
                "test_loss_quantization": loss_vq,
                #    "test_loss_sim": loss_dict.get('sim_loss'),
                #    'test_sim_max': loss_dict.get('sim_max'),
                #    'test_sim_row_max_val': loss_dict.get('sim_row_max_val'),
                },
                #    step = run_steps
                )

    return loss_recons.item(), loss_vq.item()


def generate_samples(images, model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        images = images.to(device)
        x, _, _, _ = model(images)
    return x


def main(args):
    # writer = SummaryWriter(os.path.join(os.path.join(args.output_folder, 'logs'), args.exp_name))
    save_filename = os.path.join(os.path.join(args.output_folder, 'models'), args.exp_name)
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    data_variance=1
    if args.dataset in ['mnist', 'fashion-mnist', 'cifar10', 'celeba', 'imagenet', 'ffhq', 'expINrec']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        if args.dataset == 'mnist':
            # Define the train & test datasets
            train_dataset = datasets.MNIST(args.data_folder, train=True,
                download=True, transform=transform)
            test_dataset = datasets.MNIST(args.data_folder, train=False,
                download=True, transform=transform)
            data_variance=np.var(train_dataset.data.numpy() / 255.0)
            num_channels = 1
        elif args.dataset == 'fashion-mnist':
            # Define the train & test datasets
            train_dataset = datasets.FashionMNIST(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(args.data_folder,
                train=False, download=True, transform=transform)
            data_variance=np.var(train_dataset.data.numpy() / 255.0)
            num_channels = 1
        elif args.dataset == 'cifar10':
            # Define the train & test datasets
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
            train_dataset = datasets.CIFAR10(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(args.data_folder,
                train=False, download=True, transform=transform)
            data_variance=np.var(train_dataset.data / 255.0)
            num_channels = 3
        elif args.dataset == 'celeba':
            # Define the train & test datasets
            transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
            train_dataset = datasets.CelebA(args.data_folder,
                split='train', download=True, transform=transform)
            test_dataset = datasets.CelebA(args.data_folder,
                split='valid', download=True, transform=transform)
            # print(f"len(train_dataset) = {len(train_dataset)}")
            # train_list = []
            # for i in range(len(train_dataset)):
            #     print(i, train_dataset[i][0])
            #     train_list.append(train_dataset[i][0])
            num_channels = 3
        elif args.dataset == 'imagenet':  # imagenet
            print("Loading imagenet")
            transform = transforms.Compose([
            transforms.Resize([256, 256]),  # TODO size = ?
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            train_dataset = datasets.ImageNet(args.data_folder,
                split='val', transform=transform)  # TODO: train
            test_dataset = datasets.ImageNet(args.data_folder,
                split='val', transform=transform)  # 50k
            num_channels = 3
 
            train_dataset = data_utils.Subset(train_dataset, torch.arange(10000))  # 10k
            test_dataset  = data_utils.Subset( test_dataset, torch.arange(1000))   #  1k

            print("len(train_dataset)", len(train_dataset))
            print("len(test_dataset)", len(test_dataset))

        # 仅用于调试，FFHQ的实验在VQGAN的框架下进行
        elif args.dataset == 'ffhq':
            print("Loading ffhq")
            transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = ffhq.ImagesFolder(args.data_folder,
                split='val', transform=transform)  # 60k TODO: train
            test_dataset = ffhq.ImagesFolder(args.data_folder,
                split='val', transform=transform)  # 10k
            num_channels = 3
            # ----> 截取dataset的子集
            train_dataset = data_utils.Subset(train_dataset, torch.arange(1000))  # 1k
            test_dataset  = data_utils.Subset( test_dataset, torch.arange(256))   

            print("len(train_dataset)", len(train_dataset))
            print("len(test_dataset)", len(test_dataset))
        elif args.dataset in ['expINrec']:
            print("Loading folder", args.data_folder)
            transform = transforms.Compose([
            # transforms.Resize((256,256)),
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = ffhq.ImagesFolder(args.data_folder,
                split='all', transform=transform)  
            test_dataset = ffhq.ImagesFolder(args.data_folder,
                split='all', transform=transform)  
            num_channels = 3

            print("len(train_dataset)", len(train_dataset))
            print("len(test_dataset)", len(test_dataset))
        # thumbnails128x128

        valid_dataset = test_dataset
    else:
        raise ValueError(f"dataset={args.dataset} not implemented")
    # Define the dataloaders
    print("Define the dataloaders")
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, 
        worker_init_fn=seed_worker, generator=g)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False, drop_last=True, # 
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=g)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=32, shuffle=False,
        worker_init_fn=seed_worker, generator=g)
    
    # Define the model
    print("Define the model")
    print("VQ =", args.get('vq', 'lorc_old'))
    model = Model(num_channels, args.hidden_size, args.num_residual_layers, args.num_residual_hidden,
                  args.num_embedding, args.dim_embedding, args.f, args.commitment_cost, args.distance,
                  args.anchor, 
                  first_batch = False,  #   args.first_batch, 
                  contras_loss= False,  #   args.contras_loss,
                  # lora_codebook=args.lora_codebook,  # TODO config 改为更加合适的名字， slice_codebook?
                  # evq=args.evq,
                  split_type=args.split_type,
                  args=args,
                  ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Update the model
    print("Update the model")
    best_loss = -1.
    run_step = 0
    for epoch in tqdm(range(args.num_epochs)):
        # training and testing the model
        # print(f"Epoch: {epoch}")
        train(train_loader, model, optimizer, run_step, data_variance)

        # TODO: 加速训练，可把这部分去掉
        loss_rec, loss_vq = test(valid_loader, model, run_step)

        # TODO: 加速训练，可把这部分去掉
        # visualization
        # images, _ = next(iter(test_loader))
        # rec_images = generate_samples(images, model, args)
        # input_grid = make_grid(images, nrow=8, value_range=(-1, 1), normalize=True)  # range -> value_range
        # rec_grid = make_grid(rec_images, nrow=8, value_range=(-1,1), normalize=True)
        # writer.add_image('original', input_grid, epoch + 1)
        # writer.add_image('reconstruction', rec_grid, epoch + 1)

        # save model
        if (epoch == 0) or (loss_rec < best_loss):
            best_loss = loss_rec
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)
        # only save the last epoch
        if epoch+1 == args.num_epochs:
            with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
                torch.save(model.state_dict(), f)


if __name__ == '__main__':
    time_start=datetime.datetime.now()

    from config import load_config

    cfg_all = load_config.load_cfg()

    if not os.path.exists(os.path.join(cfg_all.output_folder, 'logs')):
        os.makedirs(os.path.join(cfg_all.output_folder, 'logs'))
    if not os.path.exists(os.path.join(cfg_all.output_folder, 'models')):
        os.makedirs(os.path.join(cfg_all.output_folder, 'models'))
    if not os.path.exists(os.path.join(cfg_all.output_folder, 'models', cfg_all.exp_name)):
        os.makedirs(os.path.join(cfg_all.output_folder, 'models', cfg_all.exp_name))
    # Device
    # cfg_all.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Slurm
    # if 'SLURM_JOB_ID' in os.environ:
    #     args.exp_name += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    # if not os.path.exists(os.path.join(os.path.join(args.output_folder, 'models'), args.exp_name)):
    #     os.makedirs(os.path.join(os.path.join(args.output_folder, 'models'), args.exp_name))
    
    Enable_Wandb = cfg_all.get('Enable_Wandb', True)
    f = cfg_all.get('f', 4)
    cfg_all["f"] = f
    print('f =', cfg_all.get('f'))

    # Enable_Wandb = False  # debug
    if Enable_Wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=f"proj_EfficientCodebook_{cfg_all.get('dataset')}",
            name=cfg_all.get('exp_name'),  # display name for this run

            # track hyperparameters and run metadata
            config=dict(cfg_all)
            # config={
            # "learning_rate": 0.02,
            # "architecture": "CNN",
            # "dataset": "CIFAR-100",
            # "epochs": 10,
            # }
        )

    print("# " * 20)

    main(cfg_all)
    if Enable_Wandb:
        wandb.finish()

    time_end=datetime.datetime.now()
    print('Running time: %s'%(time_end - time_start))