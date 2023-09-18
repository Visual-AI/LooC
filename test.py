import os
import argparse
import multiprocessing as mp
import numpy as np
from sklearn.manifold import TSNE
import cv2

from skimage.transform import resize
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
from PIL import Image

import torch
from torchvision import transforms, datasets

from modules import Model
from util import tensor2im, save_image
from tqdm import tqdm
from icecream import ic 


# 对get_cmap稍微整理下
def get_colors(name, lut):
    """params:
        - name：颜色图谱，可以是字符串，也可以是colormap实例
        - lut：要得到的颜色个数，一个整数
    """
    return plt.get_cmap(name, lut)([i for i in range(lut)])



def count_each_codebook_entry_matched_times_in_different_chunk(bincounts, fig_id=1, fig_path=None):
    """将feature vector 划分为多个slice之后, 各feature上相同slice的vector组成chunk。
    统计codebook entry 在不同chunk中被匹配到的次数.
    # bincounts: list
    """
    chunk_num = len(bincounts)
    entry_num = bincounts[0].shape[0]

    bincounts_tensor = torch.stack(bincounts, dim=1)

    match_flag = torch.where(bincounts_tensor >= 1, 1, 0)  # 第i个 codebook entry 在第j个chunk 内被匹配次数>1, 则res[i, j]标记为1, 否则为0
    match_num = torch.sum(match_flag, dim=1)  # 统计每个codebook entry在多少个chunk上被匹配，即一个entry被多少个chunk使用了

    match_rate = torch.bincount(match_num, minlength=chunk_num + 1)  # 统计匹配次数的频率, Returns a tensor of shape Size([max(input) + 1])
    # miss_num = chunk_num - torch.numel(match_rate)
    # match_rate = F.pad(match_rate, [0, miss_num], "constant", 0)
    print("match_rate (int)", match_rate)

    match_rate = match_rate / entry_num * 100.  # 转为百分比

    torch.set_printoptions(precision=4, sci_mode=False)
    print("match_rate (%)", match_rate)

    if fig_path:
        plt.figure(fig_id)
        x = range(0, match_rate.shape[0], 1)  # [0, chunk_num]  # match_rate.shape[0] == chunk_num + 1
        y = match_rate.cpu().numpy()
        plt.bar(x, y)
        plt.xlabel("Num of Thunks")
        plt.ylabel("Percentage of Codebook used in diff thunks. (%)")
        plt.savefig(fig_path)
        print(fig_path)


def plot_bincounts(bincounts, fig_id, bincount_fig_path):
    print("plot bincounts")
    plt.figure(fig_id, figsize=(10, 6))
    if not isinstance(bincounts, list):
        # ic("not isinstance(bincounts, list)", type(bincounts))
        bincounts = [bincounts]

    # bin_num = 100  # debug
    bin_num = bincounts[0].shape[0]
    bottom_cnt = np.zeros(bin_num, dtype=np.int32)
    for idx, bc in enumerate(bincounts):
        # bc = bc[:bin_num]  # debug 截取一部分用于分析
        # ic(idx, bc)
        x = range(1, bc.shape[0]+1, 1)
        y = bc.cpu().numpy().astype(np.int32)
        plt.bar(x, y, bottom=bottom_cnt, label=f'slice_{idx+1}')
        bottom_cnt += y
        plt.legend()  # 显示图例

    plt.ylabel("Count Number")
    plt.xlabel("Codebook Index")
    plt.savefig(bincount_fig_path)
    ic(bincount_fig_path)


def calc_usage_and_perplexity(model,
                              bincounts, 
                              bincount_fig_path=None, 
                              sort_bincount_fig_path=None, 
                              codebook_distribution_fig_path=None,
                              fig_id=1):
    # ----debug: bincounts截取部分，进行分析
    # bincounts = bincounts[:30]  # debug

    # ---- 使用bincount来获取usage和perplexity
    print("### calc from bincounts")
    usage = torch.count_nonzero(bincounts) / torch.numel(bincounts) * 100
    print("usage of the codebook vector: {}%".format(usage))

    total_count = torch.sum(bincounts)
    avg_probs = bincounts / total_count
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    print("the perplexity of the codebook: {}".format(perplexity))

    if bincount_fig_path:
        print("plot bincounts")
        plt.figure(fig_id+1)
        # plt.plot(bincounts.cpu().numpy().astype(np.int32))

        x = range(1, bincounts.shape[0]+1, 1)
        plt.bar(x, bincounts.cpu().numpy().astype(np.int32))

        plt.ylabel("Count Number")
        plt.xlabel("Codebook Index")
        plt.savefig(bincount_fig_path)
        ic(bincount_fig_path)        

    if sort_bincount_fig_path:
        print("plot sorted bincounts")
        sort_count1 = -np.sort(-bincounts.cpu().numpy().astype(np.float32))
        plt.figure(fig_id+2)
        plt.plot(sort_count1)
        plt.ylabel("Count Number")
        plt.xlabel("Codebook Index")
        plt.savefig(sort_bincount_fig_path)
        ic(sort_bincount_fig_path)

    ########################################
    """
    print("### calc from encodings")
    # calculate the perplexity in whole test images
    encodings = torch.cat(encodings, dim=0)
    # save the codebook count
    count = torch.sum(encodings, dim=0).cpu().numpy()
    usage = (1 - len(count[count==0])/len(count)) * 100.
    print("usage of the codebook vector: {}%".format(usage))

    avg_probs = torch.mean(encodings, dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    print("the perplexity of the codebook: {}".format(perplexity))

    # save the sorted codebook count
    sort_count = -np.sort(-count)

    fig_id += 1
    plt.figure(fig_id)
    plt.plot(count)
    plt.ylabel("Count Number")
    plt.xlabel("Vocabulary Index")
    plt.savefig(os.path.join(results_path, 'validation_en.png'))

    fig_id += 1
    plt.figure(fig_id)
    plt.plot(sort_count)
    plt.ylabel("Count Number")
    plt.xlabel("Vocabulary Index")
    plt.savefig(os.path.join(results_path,'csort_validation_en.png'))

    """

    ########################################
    # visualize codebook
    if codebook_distribution_fig_path:
        print("fit TSNE")
        count = bincounts.cpu().numpy()
        code_book = model._vq_vae.embedding.weight.data.cpu()
        tsne = TSNE(n_components=2, perplexity=5, n_iter=5000, verbose=True)
        projections = tsne.fit_transform(code_book)
        print("plot codebook distribution")
        plt.figure(fig_id+3)
        plt.rcParams['xtick.bottom'] = False
        plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['ytick.left'] = False
        plt.rcParams['ytick.labelleft'] = False
        plt.scatter(*projections[count==0].T,label=str(0),color=plt.cm.Dark2(0),alpha=0.425,zorder=2,)
        plt.scatter(*projections[count>0].T,label=str(1),color=plt.cm.Dark2(1),alpha=0.425,zorder=2,)
        plt.savefig(codebook_distribution_fig_path, dpi=300)
        ic(codebook_distribution_fig_path)


def vis_match(slice_num, codebook_colors, match_idx, fig_path, ori_img=None, rec_img=None):
    """
    1. 首先，我们将codebook的每一列都视为一个“词”，并将其用颜色编码。具体来说，我们将每个词
    用HSV颜色空间中的色相表示，从0到360度，每个度数代表一个不同的词。饱和度和亮度都将设置为1，
    以确保颜色的清晰度。

    2. 然后，我们将feature map的每个像素与最接近的“词”匹配，并将其用相应的颜色填充。这样，
    我们可以看到哪些像素与哪些“词”匹配，以及它们之间的关系。
    """

    matching_colors_list = []
    if ori_img is not None:
        matching_colors_list.append(np.pad(ori_img, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=255))
        img_h, img_w = ori_img.shape[:2]
        # print("1 img_h, img_w", img_h, img_w)

    elif rec_img is not None:
        matching_colors_list.append(np.pad(rec_img, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=255))
        img_h, img_w = rec_img.shape[:2]
        # print("2 img_h, img_w", img_h, img_w)
    else:
        img_h, img_w = 8, 8  # TODO

    # 将每个feature map与codebook的匹配情况，绘制可视化图像，并将其垂直拼接
    # fig, axs = plt.subplots(nrows=8, ncols=1, figsize=(8, 16))
    # import pdb
    # pdb.set_trace()
    for s in range(slice_num):
        matching_colors = np.zeros((8, 8, 3))  # feature map resolution = [8, 8], output color image channel = 3
        for i in range(8):
            for j in range(8):
             matching_colors[i, j, :] = codebook_colors[match_idx[i, j, s], :]
        # 
        # import pdb
        # pdb.set_trace()
        # print('1 max num=', np.max(matching_colors))
        matching_colors_img = (matching_colors * 255.0).astype('uint8')
        # print('2 max num=', np.max(matching_colors_img))

        matching_colors_img = cv2.resize(matching_colors_img, dsize=(img_h, img_w), interpolation=cv2.INTER_CUBIC)
        # matching_colors_img = resize(matching_colors_img, (img_h, img_w))   # FIX: resize之后，最大值从255变成1，原因未知
        # print('3 max num=', np.max(matching_colors_img))

        # print('before pad', matching_colors_img.shape)
        matching_colors_img = np.pad(matching_colors_img, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=255)
        # print('after pad', matching_colors_img.shape)
        matching_colors_list.append(matching_colors_img)
        # 
        # axs[s].imshow(matching_colors)
        # axs[s].set_xticks([])
        # axs[s].set_yticks([])

    #
    matching_colors_array = np.stack(matching_colors_list)
    matching_colors_array = np.concatenate(matching_colors_list, axis=1)  # [h, w*N, channel], N=8

    # Lossy conversion from float64 to uint8. Range [-1.8612588436123182e-06, 1.0]
    # matching_colors_array = (matching_colors_array * 255).astype('uint8')
    save_image(matching_colors_array, fig_path)
    #
    # plt.subplots_adjust(hspace=0.05, wspace=0.05)
    # plt.savefig(fig_path)



def main(args):
    # load dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
        ])
    if args.dataset == 'mnist':
        # Define the train & test datasets
        test_dataset = datasets.MNIST(args.data_folder, train=False, download=True, transform=transform)
        num_channels = 1
    elif args.dataset == 'fashion-mnist':
        # Define the train & test datasets
        test_dataset = datasets.FashionMNIST(args.data_folder, train=False, download=True, transform=transform)
        num_channels = 1
    elif args.dataset == 'cifar10':
        # Define the train & test datasets
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = datasets.CIFAR10(args.data_folder, train=False, download=True, transform=transform)
        num_channels = 3
    elif args.dataset == 'celeba':
        # Define the train & test datasets
        transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = datasets.CelebA(args.data_folder, split='valid', download=True, transform=transform)
        num_channels = 3

    else:
        raise NotImplementedError

    # Define the dataloaders
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Define the model
    model = Model(num_channels, args.hidden_size, args.num_residual_layers, args.num_residual_hidden,
                      args.num_embedding, args.embedding_dim, distance=args.distance,
                      lora_codebook=args.lora_codebook,
                      evq=args.evq,
                      slice_num=args.slice_num,  # TODO 去除这个参数，这个参数的含义很容易混淆。
                      split_type=args.split_type,
                      args=args,
                      ) 

    # load model
    # ckpt = torch.load(os.path.join(os.path.join(os.path.join(args.output_folder, 'models'), args.model_name)))
    if '/models/' in args.model_name:
        model_path = args.model_name
    else:
        model_path = os.path.join(os.path.join(os.path.join(args.output_folder, 'models'), args.model_name))
    print("Load model from:", model_path)
    ckpt = torch.load(model_path)

    model.load_state_dict(ckpt)
    model = model.to(args.device)
    model.eval()

    # store results
    results_path = os.path.join(os.path.join(args.output_folder, 'results'), args.model_name)
    original_path = os.path.join(results_path, 'original')
    vis_path = os.path.join(results_path, 'vis')
    rec_path = os.path.join(results_path, 'rec')

    print(f"original_path: {original_path}")
    print(f"rec_path: {rec_path}")
    print(f"vis_path: {vis_path}")

    os.makedirs(original_path, exist_ok=True)
    os.makedirs(rec_path, exist_ok=True)
    os.makedirs(vis_path, exist_ok=True)

    # 将codebook的每个“entry”编码为颜色
    # hues = np.linspace(0, 360, args.num_embedding, endpoint=False)
    # codebook_colors = colors.hsv_to_rgb(np.column_stack((hues, np.ones(args.num_embedding), np.ones(args.num_embedding))))
    # args.num_embedding = 256
    codebook_colors = get_colors(name=None, lut=args.num_embedding)  # args.num_embedding x 4 
    codebook_colors = codebook_colors[:, 0:3]

    # 将codebook的编码颜色画出来
    _tmp_col = 1  # 列数
    codebook_img = np.reshape(codebook_colors, (args.num_embedding // _tmp_col, _tmp_col, 3)) # h, w, 3
    codebook_img = np.repeat(codebook_img, 32, axis=1)
    # codebook_img = np.transpose(codebook_img, (1, 0, 2))
    codebook_img = (codebook_img * 255.0).astype('uint8')
    save_image(codebook_img, 'codebook_vis.jpg')
    # exit()

    # test model
    encodings = []
    # indexes = []
    # labels = []
    # all_images = []
    # bincounts = None
    imageid = 0

    # slice_num = 1
    slice_num = args.hidden_size // args.embedding_dim
    ic(slice_num)

    model_output_list = [
        'image', 
        'bincounts_list',
        ]

    bincounts_list = [None] * slice_num

    debug_max = 20
    debug_cnt = 0
    for images, label in tqdm(test_loader):
        images = images.to(args.device)
        x_recons, loss, encoding_indices, bincount = model(images)  # TODO  # x_recon, loss, perplexity, encodings, bincount
        # -- save indexes
        # index = encoding.argmax(dim=1).view(images.size(0), -1)
        # indexes.append(index)

        # all_images.append(images.view(images.size(0), -1))
        # -- save labels
        # labels.append(label)
        # bincounts.append(bincount)

        # bincount = bincount.reshape(-1, slice_num)
        # if bincounts is None:
        #     bincounts = bincount
        # else:
        #     bincounts += bincount

        # -- save encodings
        # encodings.append(encoding)        # TODO: 这一步会造成内存溢出

        if 'bincounts_list' in model_output_list:
            # -- 根据slice 将bincount进行切分，分别进行分析
            # bincounts_list --> 改为 二维tensor， dim=1对应不同的slice
            encoding_indices_for_all_slice = encoding_indices.reshape(-1, slice_num)
            embed_num = bincount.shape[0]
            for i_n in range(slice_num):
                i_bc = torch.bincount(encoding_indices_for_all_slice[:, i_n], minlength=embed_num)  # bincount 来获得不同codebook的出现频率
                # if torch.numel(i_bc) < embed_num:
                #     miss_num = embed_num - torch.numel(i_bc)
                #     i_bc = F.pad(i_bc, [0, miss_num], "constant", 0)
                assert embed_num == torch.numel(i_bc)

                if bincounts_list[i_n] is None:
                    bincounts_list[i_n] = i_bc
                else:
                    bincounts_list[i_n] += i_bc

        if "image" in model_output_list:
            batch_size = images.shape[0]
            fm_h = int(math.sqrt( torch.numel(encoding_indices) // slice_num // batch_size ))  # height of feature map
            fm_w = fm_h                                                                        # width of feature map
            encoding_index = encoding_indices.reshape(batch_size, fm_h, fm_w, slice_num)  # [bs, h, w, slice_num]
            # -- save image
            for x_recon, image, idx in zip(x_recons, images, encoding_index):
                x_recon = tensor2im(x_recon)
                image = tensor2im(image)
                name = str(imageid).zfill(8) + '.jpg'
                save_image(image, os.path.join(original_path, name))
                save_image(x_recon, os.path.join(rec_path, name))
                imageid += 1

                vis_feature = False
                vis_feature = True
                if vis_feature:
                    # TODO 不同的slice分别展示
                    match_idx = idx  # 1024x8, bs=16, [h, w, slice_num]
                    fig_path = os.path.join(vis_path, name) 
                    vis_match(slice_num, codebook_colors, match_idx, fig_path, image, x_recon)
                if imageid >= debug_max:
                    break
                
        if imageid >= debug_max:
            break

    # idx = 0
    # calc_usage_and_perplexity(
    #     bincounts_list[idx],
    #     fig_id=1,
    #     )

    # plot_bincounts(bincounts_list, fig_id=1, bincount_fig_path=os.path.join(results_path, f'bincount_slice{slice_num}-all.png'))


    count_each_codebook_entry_matched_times_in_different_chunk(
        bincounts_list,
        fig_id=1, 
        fig_path=os.path.join(results_path,f'codebook_distribution_slice{slice_num}-usage.png')
        )
    
    exit(0)

    # 使用不同的fid_id,将每次绘图结果分别保存
    for idx in range(slice_num):
        calc_usage_and_perplexity(
            model,
            bincounts_list[idx],
            bincount_fig_path=os.path.join(results_path, f'bincount_slice{slice_num}-{idx+1}.png'),
            # sort_bincount_fig_path=os.path.join(results_path, f'sort_bincount_slice{slice_num}-{idx+1}.png'),
            # codebook_distribution_fig_path=os.path.join(results_path,f'codebook_distribution_slice{slice_num}-{idx+1}.png'),
            fig_id=1 + idx * 5,
            )

    # 使用同一个fid_id,将多次绘图结果合并到一张图
    for idx in range(1, slice_num):
        calc_usage_and_perplexity(
            model,
            bincounts_list[idx],
            bincount_fig_path=os.path.join(results_path, f'bincount_slice{slice_num}-all.png'),
            # sort_bincount_fig_path=os.path.join(results_path, f'sort_bincount_slice{slice_num}-all.png'),
            # codebook_distribution_fig_path=os.path.join(results_path,f'codebook_distribution_slice{slice_num}-all.png'),
            fig_id=1,  
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CVQ-VAE')
    # General
    parser.add_argument('--data_folder', type=str, help='name of the data folder')
    parser.add_argument('--dataset', type=str, help='name of the dataset (mnist, fashion-mnist, cifar10)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size (default: 16)')
    # Latent space
    parser.add_argument('--hidden_size', type=int, default=128, help='size of the latent vectors (default: 128)')
    parser.add_argument('--num_residual_hidden', type=int, default=32, help='size of the redisual layers (default: 32)')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='number of residual layers (default: 2)')
    # Quantiser parameters
    parser.add_argument('--embedding_dim', type=int, default=64, help='dimention of codebook (default: 64)')
    parser.add_argument('--num_embedding', type=int, default=512, help='number of codebook (default: 512)')
    parser.add_argument('--slice_num', type=int, default=0, help='number of slice (default: 0)')
    parser.add_argument('--distance', type=str, default='cos', help='distance for codevectors and features')
    parser.add_argument('--lora_codebook', action='store_true', help='using lora_codebook')
    parser.add_argument('--evq', action='store_true', help='using EfficientVectorQuantiser')
    parser.add_argument('--scale_grad_by_freq', action='store_true', help='using scale_grad_by_freq in the codebook embedding')
    parser.add_argument('--split_type', type=str, default='fixed', help='split methods (fixed, interval, random)')

    # Miscellaneous
    parser.add_argument('--output_folder', type=str, default='/scratch/shared/beegfs/cxzheng/normcode/final_vqvae/', help='name of the output folder (default: vqvae)')
    parser.add_argument('--model_name', type=str, default='fashionmnist_probrandom_contramin1/best.pt', help='name of the output folder (default: vqvae)')
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count() - 1, help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cuda', help='set the device (cpu or cuda, default: cpu)')
    args = parser.parse_args()

    main(args)