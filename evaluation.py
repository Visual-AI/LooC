import os
import lpips
import numpy as np
import argparse
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from tqdm import tqdm

from image_folder import make_dataset
from fid_score import calculate_fid_given_paths

# parser = argparse.ArgumentParser(description='Image quality evaluations on the dataset')
# # parser.add_argument('--gt_path', type=str, default='../results/', help='path to original gt data')
# # parser.add_argument('--g_path', type=str, default='../results/', help='path to the generated data')
# parser.add_argument('--num_test', type=int, default=0, help='how many examples to load for testing')
# # parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

# args = parser.parse_args()
lpips_vgg = lpips.LPIPS(net='vgg')

from config import load_config

cfg_all = load_config.load_cfg()

args = cfg_all

args.gt_path = os.path.join(cfg_all.output_folder, 'results', cfg_all.exp_name, 'original')
args.g_path = os.path.join(cfg_all.output_folder, 'results', cfg_all.exp_name, 'rec')

# args.gt_path = '/home/jieli/proj/VAE/exps/exp_1101/output/results/mnist_cos_closest_8192x4/best.pt/original'
# args.g_path = '/home/jieli/proj/VAE/exps/exp_1101/output/results/mnist_cos_closest_8192x4/best.pt/rec'

print('args.gt_path', args.gt_path)
print('args.g_path', args.g_path)

args.use_gpu = True if torch.cuda.is_available() else False

if (args.use_gpu):
	lpips_vgg.cuda()

def calculate_score(img_gt, img_test):
    """
    function to calculate the image quality score
    :param img_gt: original image
    :param img_test: generated image
    :return: mae, ssim, psnr
    """

    l1loss = np.mean(np.abs(img_gt-img_test))

    psnr_score = psnr(img_gt, img_test, data_range=1)

    ssim_score = ssim(img_gt, img_test, multichannel=True, data_range=1, 
                      win_size=11,
                      channel_axis=2,
                      )
    img0 = torch.from_numpy(img_gt).permute(2, 0, 1)
    img1 = torch.from_numpy(img_test).permute(2, 0, 1)
    if args.use_gpu:
        img0 = img0.cuda()
        img1 = img1.cuda()
    lpips_dis = lpips_vgg(img0, img1, normalize=True)

    if args.use_gpu:
        lpips_dis = lpips_dis.cpu().data.numpy().item()
    else:
        lpips_dis = lpips_dis.data.numpy().item()

    return l1loss, ssim_score, psnr_score, lpips_dis


if __name__ == '__main__':
    print("Make dataset ...")
    gt_paths, gt_size = make_dataset(args.gt_path)
    g_paths, g_size = make_dataset(args.g_path)

    print("calculate_fid ...")
    fid_score = calculate_fid_given_paths([args.gt_path, args.g_path], batch_size=50, cuda=True, dims=2048)

    l1losses = []
    ssims = []
    psnrs = []
    lpipses = []

    args.num_test = args.get("num_test", 0)  # set default 
    size = args.num_test if args.num_test > 0 else gt_size

    for i in tqdm(range(size)):
        gt_img = Image.open(gt_paths[i]).convert('RGB')
        gt_numpy = np.array(gt_img).astype(np.float32) / 255.0
        
        g_img = Image.open(g_paths[i]).convert('RGB')
        g_numpy = np.array(g_img).astype(np.float32) / 255.0
        l1loss, ssim_score, psnr_score, lpips_score = calculate_score(gt_numpy, g_numpy)
        # print(l1loss, ssim_score, psnr_score, lpips_score)
        l1losses.append(l1loss)
        ssims.append(ssim_score)
        psnrs.append(psnr_score)
        lpipses.append(lpips_score)

    print('{:>10},{:>10},{:>10},{:>10},{:>10}'.format('l1loss', 'SSIM↑', 'PSNR↑', 'LPIPS↓', 'FID-Score↓'))
    print('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.format(np.mean(l1losses), np.mean(ssims), np.mean(psnrs), np.mean(lpipses), np.mean(fid_score)))
    print('{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.format(np.var(l1losses), np.var(ssims), np.var(psnrs), np.var(lpipses)))