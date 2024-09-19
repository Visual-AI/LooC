import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from quantizer_zoo.VQ_VAE import VectorQuantizer

from quantizer_zoo.LoRC_VAE.quantise import VectorQuantizer
import pdb


class ProductQuantizer(nn.Module):
    def __init__(self, feature_channel, n_e, e_dim):
        super(ProductQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        # self.beta = beta

        # self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        

        # dim = 8           # dim of each codevector
        # num = 1024        # num of codevectors
        # num * dim = 1024 * 8 = num' * m * dim = 8 * 128 * 8

        # m = feature_channel 1024 / codevector_dim 8 = 128
        # num' = 8
        # m = 128
        m = feature_channel // self.e_dim
        self.pq_dict = {}
        for idx in range(m):
            self.pq_dict[f'{idx}'] = VectorQuantizer(n_e, e_dim, 0.25)
        self.pq_dict = nn.ModuleDict(self.pq_dict)

        print(f"---> ProductQuantizer: {m} distinct codebook init success.")

    def forward(self, z):
        # pdb.set_trace()
        bs, c, h, w = z.shape

        # reshape z -> (batch, height, width, channel)
        z = z.permute(0, 2, 3, 1).contiguous()

        m = c // self.e_dim
        z_sub = z.view(bs, h, w, -1, self.e_dim)

        z_vq_list = []
        loss_list = []
        min_encoding_indices_list = []
        bin_count_list = []
        
        for idx in range(m):
           z_sub_i = z_sub[:, :, :, idx, :]
           z_vq_i, loss_i, (min_encoding_indices_i, bin_count_i) = self.pq_dict[f'{idx}'](z_sub_i)
           z_vq_list.append(z_vq_i)
           loss_list.append(loss_i)
           min_encoding_indices_list.append(min_encoding_indices_i)
           bin_count_list.append(bin_count_i)
        # pdb.set_trace()
        z_vq = torch.concat(z_vq_list, dim=3)
        loss = sum(loss_list)
        min_encoding_indices = torch.concat(min_encoding_indices_list, dim=0)
        bin_count = torch.stack(bin_count_list, dim=1).sum(dim=1)

        z_vq = z_vq.permute(0, 3, 1, 2).contiguous()  # bs, h, w, c --> bs, c, h, w
        loss = {"loss": loss}

        return z_vq, loss, (min_encoding_indices, bin_count)


