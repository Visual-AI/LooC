import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange



class Codebook(nn.Module):
    """
    """
    def __init__(self, entry_num, entry_dim):
        super().__init__()

        # total number of 'codebook entries' in the code bank, the codebook is viewed as a flattened vector sequence.
        self.entry_num = entry_num    

        # dim of each 'codebook entry'
        self.entry_dim = entry_dim      

        self.embedding = nn.Embedding(self.entry_num, self.entry_dim)

        # TODO 控制不同方法的size一致
        self.size = self.entry_num * self.entry_dim  

        self.init()

    def init(self):
        self.embedding.weight.data.uniform_(-1.0 / self.entry_num, 1.0 / self.entry_num)


# VectorQuantizer with 2D Codebook
class VectorQuantizer_2DCB(nn.Module):
    def __init__(self, n_e, e_dim, e_num=None, beta=0.25, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=False):
        super().__init__()

        self.codebook = Codebook(entry_num=1024, entry_dim=64)
    
    def forward(self, z):
        b, c, h, w = z.size()
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        # c = self.embed_dim * N
        z_flattened = z.view(-1, self.embed_dim)
    
    





# ---- Aug 14, 2023
# Online Codebook (OCB)
class Codebook_bk(nn.Module):
    """
    """
    def __init__(self, emb_len, emb_dim, slice_num=8, beta=0.25):
        super().__init__()

        # only inplemented slice_num==8
        # TODO: [4, 8, 16, 32, 64]
        assert slice_num in [8], f"slice_num={slice_num} not in the list [8]."
        # total number of 'embedding code vector' in the code bank, the code bank is viewed as a flattened vector sequence.
        self.emb_len = emb_len    

        # dim of each 'embedding code vector'
        self.emb_dim = emb_dim      

        self.embedding = nn.Embedding(self.emb_len, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.emb_len, 1.0 / self.emb_len)
        # self.normalize(self.embedding, dim=1)

        # number of 'embedding code vector' within each feature vector. Each feature vector is sliced to several 'embedding code vector'.
        self.slice_num = slice_num    # TODO, 放到forward里面，可以自动计算
        # slice_num in [4, 8, 16, 64] ? 是否可以更通用一些？3 * 3？
        # 4 = 2 * 2; 16 = 4 * 4; 64 = 8 * 8
        # 8 = 2 * 2 * 2; 64 = 4 * 4 * 4;

        # # 校验slice_num的标志位，未校验时为None，如果校验成功则为True，失败为False
        # self.ck_enum = None  

        # 仅针对slice_num=8时：scale_factor = 2，且将2D feture map 拓展为3D feature map来处理
        # scale_factor = 2
        # self.encoder = nn.Sequential(
        #     # nn.LayerNorm(normalized_shape=in_embed_dim, eps=1e-6),
        #     # SelectElement(index=0),
        #     # nn.Linear(in_embed_dim, out_embed_dim, bias=False),

        #     # (N, C, D, H_in, W_in) --> (N, C, D*scale_factor, H_in*scale_factor, W_in*scale_factor)
        #     nn.Upsample(scale_factor=scale_factor, mode='trilinear'),  
        # )

        # TODO 在每次update后进行normalize 
        # TODO 防止匹配时来回跳变
        # TODO 将embedding的数据类型限制为int8 

        # beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        self.beta = beta   

        # regularization for embedding
        self.weight_decay = 0.01  # SGD的weight decay默认为e-5，此处设置了更大一些的值来增强约束
        self.ord = 1

        # in_embed_dim = 256
        # out_embed_dim = 256 * 8

        print(f"Codebank2D: init success.", 
              f"emb_len={self.emb_len}",
              f"emb_dim={self.emb_dim}",
              f"slice_num={self.slice_num}",
              f"beta={self.beta}",
              f"weight_decay={self.weight_decay}"
              )
        
    # def quant_uint8(self):  # 此时codebank全部可能值就固定为uint8的值域
    #     r = 0, 255

    # def quant_int8(self):
    #     r = -128, 127

    def pearson(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        # return cost
        return 0.5 + 0.5 * cost

    def normalize(emb, dim=1):
        emb.weight.data = F.normalize(emb.weight.data, p=2.0, dim=dim)

    def pixelshuffle3D_downscale(self, x, scale_factor):
        # example: when scale_factor = 2,
        # b, c, d, h, w --> b, c, d/2, 2, h/2, 2, w/2, 2
        #               --> b, c, 2, 2, 2, d/2, h/2, w/2
        #               --> b, c*8, d/2, h/2, w/2
        bs, c, d, h, w = x.size()
        x = x.view([bs, c, d//scale_factor, scale_factor, h//scale_factor, scale_factor, w//scale_factor, scale_factor])
        x = torch.permute(x, (0, 1, 3, 5, 7, 2, 4, 6))
        # x = x.view([bs, c*scale_factor**3, d//scale_factor, h//scale_factor, w//scale_factor])
        x = x.reshape([bs, c*scale_factor**3, d//scale_factor, h//scale_factor, w//scale_factor])
        return x
    
    def pixelshuffle3D_upscale(self, x, scale_factor):
        # example: when scale_factor = 2,
        # b, c, d, h, w --> b, c//8, 2, 2, 2, d, h, w
        #               --> b, c//8, 2, 2, 2, d, h, w
        #               --> b, c*8, d/2, h/2, w/2
        bs, c, d, h, w = x.size()
        x = x.view([bs, c//(scale_factor**3), scale_factor, scale_factor, scale_factor, d, h, w])
        x = torch.permute(x, (0, 1, 5, 2, 6, 3, 7, 4))
        x = x.reshape([bs, c//(scale_factor**3), d*scale_factor, h*scale_factor, w*scale_factor])
        return x
    
    def feature_unfolding(self, x, scale_factor):
        """ 仅实现slice_num=8时, 将2D feature map 拓展为3D feature map来处理
        """
        x = torch.unsqueeze(x, dim=2)                           # b c h w      --> b c d h w, d=1
        x = F.interpolate(x, scale_factor, mode='trilinear')    # b c d h w    --> b c 2d 2h 2w
        x = self.pixelshuffle3D_downscale(x, scale_factor)      # b c 2d 2h 2w --> b 8c d h w
        x = torch.squeeze(x, dim=2)                             # b 8c d h w   --> b 8c h w, d=1

        x = rearrange(x, 'b c h w -> b h w c').contiguous()     # [b, h, w, c]
        x_flattened = x.view(-1, self.emb_dim)                  # c = slice_num * emb_dim.  [b h w c] --> [b*h*w*slice_num emb_dim]

        return x_flattened
    
    def feature_folding(self, x, scale_factor):
        x = torch.unsqueeze(x, dim=2)                       # b 8c h w   --> b 8c d h w, d=1
        x = self.pixelshuffle3D_upscale(x, scale_factor)    # b 8c d h w --> b c 2d 2h 2w
        x = F.avg_pool3d(x, kernel_size=scale_factor)       # b c 2d 2h 2w --> b c d h w
        x = torch.squeeze(x, dim=2)                         # b c d h w --> b c h w, d=1
        return x

    def regularize(self, embed):
        return self.weight_decay * torch.linalg.norm(embed.weight, ord=self.ord)

    def forward(self, z):
        b, c, h, w = z.size()
        # ---- 
        scale_factor=2  # TODO
        z_flattened = self.feature_unfolding(z, scale_factor)  # [b*h*w*slice_num emb_dim]

        # if self.ck_enum:
        #     pass
        # elif self.ck_enum is None:  # 
        #     _c = z.size()[1]
        #     if self.e_num is None:
        #         self.e_num = _c // self.e_dim
        #     assert _c == self.e_num*self.e_dim, "_c == self.e_num*self.e_dim"

        # reshape z -> (batch, height, width, channel) and flatten
        # print("z.size()", z.size())  # [b, c, h, w]
        
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)   # b*h*w*slice_num
        z_q = self.embedding(min_encoding_indices)      # [b*h*w*slice_num, emb_dim] 
        z_q = z_q.view(z.shape)                         # [b, h, w, slice_num*emb_dim] 

        # TODO: 
        #   [ ] 根据train、test，决定是否计算loss
        #   [ ] 将loss的计算抽取为单独的函数
        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
            torch.mean((z_q - z.detach()) ** 2) + \
            self.pearson(z_q, z.detach()) + \
            self.regularize(self.embedding)

        # loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()



        return z_q, loss, min_encoding_indices

    # def get_codebook_entry(self, indices, shape):
    #     # shape specifying (batch, height, width, channel)
    #     if self.remap is not None:
    #         indices = indices.reshape(shape[0],-1) # add batch axis
    #         indices = self.unmap_to_all(indices)
    #         indices = indices.reshape(-1) # flatten again

    #     # get quantized latent vectors
    #     z_q = self.embedding(indices)

    #     if shape is not None:
    #         z_q = z_q.view(shape)
    #         # reshape back to match original input shape
    #         z_q = z_q.permute(0, 3, 1, 2).contiguous()

    #     return z_q


# VectorQuantizer with Online Codebook
class VectorQuantizer_OCB_bk(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    def __init__(self, n_e, e_dim, e_num=None, beta=0.25, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=False):
        super().__init__()
        self.n_e = n_e          # total number of embedding code vector

        self.e_num = e_num      # number of embedding code vector within each feature vector.
        self.e_dim = e_dim      # dim of each code vector

        self.ck_enum = None  # 校验e_num的标志位，未校验时为None，如果校验成功则为True，失败为False

        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        # self.normalize(self.embedding, dim=1)

        # TODO 在每次update后进行normalize 
        # TODO 防止匹配时来回跳变
        # TODO 将embedding的数据类型限制为int8 

        # TODO regularization for embedding
        self.weight_decay = 0.01  # SGD的weight decay默认为e-5，此处设置了更大一些的值来增强约束
        self.ord = 1

        self.remap = remap
        # if self.remap is not None:
        #     self.register_buffer("used", torch.tensor(np.load(self.remap)))
        #     self.re_embed = self.used.shape[0]
        #     self.unknown_index = unknown_index # "random" or "extra" or integer
        #     if self.unknown_index == "extra":
        #         self.unknown_index = self.re_embed
        #         self.re_embed = self.re_embed+1
        #     print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
        #           f"Using {self.unknown_index} for unknown indices.")
        # else:
        #     self.re_embed = n_e
        self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

        in_embed_dim = 256
        out_embed_dim = 256 * 8

        self.encoder = nn.Sequential(
            # nn.LayerNorm(normalized_shape=in_embed_dim, eps=1e-6),
            # SelectElement(index=0),
            # nn.Linear(in_embed_dim, out_embed_dim, bias=False),
            nn.Upsample(scale_factor=2, mode='trilinear'),  # N,C,D,H_in,W_in --> N,C,D*scale_factor,H_in*scale_factor,W_in*scale_factor
        )

        print(f"VectorQuantizer2D_v2: init success.", 
              f"n_e={self.n_e}",
              f"e_dim={self.e_dim}",
            #   f"e_num={self.e_num}",
              f"beta={self.beta}",
              f"sane_index_shape={self.sane_index_shape}",
              f"weight_decay={self.weight_decay}"
              )
        
    # def quant_uint8(self):  # 此时codebank全部可能值就固定为uint8的值域
    #     r = 0, 255

    # def quant_int8(self):
    #     r = -128, 127

    def pearson(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        # return cost
        return 0.5 + 0.5 * cost

    def normalize(emb, dim=1):
        emb.weight.data = F.normalize(emb.weight.data, p=2.0, dim=dim)

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    # def unmap_to_all(self, inds):
    #     ishape = inds.shape
    #     assert len(ishape)>1
    #     inds = inds.reshape(ishape[0],-1)
    #     used = self.used.to(inds)
    #     if self.re_embed > self.used.shape[0]: # extra token
    #         inds[inds>=self.used.shape[0]] = 0 # simply set to zero
    #     back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
    #     return back.reshape(ishape)

    def pixelshuffle3D_downscale(self, x, scale_factor):
        # scale_factor = 2
        # b, c, d, h, w --> b, c, d/2, 2, h/2, 2, w/2, 2
        #               --> b, c, 2, 2, 2, d/2, h/2, w/2
        #               --> b, c*8, d/2, h/2, w/2
        bs, c, d, h, w = x.size()
        x = x.view([bs, c, d//scale_factor, scale_factor, h//scale_factor, scale_factor, w//scale_factor, scale_factor])
        x = torch.permute(x, (0, 1, 3, 5, 7, 2, 4, 6))
        # x = x.view([bs, c*scale_factor**3, d//scale_factor, h//scale_factor, w//scale_factor])
        x = x.reshape([bs, c*scale_factor**3, d//scale_factor, h//scale_factor, w//scale_factor])
        return x
    
    def pixelshuffle3D_upscale(self, x, scale_factor):
        # not implemented
        # scale_factor = 2
        # b, c, d, h, w --> b, c//8, 2, 2, 2, d, h, w
        #               --> b, c//8, 2, 2, 2, d, h, w
        #               --> b, c*8, d/2, h/2, w/2
        bs, c, d, h, w = x.size()
        x = x.view([bs, c//(scale_factor**3), scale_factor, scale_factor, scale_factor, d, h, w])
        x = torch.permute(x, (0, 1, 5, 2, 6, 3, 7, 4))
        # x = x.view([bs, c//(scale_factor**3), d*scale_factor, h*scale_factor, w*scale_factor])
        x = x.reshape([bs, c//(scale_factor**3), d*scale_factor, h*scale_factor, w*scale_factor])
        return x

    def regularize(self, embed):
        return self.weight_decay * torch.linalg.norm(embed.weight, ord=self.ord)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"

        b, c, h, w = z.size()
        # ---- 
        scale_factor=2
        z = torch.unsqueeze(z, dim=2)                           # b c h w      --> b c d h w, d=1
        z = self.encoder(z)                                     # b c d h w    --> b c 2d 2h 2w
        z = self.pixelshuffle3D_downscale(z, scale_factor)      # b c 2d 2h 2w --> b 8c d h w
        z = torch.squeeze(z, dim=2)                             # b 8c d h w   --> b 8c h w, d=1

        # if self.ck_enum:
        #     pass
        # elif self.ck_enum is None:  # 
        #     _c = z.size()[1]
        #     if self.e_num is None:
        #         self.e_num = _c // self.e_dim
        #     assert _c == self.e_num*self.e_dim, "_c == self.e_num*self.e_dim"

        # reshape z -> (batch, height, width, channel) and flatten
        # print("z.size()", z.size())  # [b, c, h, w]
        z = rearrange(z, 'b c h w -> b h w c').contiguous()  # [b, h, w, c]

        z_flattened = z.view(-1, self.e_dim)  # c = e_num*e_dim.  [b h w c] --> [b*h*w*e_num e_dim]
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)  # b*h*w*e_num
        z_q = self.embedding(min_encoding_indices)  # [b*h*w*e_num, e_dim] 
        z_q = z_q.view(z.shape)                     # [b, h, w, e_num*e_dim] 

        # 
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
            torch.mean((z_q - z.detach()) ** 2) + \
            self.pearson(z_q, z.detach()) + \
            self.regularize(self.embedding)

        # loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        # if self.remap is not None:
        #     print("######^^^^^^##########self.remap is not None")
        #     min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
        #     min_encoding_indices = self.remap_to_used(min_encoding_indices)
        #     min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        # if self.sane_index_shape:
        #     print("######^^^^^^##########self.sane_index_shape")
        #     min_encoding_indices = min_encoding_indices.reshape(
        #         z_q.shape[0], z_q.shape[2], z_q.shape[3])

        z_q = torch.unsqueeze(z_q, dim=2)                   # b 8c h w   --> b 8c d h w, d=1
        z_q = self.pixelshuffle3D_upscale(z_q, scale_factor)# b 8c d h w --> b c 2d 2h 2w
        z_q = F.avg_pool3d(z_q, kernel_size=scale_factor)   # b c 2d 2h 2w  --> b c d h w
        z_q = torch.squeeze(z_q, dim=2)

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    # def get_codebook_entry(self, indices, shape):
    #     # shape specifying (batch, height, width, channel)
    #     if self.remap is not None:
    #         indices = indices.reshape(shape[0],-1) # add batch axis
    #         indices = self.unmap_to_all(indices)
    #         indices = indices.reshape(-1) # flatten again

    #     # get quantized latent vectors
    #     z_q = self.embedding(indices)

    #     if shape is not None:
    #         z_q = z_q.view(shape)
    #         # reshape back to match original input shape
    #         z_q = z_q.permute(0, 3, 1, 2).contiguous()

    #     return z_q

# -----

# class CodebookBank(nn.Module):
