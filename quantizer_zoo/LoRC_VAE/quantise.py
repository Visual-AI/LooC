import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch import einsum
from einops import rearrange
from icecream import ic 
import pdb


vq_args = {
    "distance": 'cos',
    "anchor": 'probrandom',
    "split_type": 'fixed',

    "shuffle_scale": 0,

    "contras_loss": False,
    "scale_sim_loss": 0, 

    "return_loss_dict": False,          # True， 则将loss 的每一项保存在dict中，并返回dict


}

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


# 让 codevector 两两之间的距离越远越好。
def uniform_loss(x, t=2):
   return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def leaky_uniform_loss(x, t=2):
   return -(1 - torch.pdist(x, p=2).pow(2).mul(-t).exp()).log().mean()

class VectorQuantizer(nn.Module):
    """
    Improved version over vector quantiser, with the dynamic initialisation
    for these unoptimised "dead" points.
    embed_num: number of codebook entry
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    contras_loss: if true, use the contras_loss to further improve the performance
    """
    def __init__(self, embed_num, embed_dim, beta, 
                 args=None, 
                 remap=None,
                 sane_index_shape=False):
        super().__init__()
        if (remap is not None) or sane_index_shape:
            raise NotImplementedError
        if args is None:
            print("args is None, using default settings.")
            args = vq_args

        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.beta = beta
        self.distance = args.get('distance')
        self.anchor = args.get('anchor')
        self.contras_loss = args.get('contras_loss', False)
        self.scale_sim_loss = args.get('scale_sim_loss', 0)
        self.return_loss_dict = args.get('return_loss_dict', True)

        self.uniform_loss = args.get('uniform_loss', False)
        self.leaky_uniform_loss = args.get('leaky_uniform_loss', False)
        
        self.init_on_the_fly = False

        if self.scale_sim_loss == 0:
            self.calc_sim_loss= args.get('calc_sim_loss', False)
        else:
            self.calc_sim_loss= args.get('calc_sim_loss', True)

        self.random_skip_vq = args.get('random_skip_vq', 0)  # training

        self.on_the_fly_inference = False

        # --
        self.split_type = args.get('split_type')
        self.decay = 0.99
        self.shuffle_scale = args.get('shuffle_scale', 0)

        self.pool = FeaturePool(self.embed_num, self.embed_dim)
        self.embedding = nn.Embedding(self.embed_num, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.embed_num, 1.0 / self.embed_num)
        self.register_buffer("embed_prob", torch.zeros(self.embed_num))

        print(f"EfficientVectorQuantiser: codebook_size = {self.embed_num} * {self.embed_dim} = {self.embed_num * self.embed_dim}")
        print(f"split_type = {self.split_type}")
        print(f"anchor = {self.anchor}")
        print(f"self.scale_sim_loss = {self.scale_sim_loss}")
        print(f"self.calc_sim_loss = {self.calc_sim_loss}")
        print(f"self.uniform_loss = {self.uniform_loss}")
        print(f"self.leaky_uniform_loss = {self.leaky_uniform_loss}")
        print(f"self.contras_loss = {self.contras_loss}")
        print("self.training =", self.training)
        print("self.shuffle_scale =", self.shuffle_scale)
        print("self.return_loss_dict =", self.return_loss_dict)
        print("self.random_skip_vq =", self.random_skip_vq)
        print("self.on_the_fly_inference =", self.on_the_fly_inference)
        print("---> Looc VQ VectorQuantizer: init success.")

    def init_on_the_fly_embedding(self):
        codebook = self.embedding.weight
        if self.distance == 'cos':
            normed_codebook = F.normalize(codebook, dim=1)
            codebook = normed_codebook

        pdist_l1 = torch.pdist(codebook, p=1)
        self.min_pdist_l1 = torch.min(pdist_l1)
        self.max_pdist_l1 = torch.max(pdist_l1)
        self.mean_pdist_l1 = torch.mean(pdist_l1)

        pdist_l2 = torch.pdist(codebook, p=2)
        self.min_pdist_l2 = torch.min(pdist_l2)
        self.max_pdist_l2 = torch.max(pdist_l2)
        self.mean_pdist_l2 = torch.mean(pdist_l2)
        self.init_on_the_fly = True
        print("init_on_the_fly_embedding success")

        

    def upsample(self, x, scale_factor=2):
        # N,C,H_in,W_in --> N,C,H_in*scale_factor,W_in*scale_factor
        y = F.interpolate(x, scale_factor=scale_factor, mode='bilinear')
        # shape (∗,C,H×r,W×r) to (∗,C×r^2,H,W)
        y = F.pixel_unshuffle(y, downscale_factor=scale_factor) 
        # bs, c, h, w = x.size()
        # y = y.view([bs, c, h, scale_factor, w, scale_factor])
        # y = torch.permute(y, (0, 1, 3, 5, 2, 4))
        # y = y.reshape([bs, c*scale_factor**2, h, w])
        return y
    

    def downsample(self, x, scale_factor=2):
        # shape (∗,C×r^2,H,W) to (∗,C,H×r,W×r) 
        y = F.pixel_shuffle(x, upscale_factor=scale_factor) 
        y = F.avg_pool2d(y, kernel_size=scale_factor)
        return y

        
    def preprocess(self, z, embed_dim):
        # Split the feature vector into multiple segments with fixed, interval, or random gaps
        if self.split_type == 'fixed':
            z_flattened = z.view(-1, embed_dim)

        # 采用固定间隔的取样
        elif self.split_type == 'interval':
            if hasattr(self, 'idx'):
                idx = self.idx
            else:
                feature_dim = z.shape[-1]
                slice_num = feature_dim // embed_dim
                idx = []
                for start_idx in range(slice_num):
                    idx += list(range(start_idx, feature_dim, slice_num))
                self.idx = idx
                print("split_type == 'interval', idx =", idx)

            z_flattened = z[:, :, :, idx]  # bs, h, w, c

            z_flattened = z_flattened.view(-1, embed_dim)

        # 每次forward时，随机切分
        elif self.split_type == 'random':
            feature_dim = z.shape[-1]
            idx = list(range(feature_dim))
            idx_shuffled = random.sample(idx, len(idx))
            z_flattened = z[:, :, :, idx_shuffled]  # bs, h, w, c
            z_flattened = z_flattened.view(-1, embed_dim)
        
        # TODO: ablation studyfix random,# 随机确定切分的索引，然后每一次forward都采用同样的切分方式

        return z_flattened
    
    def forward(self, z):
        if not self.training and not self.init_on_the_fly:
            self.init_on_the_fly_embedding()

        if not isinstance(z, list):
            return self._forward(z)
       
        z_q_list = []
        out_loss_list = []
        for zi in z:
            z_q_i, out_loss_i, (encoding_indices, bin_count) = self._forward(zi)
            z_q_list.append(z_q_i)
            out_loss_list.append(out_loss_i)
        z_q = sum(z_q_list)  # element-wise add.

        # Done: merge loss
        if self.return_loss_dict:
            out_loss = {}
            for _key in out_loss_list[0]:
                _v = out_loss_list[0].get(_key)
                for dict_i in out_loss_list[1:]:
                    _v += dict_i.get(_key)
                out_loss.update({_key: _v})
        else:
            out_loss = sum(out_loss_list)
        
        # TODO merge (encoding_indices, bin_count)

        return z_q, out_loss, (encoding_indices, bin_count)


    def _forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"

        # ic('quantise->z.shape: ', z.shape)  # bs * C * 16 * 16

        # on-the-fly-inference 的阈值，若相似度小于该阈值，则视为 outlier，此时跳过该 vector 的 VQ
        t_min_sim = 0.8  # arccos(0.7) = 45deg

        if self.shuffle_scale:
            z = self.upsample(z, scale_factor=self.shuffle_scale)
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()

        embed_dim = self.embed_dim
        embed_num = self.embed_num
        embed_weight = self.embedding.weight
        
        z_flattened = self.preprocess(z, embed_dim)

        # calculate the distance
        if self.distance == 'l2':
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            dist = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(embed_weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(embed_weight, 'n d-> d n'))
        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j 
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()       # [bs * h * w * slice_num, embed_dim]
            normed_codebook = F.normalize(embed_weight, dim=1)         # [embed_num, embed_dim]

            # 以下这步einsum的显存占用，与embed_num成正比，与embed_dim^2成反比。因而embed_num x embed_dim = 8192 x 2时，显存占用很高。
            # dist = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))  # [bs * h * w * slice_num, embed_num], # [1024*8*8*16=1048576, 4096]  # embed_num=4096, embed_dim=8, 300M
            # print("dist.shape:", dist.shape, "z.shape:", z.shape, "self.embedding.weight.shape:", self.embedding.weight.shape)
            # cosine_similarity = torch.matmul(normed_z_flattened, normed_codebook.t())  # {a} . {b}
            # cos_sim = {a} . {b} / |a|*|b|, 值域为[-1， 1]。1为相同，-1则相反

            feature_dim = z.shape[-1]
            slice_num = feature_dim // embed_dim
            
            dist = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))  # [bs * h * w * slice_num, embed_num], # [1024*8*8*16=1048576, 4096]  # embed_num=4096, embed_dim=8, 300M
            encoding_indices = torch.argmax(dist, dim=1)
            encoding_indices_dim0 = torch.argmax(dist, dim=0)
            if not self.training and self.on_the_fly_inference:
                mindist, encoding_indices = torch.max(dist, dim=1)  # max_similary
                res = mindist < t_min_sim  # arccos(0.7) = 45deg
                res_num = res.float().sum()
                if res_num > 0:
                    # raise ValueError(f"res_num = {res_num}")
                    print(f"res_num = {res_num}")
            # pdb.set_trace()
        # ic("self.distance = ", self.distance)
        # ic(encoding_indices.shape, encoding_indices_dim0.shape)
        # TODO 改进dist

        # import pdb
        # pdb.set_trace()
        # encoding
        # 4096x8, 2.4G TODO 降低显存  # indices.shape = [bs * h * w * slice_num, self.embed_num]
        # -- look up the closest point for the indices
        # sort_distance, indices = dist.sort(dim=1)   # dim = 1
        # encoding_indices = indices[:,-1]  # [bs * h * w * slice_num, ]
        if self.distance != 'cos':
            encoding_indices = torch.argmax(dist, dim=1)  # [bs * h * w * slice_num, ]， 比sort 节省显存，尤其是当self.embed_num较大时
       
        #  --------------------
        # [bs * h * w * slice_num, embed_num] # embed_dim 越小，则slice_num越大；codebook_size相同时，self.embed_num也越大；
        # 因此占用显存较大，呈2次方增长
        # encodings = torch.zeros(encoding_indices.shape[0], self.embed_num, device=z.device)  
        # encodings.scatter_(dim=1, index=encoding_indices.unsqueeze(1), src=1)
        # encodings 是每个feature在对应codebook索引位置是否是最近匹配的0/1 标志矩阵。可以看做是codebook在feature中的概率分布情况

        bin_count = torch.bincount(encoding_indices, minlength=embed_num)  # bincount 来获得不同codebook的出现频率

        # quantise and unflatten
        # # [bs * h * w * slice_num, embed_num] [embed_num, embed_dim] --> [bs * h * w * slice_num, embed_dim]
        # z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)  
        z_q = embed_weight[encoding_indices, :] # num, dim

        if not self.training and self.on_the_fly_inference: #  and res_num > 0:
            # pdb.set_trace()
            # res = mindist < t_min_sim
            z_q[res, :] = z_flattened[res, :]

        # 若将 random_skip_vq 放在此处，则 vq loss=0
        # if self.training and self.random_skip_vq: # and random.random() < self.random_skip_vq:
        #     # pdb.set_trace()
        #     # --- skip a batch
        #     # if random.random() < self.random_skip_vq:
        #     #     z_q = z  # skip the vq

        #     # --- skip several samples in each batch
        #     skip_idx = torch.rand(z_q.shape[0]) < self.random_skip_vq
        #     z_q[skip_idx, :]  = z_flattened[skip_idx, :]

        z_q = z_q.view(z.shape)

        # TODO： 使用dict 进行返回loss
        # sim_loss = None
        loss_dict = {}

        # compute loss for embedding
        # loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)  # loss值在约1-5的范围内
        loss_z = torch.mean((z_q.detach()-z)**2)
        loss_q = torch.mean((z_q - z.detach()) ** 2)  # loss_quant
        loss = self.beta * loss_z + loss_q
        loss_dict.update({'loss_z': loss_z, 'loss_q': loss_q})

        if self.training and self.random_skip_vq: # and random.random() < self.random_skip_vq:
            z_q = z_q.view(-1, embed_dim)
            # --- skip a batch
            # if random.random() < self.random_skip_vq:
            #     z_q = z  # skip the vq
            # --- skip several samples in each batch
            skip_idx = torch.rand(z_q.shape[0]) < self.random_skip_vq
            z_q[skip_idx, :]  = z_flattened[skip_idx, :]
            z_q = z_q.view(z.shape)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        # count
        # import pdb
        # pdb.set_trace()
        # TODO remove [Done: 可使用bincountlai计算perplexity，要比sort快很多，也更省显存]
        # avg_probs = torch.mean(encodings, dim=0)  # 可以用bincount来代替
        avg_probs = bin_count / torch.sum(bin_count)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        # min_encodings = encodings

        # online clustered reinitialisation for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            # add: each element of 'avg_probs' is scaled by alpha before being used.
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)  # TODO remove
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom', 'disturbance']:
                # closest sampling
                # 取最接近codebook的self.embed_num个flattened feature vector 
                if self.anchor == 'closest':
                    # encoding_indices_dim0 = torch.argmax(dist, dim=0)
                    # sort_distance, indices = dist.sort(dim=0)
                    # random_feat = z_flattened.detach()[indices[-1,:]]
                    random_feat = z_flattened.detach()[encoding_indices_dim0]

                    # print("encoding_indices.shape =", encoding_indices.shape, "indices.shape =", indices.shape)
                      
                # feature pool based random sampling
                # 将以往的feature保存在feature pool中，随机从中给出feature
                elif self.anchor == 'random':
                    random_feat = self.pool.query(z_flattened.detach())  
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(dist.t(), dim=1)  #  消耗显存
                    # 将norm_distance作为采样的权重，取num_samples个数，返回的是索引。
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)  
                    random_feat = z_flattened.detach()[prob]
                # TODO
                elif self.anchor == 'disturbance':  
                    random_feat = self.pool.query(z_flattened.detach())
                    # For each bit of feature, add a random disturbance between [-1, 1]
                    # 对feature的每一位，增加一个[-1, 1] 之间的随机扰动
                    random_feat += (torch.rand((self.embed_num, self.embed_dim)) * 2 - 1).to(random_feat.device)        

                # decay parameter based on the average usage
                decay = torch.exp(-(self.embed_prob*self.embed_num*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                # decay = 0.99
                # 初始值都比较小（0.002），逐渐增大（0.4）
                # 说明随着训练逐渐收敛，codebook被匹配的向量逐渐固化下来，并且部分codebook的匹配较少。
                # print("torch.max(decay)=", torch.max(decay))  
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay

            elif self.anchor in ['none', 'mutation']:
                if self.anchor == 'none':  # 
                    pass
                elif self.anchor == 'mutation':  # 基因突变
                    # TODO 若采用突变的方式，则需要维护两个codebook， 仅对需要更新的codebook进行突变， 且不再进行decay
                    # --- 单点突变
                    # - 随机选择一个位置，增加一个随机的扰动

                    # --- 整体突变
                    # - 对entry的每一位，增加一个[-1, 1] 之间的随机扰动
                    # random_feat = (torch.rand((z_flattened.size(0), self.embed_dim)) * 2 - 1)/ z_flattened.size(0)
                    # torch.rand((self.embed_num, self.embed_dim))* 2 - 1

                    # --- 结构突变
                    # - 随机交换2个单点位置
                    # - 随机位置截断，然后交换顺序

                    # --- 交叉突变
                    # - 随机从其他entry中，选择一部分进行替换
                    
                    raise NotImplementedError
            else:
                raise NotImplementedError
            
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = dist.sort(dim=0)  # FIX 无需重复计算 
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.embed_num)):,:].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07         # 为什么 /0.07?
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                # dis_neg 的target 设为0 是否合适？我觉得dis_neg对应的feature 和 codebook应该距离远一点，相当于类间差异应该大一点。
                loss +=  contra_loss
                loss_dict.update({'contra_loss': contra_loss})

            # codebook的每个entry之间尽可能正交
            # 计算两两的相似度
            if self.calc_sim_loss:
                
                cos_sim = torch.matmul(normed_codebook, normed_codebook.t())
                # cos_sim = 0.5 + 0.5 * cos_sim  # [-1, 1] --> [0, 1],  注意，此时相同的元素的相似度≠1，而是=0.5
                # cos_sim 按列求和，含义为：该entry 与其他entry的相似度之和
                # cos_sim 按列求max，含义为：该entry 与其他entry的相似度最大的值

                # import pdb
                # pdb.set_trace()
                # ---- 一些信息记录
                diag = torch.diag(cos_sim)  # 取对角线元素，输出为 1*N
                sim_diag = torch.diag_embed(diag)   # 由 diag 恢复为N*N
                sim_masked = cos_sim - sim_diag     # cos_sim 对角线置 0
                # sim_masked = cos_sim.fill_diagonal_(float(0))  # 对角线置 0
                #
                sim_sum_each_row = torch.sum(sim_masked, dim=1)  # 每一行内的元素相加
                sim_row_max_val = torch.max(sim_sum_each_row)        # 相似度之和的最大值   
                # sim_row_max_idx = torch.argmax(sim_sum_each_row)     # 相似度之和的最大值的索引   
                # sim_max_each_row, sim_maxidx_each_row = torch.max(sim_masked, dim=1)
                # sim_max = torch.max(sim_max_each_row)  # torch.max(sim_max_each_row) == torch.max(sim_masked)
                sim_max = torch.max(sim_masked)
                loss_dict.update({'sim_max': sim_max})
                loss_dict.update({'sim_row_max_val': sim_row_max_val})

                # 去掉对角线上自相似度==1的元素
                # sim_loss = (torch.sum(cos_sim) - embed_num) / (embed_num * (embed_num - 1))  # sim_loss值在0.5 左右

                sim_loss = (torch.sum(abs(cos_sim)) - embed_num) / (embed_num * (embed_num - 1))  # sim_loss值在0.5 左右

                if self.scale_sim_loss:  # 若scale==0，则仅记录，不参与训练
                    loss += self.scale_sim_loss * sim_loss
                # todo: 可以进一步考虑增加mask，只对相似度>阈值T=0.5的计算loss，或者将数量作为loss
                # print('sim_loss:', sim_loss, 'loss:', loss)
                loss_dict.update({'sim_loss': sim_loss}) 
                # embed_num == self.embed_num == normed_codebook.size(0)

                # torch.min(cos_sim)
                # torch.min(cos_sim2)
                # torch.max(cos_sim2)
            if self.uniform_loss:
                _uniform_loss = 0.1 * uniform_loss(normed_codebook, t=2)
                loss += _uniform_loss
                loss_dict.update({'uniform_loss': _uniform_loss}) 
            
            if self.leaky_uniform_loss:
                _leaky_uniform_loss = leaky_uniform_loss(normed_codebook, t=2)
                loss += _leaky_uniform_loss
                loss_dict.update({'leaky_uniform_loss': _leaky_uniform_loss})

        # perplexity = None       # TODO 
        # return z_q, loss, (perplexity, min_encodings, encoding_indices, bin_count)
        loss_dict.update({'loss': loss})

        if self.shuffle_scale:
            z_q = self.downsample(z_q, scale_factor=self.shuffle_scale)

        out_loss = loss_dict if self.return_loss_dict else loss

        # return z_q, loss, (encoding_indices, bin_count)
        return z_q, out_loss, (encoding_indices, bin_count)


class FeaturePool():
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1)/ pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)    
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features