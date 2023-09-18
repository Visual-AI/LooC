import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch import einsum
from einops import rearrange
from icecream import ic 


class VectorQuantiser(nn.Module):
    """
    Improved version over vector quantiser, with the dynamic initialisation
    for these unoptimised "dead" points.
    num_embed: number of codebook entry
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    first_batch: if true, the offline version of our model
    contras_loss: if true, use the contras_loss to further improve the performance
    """
    def __init__(self, num_embed, embed_dim, beta, distance='cos', 
                 anchor='probrandom', first_batch=False, contras_loss=False, 
                 slice_num=None):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False
        # slice_num != None, 则是使用slice 的方式处理 CVQ-VAE的codebook
        self.slice_num = slice_num
        if self.slice_num:
            print(f"slice_num = {self.slice_num}, 则是使用slice 的方式处理 CVQ-VAE的codebook. 仅用于test")
        self.pool = FeaturePool(self.num_embed, self.embed_dim)
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))

        # TODO 控制不同方法的size一致
        self.codebook_size = self.num_embed * self.embed_dim  
        print(f"codebook_size = num_embed * embed_dim = {self.num_embed} * {self.embed_dim} = {self.codebook_size}")
 
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"

        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()

        if self.slice_num and not self.training:  # slice_num == None, 则是使用slice 的方式处理 CVQ-VAE的codebook
            # self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
            embed_dim = self.embed_dim / self.slice_num
            embed_weight = self.embedding.weight.reshape(self.num_embed * self.slice_num, embed_dim)
        else:
            embed_dim = self.embed_dim
            embed_weight = self.embedding.weight

        z_flattened = z.view(-1, embed_dim)

        # clculate the distance
        if self.distance == 'l2':
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            dist = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(self.embedding.weight, 'n d-> d n'))
        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j 
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            dist = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))  # 4096x8, 300M
            # cosine_similarity = torch.matmul(normed_z_flattened, normed_codebook.t())  # {a} . {b}
            # cos_sim = {a} . {b} / |a|*|b|, 值域为[-1， 1]。1为相同，-1则相反
        # TODO 改进dist

        # import pdb
        # pdb.set_trace()
        # encoding
        # 4096x8, 2.4G TODO 降低显存  # indices.shape = [bs * h * w * slice_num, self.num_embed]
        # -- look up the closest point for the indices
        sort_distance, indices = dist.sort(dim=1)   # dim = 1
        encoding_indices = indices[:,-1]  # [bs * h * w * slice_num, ]
        
        #  --------------------
        # [bs * h * w * slice_num, num_embed] # embed_dim 越小，则slice_num越大；codebook_size相同时，self.num_embed也越大；
        # 因此占用显存较大，呈2次方增长
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embed, device=z.device)  
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)  # encodings.scatter_(dim=1, index=encoding_indices.unsqueeze(1), src=1)

        # quantise and unflatten
        # # [bs * h * w * slice_num, num_embed] [num_embed, embed_dim] --> [bs * h * w * slice_num, embed_dim] 
        z_q = torch.matmul(encodings, embed_weight).view(z.shape)
        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        # count
        # import pdb
        # pdb.set_trace()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        min_encodings = encodings

        if not self.training:
            bin_count = torch.bincount(encoding_indices, minlength=self.num_embed)  # bincount 来获得不同codebook的出现频率
        else:
            bin_count = None

        # online clustered reinitialisation for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                # TODO: sample 策略似乎影响不大
                # 由于bug， 去掉了108行，变成了sample时 采用最后一个sample 的self.num_embed个flattened feature vector 
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = dist.sort(dim=0)  # dim = 0
                    random_feat = z_flattened.detach()[indices[-1,:]]  # 取最接近codebook的self.num_embed个flattened feature vector 
                # feature pool based random sampling
                elif self.anchor == 'random':
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(dist.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = torch.exp(-(self.embed_prob*self.num_embed*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = dist.sort(dim=0)  # FIX self.anchor == 'closest'时，无需重复计算
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.num_embed)):,:].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss +=  contra_loss

        # return z_q, loss, (perplexity, min_encodings, encoding_indices)
        return z_q, loss, (encoding_indices, bin_count)


class EfficientVectorQuantiser(nn.Module):
    """
    Improved version over vector quantiser, with the dynamic initialisation
    for these unoptimised "dead" points.
    num_embed: number of codebook entry
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    first_batch: if true, the offline version of our model
    contras_loss: if true, use the contras_loss to further improve the performance
    """
    def __init__(self, num_embed, embed_dim, beta, distance='cos', 
                 anchor='probrandom', first_batch=False, contras_loss=False,
                 slice_num=None,
                 split_type='fixed',
                 args=None):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        # slice_num != None, 则是使用slice 的方式处理 CVQ-VAE的codebook
        self.slice_num = slice_num
        if self.slice_num:
            print(f"slice_num = {self.slice_num}, 则是使用slice 的方式处理 CVQ-VAE的codebook. 仅用于test")
        
        self.pool = FeaturePool(self.num_embed, self.embed_dim)
        print('type(args)', type(args))
        scale_grad_by_freq = args.scale_grad_by_freq if args else False

        # self.embedding = nn.Embedding(self.num_embed, self.embed_dim)  # TODO
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim, scale_grad_by_freq=scale_grad_by_freq)  # default = False
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))

        self.split_type = split_type
        
        # TODO 控制不同方法的size一致
        self.codebook_size = self.num_embed * self.embed_dim  
        print(f"EfficientVectorQuantiser: codebook_size = num_embed * embed_dim = {self.num_embed} * {self.embed_dim} = {self.codebook_size}")
        print(f"split_type = {split_type}")
        print(f"anchor = {anchor}")
        print(f"scale_grad_by_freq = {scale_grad_by_freq}")

    def embedding_split(self, z, embed_dim):
        # Split the feature vector into multiple segments with fixed, interval, or random gaps
        if self.split_type == 'fixed':
            z_flattened = z.view(-1, embed_dim)

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

        elif self.split_type == 'random':
            feature_dim = z.shape[-1]
            idx = list(range(feature_dim))
            idx_shuffled = random.sample(idx, len(idx))
            z_flattened = z[:, :, :, idx_shuffled]  # bs, h, w, c
            z_flattened = z_flattened.view(-1, embed_dim)

        return z_flattened

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"

        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()

        if self.slice_num and not self.training:  # slice_num == None, 则是使用slice 的方式处理 CVQ-VAE的codebook
            # self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
            embed_dim = int(self.embed_dim / self.slice_num)
            embed_num = int(self.num_embed * self.slice_num)
            # print("embed_num, embed_dim", embed_num, embed_dim)
            embed_weight = self.embedding.weight.reshape((embed_num, embed_dim))
        else:
            embed_dim = self.embed_dim
            embed_num = self.num_embed
            embed_weight = self.embedding.weight
        
        # z_flattened = z.view(-1, embed_dim)
        # import pdb
        # pdb.set_trace()
        z_flattened = self.embedding_split(z, embed_dim)

        # clculate the distance
        if self.distance == 'l2':
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            dist = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(embed_weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(embed_weight, 'n d-> d n'))
        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j 
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()       # [bs * h * w * slice_num, embed_dim]
            normed_codebook = F.normalize(embed_weight, dim=1)         # [num_embed, embed_dim]

            # 以下这步einsum的显存占用，与embed_num成正比，与embed_dim^2成反比。因而embed_num x embed_dim = 8192 x 2时，显存占用很高。
            # dist = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))  # [bs * h * w * slice_num, num_embed], # [1024*8*8*16=1048576, 4096]  # num_embed=4096, embed_dim=8, 300M
            # print("dist.shape:", dist.shape, "z.shape:", z.shape, "self.embedding.weight.shape:", self.embedding.weight.shape)
            # cosine_similarity = torch.matmul(normed_z_flattened, normed_codebook.t())  # {a} . {b}
            # cos_sim = {a} . {b} / |a|*|b|, 值域为[-1， 1]。1为相同，-1则相反

            feature_dim = z.shape[-1]
            slice_num = feature_dim // embed_dim
            
            if embed_num * slice_num >= 2048 * 64:  
                # -- slice num 较大或codebook较长时，分段计算距离（相似度），取相似度最高的index  
                # -- 将中间变量“相似度矩阵”给丢掉，能够减少显存消耗，但是不能减少计算量
                # dist_list= []
                _tmp_n = slice_num
                encoding_indices_list = []
                normed_z_flattened = normed_z_flattened.view(-1, _tmp_n, embed_dim)  # [bs * h * w, slice_num, embed_dim]  # _tmp_n = slice_num

                for i in range(_tmp_n):
                    dist_i = torch.einsum('bd,dn->bn', normed_z_flattened[:, i], rearrange(normed_codebook, 'n d -> d n'))
                    # dist_list.append(dist_i)  # 这一步依旧持续增长显存
                    encoding_indices_i = torch.argmax(dist_i, dim=1)
                    encoding_indices_list.append(encoding_indices_i)

                encoding_indices = torch.cat(encoding_indices_list)
            else:
                dist = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))  # [bs * h * w * slice_num, num_embed], # [1024*8*8*16=1048576, 4096]  # num_embed=4096, embed_dim=8, 300M
                encoding_indices = torch.argmax(dist, dim=1)

        # TODO 改进dist

        # import pdb
        # pdb.set_trace()
        # encoding
        # 4096x8, 2.4G TODO 降低显存  # indices.shape = [bs * h * w * slice_num, self.num_embed]
        # -- look up the closest point for the indices
        # sort_distance, indices = dist.sort(dim=1)   # dim = 1
        # encoding_indices = indices[:,-1]  # [bs * h * w * slice_num, ]
        if self.distance != 'cos':
            encoding_indices = torch.argmax(dist, dim=1)  # [bs * h * w * slice_num, ]， 比sort 节省显存，尤其是当self.num_embed较大时
       
        #  --------------------
        # [bs * h * w * slice_num, num_embed] # embed_dim 越小，则slice_num越大；codebook_size相同时，self.num_embed也越大；
        # 因此占用显存较大，呈2次方增长
        # encodings = torch.zeros(encoding_indices.shape[0], self.num_embed, device=z.device)  
        # encodings.scatter_(dim=1, index=encoding_indices.unsqueeze(1), src=1)
        # encodings 是每个feature在对应codebook索引位置是否是最近匹配的0/1 标志矩阵。可以看做是codebook在feature中的概率分布情况

        if not self.training:
            bin_count = torch.bincount(encoding_indices, minlength=embed_num)  # bincount 来获得不同codebook的出现频率
            # # 补全bin_count
            # if torch.numel(bin_count) < embed_num:
            #     miss_num = embed_num - torch.numel(bin_count)
            #     bin_count = F.pad(bin_count, [0, miss_num], "constant", 0)

            # nonzero = torch.count_nonzero(bin_count)
            # ic(embed_num - nonzero, z.shape[0], z_flattened.shape[0])
            # ic(embed_num, self.num_embed)

            # import pdb
            # pdb.set_trace()
            # encoding_indices_2 2048

            # encodings = torch.zeros(encoding_indices.shape[0], self.num_embed, dtype=torch.float, device=z.device)   # 2048, 512
            # encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
            # avg_probs = bin_count / torch.sum(bin_count)
            # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            # min_encodings = encodings
        else:
            bin_count = None
            # min_encodings = None
            # perplexity = None
        # 对于未匹配的codebook怎么处理？

        # quantise and unflatten
        # # [bs * h * w * slice_num, num_embed] [num_embed, embed_dim] --> [bs * h * w * slice_num, embed_dim]
        # z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)  
        z_q = embed_weight[encoding_indices, :] # num, dim

        z_q = z_q.view(z.shape)  

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        # count
        # import pdb
        # pdb.set_trace()
        # TODO remove [Done: 可使用bincountlai计算perplexity，要比sort快很多，也更省显存]
        # avg_probs = torch.mean(encodings, dim=0)  # 可以用bincount来代替
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        # min_encodings = encodings

        # online clustered reinitialisation for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            # add: each element of 'avg_probs' is scaled by alpha before being used.
            # self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)  # TODO remove
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom', 'disturbance'] and (not self.init):
                # TODO: sample 策略似乎影响不大
                # 由于bug， 去掉了108行，变成了sample时 采用最后一个sample 的self.num_embed个flattened feature vector 
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = dist.sort(dim=0)  # dim = 0
                    random_feat = z_flattened.detach()[indices[-1,:]]  # 取最接近codebook的self.num_embed个flattened feature vector 
                # feature pool based random sampling
                elif self.anchor == 'random':
                    random_feat = self.pool.query(z_flattened.detach())  # 将以往的feature保存在feature pool中，随机从中给出feature
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(dist.t(), dim=1)  #  消耗显存
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)  # 将norm_distance作为采样的权重，取num_samples个数，返回的是索引。
                    random_feat = z_flattened.detach()[prob]
                
                elif self.anchor == 'disturbance':  
                    random_feat = self.pool.query(z_flattened.detach())
                    # For each bit of feature, add a random disturbance between [-1, 1]
                    # 对feature的每一位，增加一个[-1, 1] 之间的随机扰动
                    random_feat += (torch.rand((self.num_embed, self.embed_dim)) * 2 - 1).to(random_feat.device)        

                # decay parameter based on the average usage
                # # TODO remove
                # decay = torch.exp(-(self.embed_prob*self.num_embed*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                decay = 0.99
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
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
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.num_embed)):,:].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss +=  contra_loss

        # perplexity = None       # TODO 
        # return z_q, loss, (perplexity, min_encodings, encoding_indices, bin_count)
        return z_q, loss, (encoding_indices, bin_count)


class IncrementalVectorQuantiser(nn.Module):
    """
    Improved version over vector quantiser, with the dynamic initialisation
    for these unoptimised "dead" points.
    num_embed: number of codebook entry
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    first_batch: if true, the offline version of our model
    contras_loss: if true, use the contras_loss to further improve the performance
    """
    def __init__(self, num_embed, embed_dim, beta, distance='cos', 
                 anchor='random', first_batch=False, contras_loss=False,
                 slice_num=None,
                 split_type='fixed'):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        # slice_num != None, 则是使用slice 的方式处理 CVQ-VAE的codebook
        self.slice_num = slice_num
        if self.slice_num:
            print(f"slice_num = {self.slice_num}, 则是使用slice 的方式处理 CVQ-VAE的codebook. 仅用于test")
        
        self.fixed_embedding = nn.Embedding(self.num_embed, self.embed_dim)

        self.pool = FeaturePool(self.num_embed, self.embed_dim, scale_grad_by_freq=True)
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))

        self.split_type = split_type
        
        # TODO 控制不同方法的size一致
        self.codebook_size = self.num_embed * self.embed_dim  
        print(f"EfficientVectorQuantiser: codebook_size = num_embed * embed_dim = {self.num_embed} * {self.embed_dim} = {self.codebook_size}")
        print(f"split_type = {split_type}")
        print(f"anchor = {anchor}")

    def embedding_split(self, z, embed_dim):
        # Split the feature vector into multiple segments with fixed, interval, or random gaps
        if self.split_type == 'fixed':
            z_flattened = z.view(-1, embed_dim)

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

        elif self.split_type == 'random':
            feature_dim = z.shape[-1]
            idx = list(range(feature_dim))
            idx_shuffled = random.sample(idx, len(idx))
            z_flattened = z[:, :, :, idx_shuffled]  # bs, h, w, c
            z_flattened = z_flattened.view(-1, embed_dim)

        return z_flattened

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"

        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()

        if self.slice_num and not self.training:  # slice_num == None, 则是使用slice 的方式处理 CVQ-VAE的codebook
            # self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
            embed_dim = int(self.embed_dim / self.slice_num)
            embed_num = int(self.num_embed * self.slice_num)
            # print("embed_num, embed_dim", embed_num, embed_dim)
            embed_weight = self.embedding.weight.reshape((embed_num, embed_dim))
        else:
            embed_dim = self.embed_dim
            embed_num = self.num_embed
            embed_weight = self.embedding.weight
        
        # z_flattened = z.view(-1, embed_dim)

        z_flattened = self.embedding_split(z, embed_dim)

        # clculate the distance
        if self.distance == 'l2':
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            dist = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(embed_weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(embed_weight, 'n d-> d n'))
        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j 
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()       # [bs * h * w * slice_num, embed_dim]
            normed_codebook = F.normalize(embed_weight, dim=1)         # [num_embed, embed_dim]
            dist = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))  # [bs * h * w * slice_num, num_embed], # [1024*8*8*16=1048576, 4096]  # num_embed=4096, embed_dim=8, 300M
            # print("dist.shape:", dist.shape, "z.shape:", z.shape, "self.embedding.weight.shape:", self.embedding.weight.shape)
            # cosine_similarity = torch.matmul(normed_z_flattened, normed_codebook.t())  # {a} . {b}
            # cos_sim = {a} . {b} / |a|*|b|, 值域为[-1， 1]。1为相同，-1则相反
        # TODO 改进dist

        # import pdb
        # pdb.set_trace()
        # encoding
        # 4096x8, 2.4G TODO 降低显存  # indices.shape = [bs * h * w * slice_num, self.num_embed]
        # -- look up the closest point for the indices
        # sort_distance, indices = dist.sort(dim=1)   # dim = 1
        # encoding_indices = indices[:,-1]  # [bs * h * w * slice_num, ]
        encoding_indices = torch.argmax(dist, dim=1)  # [bs * h * w * slice_num, ]， 比sort 节省显存，尤其是当self.num_embed较大时
        
        #  --------------------
        # [bs * h * w * slice_num, num_embed] # embed_dim 越小，则slice_num越大；codebook_size相同时，self.num_embed也越大；
        # 因此占用显存较大，呈2次方增长
        # encodings = torch.zeros(encoding_indices.shape[0], self.num_embed, device=z.device)  
        # encodings.scatter_(dim=1, index=encoding_indices.unsqueeze(1), src=1)
        # encodings 是每个feature在对应codebook索引位置是否是最近匹配的0/1 标志矩阵。可以看做是codebook在feature中的概率分布情况

        if not self.training:
            bin_count = torch.bincount(encoding_indices, minlength=embed_num)  # bincount 来获得不同codebook的出现频率
            # # 补全bin_count
            # if torch.numel(bin_count) < embed_num:
            #     miss_num = embed_num - torch.numel(bin_count)
            #     bin_count = F.pad(bin_count, [0, miss_num], "constant", 0)

            # nonzero = torch.count_nonzero(bin_count)
            # ic(embed_num - nonzero, z.shape[0], z_flattened.shape[0])
            # ic(embed_num, self.num_embed)

            # import pdb
            # pdb.set_trace()
            # encoding_indices_2 2048

            # encodings = torch.zeros(encoding_indices.shape[0], self.num_embed, dtype=torch.float, device=z.device)   # 2048, 512
            # encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
            # avg_probs = bin_count / torch.sum(bin_count)
            # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            # min_encodings = encodings
        else:
            bin_count = None
            # min_encodings = None
            # perplexity = None
        # 对于未匹配的codebook怎么处理？

        # quantise and unflatten
        # # [bs * h * w * slice_num, num_embed] [num_embed, embed_dim] --> [bs * h * w * slice_num, embed_dim]
        # z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)  
        z_q = embed_weight[encoding_indices, :] # num, dim

        z_q = z_q.view(z.shape)  

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        # count
        # import pdb
        # pdb.set_trace()
        # TODO remove [Done: 可使用bincountlai计算perplexity，要比sort快很多，也更省显存]
        # avg_probs = torch.mean(encodings, dim=0)  # 可以用bincount来代替
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        # min_encodings = encodings

        # online clustered reinitialisation for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            # self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)  # TODO remove
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom', 'disturbance'] and (not self.init):
                # TODO: sample 策略似乎影响不大
                # 由于bug， 去掉了108行，变成了sample时 采用最后一个sample 的self.num_embed个flattened feature vector 
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = dist.sort(dim=0)  # dim = 0
                    random_feat = z_flattened.detach()[indices[-1,:]]  # 取最接近codebook的self.num_embed个flattened feature vector 
                # feature pool based random sampling
                elif self.anchor == 'random':
                    random_feat = self.pool.query(z_flattened.detach())  # 将以往的feature保存在feature pool中，随机从中给出feature
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(dist.t(), dim=1)  #  消耗显存
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                
                elif self.anchor == 'disturbance':  
                    random_feat = self.pool.query(z_flattened.detach())
                    # For each bit of feature, add a random disturbance between [-1, 1]
                    # 对feature的每一位，增加一个[-1, 1] 之间的随机扰动
                    random_feat += (torch.rand((self.num_embed, self.embed_dim)) * 2 - 1).to(random_feat.device)

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
                    


                # decay parameter based on the average usage
                # # TODO remove
                # decay = torch.exp(-(self.embed_prob*self.num_embed*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                decay = 0.99
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = dist.sort(dim=0)  # FIX 无需重复计算
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.num_embed)):,:].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss +=  contra_loss

        # perplexity = None       # TODO 
        # return z_q, loss, (perplexity, min_encodings, encoding_indices, bin_count)
        return z_q, loss, (encoding_indices, bin_count)



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