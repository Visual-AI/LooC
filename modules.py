import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,f=4):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        if f==4:
            self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        elif f==8:
            self._conv_3 = nn.Sequential(
                nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1),
                nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=2, padding=1)
            )
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, output_channels,f=4):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        if f==4:
            self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=output_channels,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        elif f==8:
            self._conv_trans_2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=num_hiddens,
                                                kernel_size=3, 
                                                stride=2, padding=1),
                nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=output_channels,
                                                kernel_size=4, 
                                                stride=2, padding=1)
            )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)
    

class Model(nn.Module):
    def __init__(self, input_dim, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, 
                 embedding_dim, f=4, commitment_cost=0.25, distance='l2', 
                 anchor='closest', first_batch=False, contras_loss=True, 
                 split_type='fixed',
                 args=None):
        super(Model, self).__init__()

        # num_hiddens == embedding_dim*slice_num 才是最终输出的dim
        decoder_in_channel = num_hiddens
        _pre_out_channel = num_hiddens

        print(f"decoder_in_channel = {decoder_in_channel}")
        print(f"_pre_out_channel = {_pre_out_channel}")
        f = 4
            
        self._encoder = Encoder(input_dim, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens,f)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=_pre_out_channel,
                                      kernel_size=1, 
                                      stride=1)
        vq=args.get('vq', 'lorc_old')

        if vq == 'vq':
            from quantizer_zoo.VQ_VAE.quantize import VectorQuantizer
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            
            # embed_num, embed_dim, beta, 
            #      args=None, 
            #      remap=None,
            #      sane_index_shape=False):
            

        elif vq == 'cvq':
            from quantizer_zoo.CVQ.quantise import VectorQuantiser
            self._vq_vae = VectorQuantiser(num_embeddings, embedding_dim, commitment_cost, distance=distance, 
                                       anchor=anchor, first_batch=first_batch, contras_loss=contras_loss) 
        elif vq == 'lorc':    
            from quantizer_zoo.LoRC_VAE.quantise import VectorQuantizer
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, 
                                           args=args,
                                    #    anchor=anchor, first_batch=first_batch, contras_loss=contras_loss,
                                    #    split_type=split_type,
                                       
                                        )
        elif vq == 'lorc_old':    
            from code_backup.quantise import EfficientVectorQuantiser
            self._vq_vae = EfficientVectorQuantiser(num_embeddings, embedding_dim, commitment_cost, distance=distance, 
                                       anchor=anchor, first_batch=first_batch, contras_loss=contras_loss,
                                       split_type=split_type,
                                       args=args)
        else:
            raise NotImplementedError(f"{vq} not ImplementedError")
        
        self._decoder = Decoder(decoder_in_channel,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens,
                                input_dim,f)

    def encode(self, x):
        z_e_x = self._encoder(x)
        z_e_x = self._pre_vq_conv(z_e_x)
        loss, quantized, perplexity, _ = self._vq_vae(z_e_x)
        return loss, quantized, perplexity

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        # quantized, loss, (perplexity, encodings, _, bincount) = self._vq_vae(z)
        quantized, loss, (encoding_indices, bincount) = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        # return x_recon, loss, perplexity, encodings, bincount
        return x_recon, loss, encoding_indices, bincount