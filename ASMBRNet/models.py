from collections import OrderedDict
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
from bra_legacy import BiLevelRoutingAttention
from _common import Attention, AttentionLePE, DWConv


# UNet Backbone
def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class UNetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x
def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    # if name == 'sum':
    #     return Summer(PositionalEncodingPermute2D(emb_dim))
    # elif name == 'npe.sin':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='sin')
    # elif name == 'npe.coord':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='coord')
    # elif name == 'hpe.conv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='conv', res_shortcut=True)
    # elif name == 'hpe.dsconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='dsconv', res_shortcut=True)
    # elif name == 'hpe.pointconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='pointconv', res_shortcut=True)
    else:
        raise ValueError(f'PE name {name} is not surpported!')


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                 num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=False,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                                qk_scale=qk_scale, kv_per_win=kv_per_win,
                                                kv_downsample_ratio=kv_downsample_ratio,
                                                kv_downsample_kernel=kv_downsample_kernel,
                                                kv_downsample_mode=kv_downsample_mode,
                                                topk=topk, param_attention=param_attention, param_routing=param_routing,
                                                diff_routing=diff_routing, soft_routing=soft_routing,
                                                side_dwconv=side_dwconv,
                                                auto_pad=auto_pad)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'),  # compatiability
                                      nn.Conv2d(dim, dim, 1),  # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim),  # pseudo attention
                                      nn.Conv2d(dim, dim, 1),  # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                      )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))  # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
        else:  # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))  # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x



# Main Network for extracting buildings str.1: taking BiFormer as leadback and UNet as assisting backbone, the stages 1-4 of UNet is used, adding a stage in BiFormer
# and add a module of biformer, finally performing a 2*2 upsampling.
class ASMBRNet(nn.Module):
    output_downscaled = 1
    module = UNetModule
    def __init__(self, depth=[2, 2, 2, 8, 2], in_chans=483, num_classes=1, embed_dim=[32, 64, 128, 256, 512],
                 head_dim=32, qk_scale=None, representation_size=None,
                 drop_path_rate=0.,
                 use_checkpoint_stages=[],
                 n_win=8,
                 kv_downsample_mode='identity',
                 kv_per_wins=[-1, -1, -1, -1, -1],
                 topks=[1, 1, 4, 16, -2],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[32, 64, 128, 256, 512],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=False,
                 kv_downsample_kernels=[2, 2, 2, 1, 1],
                 kv_downsample_ratios=[2, 2, 2, 1, 1],
                 mlp_ratios=[4, 4, 4, 4, 4],
                 param_attention='qkvo',
                 mlp_dwconv=False,
                 input_channels: int = 3,
                 filters_base: int = 32,
                 down_filter_factors=(1, 2, 4, 8, 16),
                 up_filter_factors=[96, 192, 384, 768],
                 up_filter_factors1=[32, 64, 128, 256],
                 bottom_s=4,
                 add_output=True):

        super(ASMBRNet, self).__init__()
        # constructing a BiFormer referring to "BiFormer: Vision Transformer with Bi-Level Routing Attention"
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0],))

        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)
        in_channels = [480, 448, 384, 256]
        out_channels = [64, 128, 256, 512]
        # 循环构建下采样层
        for in_ch, out_ch in zip(in_channels, out_channels):
            downsample_layer = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(out_ch)
            )
            self.downsample_layers.append(downsample_layer)
        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(5):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        topk=topks[i],
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]
        self.norm = nn.BatchNorm2d(embed_dim[-1])

        # constructing a UNet backbone##########################################################################

        self.num_classes = num_classes
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        self.assisting_down, self.assisting_up = nn.ModuleList(), nn.ModuleList()
        self.assisting_down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.assisting_down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in list(zip(up_filter_factors, up_filter_factors1)):
            self.assisting_up.append(self.module(prev_i, nf))

        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.Upsample(scale_factor=2)
        self.downsamplers = [None] + [pool] * (len(self.assisting_down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.assisting_up)
        self.downsampler = nn.MaxPool2d(2, 2)
        upsample = nn.Upsample(scale_factor=2)
        upsample1 = nn.Upsample(scale_factor=4)
        upsample2 = nn.Upsample(scale_factor=8)
        upsample3 = nn.Upsample(scale_factor=16)
        upsample4 = nn.Upsample(scale_factor=32)
        self.assistingUPS = nn.Sequential(upsample, upsample1, upsample2, upsample3, upsample4)
        self.add_output = add_output
        if add_output:
            self.conv_final1 = nn.Conv2d(32, num_classes, 1)
        if add_output:
            self.conv_final2 = nn.Conv2d(32, num_classes, 1)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        pretrained_ASFE_weights = torch.load('/ASFE_weights.pt')
        AE_weights = {}
        ME_weights = {}
        DL_weights = {}
        for key, value in pretrained_ASFE_weights.items():
            # 检查新的键是否存在于模型的 state_dict 中
            if 'downsample_layers' in key:
                DL_weights[key] = value
            elif 'stages' in key:
                ME_weights[key] = value
            elif 'assisting_down' in key:
                AE_weights[key] = value
        self.downsample_layers.load_state_dict(DL_weights, strict=False)
        self.stages.load_state_dict(ME_weights, strict=False)
        self.assisting_down.load_state_dict(AE_weights, strict=False)
        print('Successfully loaded pre-training weights!')


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        # encoding local features by UNet
        assisting_xs = []
        lead_xs = []
        for assisiting_downsample, assisiting_down in zip(self.downsamplers, self.assisting_down):
            x_in = x if assisiting_downsample is None else assisiting_downsample(assisting_xs[-1])
            x_out = assisiting_down(x_in)
            assisting_xs.append(x_out)    #xs:list 5, 0:[32,512,512], 1:[64,256,256], 2:[128,128,128], 3:[256,64,64]

        ass_fea1 = torch.cat([assisting_xs[0], F.interpolate(assisting_xs[1], size=(512, 512), mode='nearest'), F.interpolate(assisting_xs[2], size=(512, 512), mode='nearest'), F.interpolate(assisting_xs[3], size=(512, 512), mode='nearest')], dim=1)
        ass_fea2 = torch.cat([F.interpolate(assisting_xs[1], size=(256, 256), mode='nearest'), F.interpolate(assisting_xs[2], size=(256, 256), mode='nearest'), F.interpolate(assisting_xs[3], size=(256, 256), mode='nearest')], dim=1)
        ass_fea3 = torch.cat([F.interpolate(assisting_xs[2], size=(128, 128), mode='nearest'), F.interpolate(assisting_xs[3], size=(128, 128), mode='nearest')], dim=1)
        ass_fea4 = F.interpolate(assisting_xs[3], size=(64, 64), mode='nearest')
        ass_fea = [ass_fea1, ass_fea2, ass_fea3, ass_fea4]
        # encoding local features by BiFormer
        Bs = []
        for i in range(5):
            if i == 0:
                x = self.downsample_layers[i](torch.cat([x, ass_fea[i]], dim=1)) # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
                x = self.stages[i](x)
            elif i<4:
                x = self.downsample_layers[i](torch.cat([x, ass_fea[i]], dim=1)) # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
                x = self.stages[i](x)
            else:
                x = self.downsample_layers[i](x) # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
                x = self.stages[i](x)
            Bs.append(x)  # X1(1,64,128,128), X2(1,128,64,64), X3(1,256,32,32), X4(1,512,16,16)
        # x = self.norm(x)
        # x = self.pre_logits(x)
        x_out1 = Bs[-1]
        x_out2 = Bs[-1]
        x1_fea = []

        # Decoder mask estimation
        for x_skip, upsample, up in reversed(list(zip(Bs[:-1], self.upsamplers, self.assisting_up))):
            x_out1 = upsample(x_out1)
            x_out1 = up(torch.cat([x_out1, x_skip], 1))
            x1_fea.append(x_out1)
        x_out1 = self.upsamplers[0](x_out1)
        x1_fea.append(x_out1)

        # Decoder contour estimation
        for x_skip, upsample, up in reversed(list(zip(Bs[:-1], self.upsamplers, self.assisting_up))):
            x_out2 = upsample(x_out2)
            x_out2 = up(torch.cat([x_out2, x_skip], 1))
        x_out2 = self.upsamplers[0](x_out2)
        if self.add_output:

            x_out1 = self.conv_final1(x_out1)
            if self.num_classes > 1:
                x_out1 = F.log_softmax(x_out1, dim=1)

        if self.add_output:
            x_out2 = self.conv_final2(x_out2)
            if self.num_classes > 1:
                x_out2 = F.log_softmax(x_out2, dim=1)

        return x_out1, x_out2