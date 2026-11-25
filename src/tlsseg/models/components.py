"""Supplied SSFPN attention and feature-reconstruction components.

Function/class bodies are preserved from the user-supplied snapshot
``decoder_refine_search_20261008/code/frozen_base/tls_ssfp_dit.py``.
The unused legacy ResNet whole-model and direct pixel classifier are omitted.
No new license is assigned here; see the repository provenance documentation.

Despite its historical name, ``SynchronizedBatchNorm2d`` is GroupNorm and has
no running batch statistics or distributed synchronization.
"""
import math

import torch
from torch import einsum, nn
from torch.nn import functional as F
from einops import rearrange

SOURCE_COMPONENT_SHA256 = "4818c73a391d6772c62fbb7d59b99d902442ba2c7e02aed8cf64c591619a5f4e"


DENOISER_DIT_FEATURE_SIZE = 56


DENOISER_DIT_PATCH_SIZE = 4


DENOISER_DIT_EMBED_DIM = 128


DENOISER_DIT_DEPTH = 2


DENOISER_DIT_NUM_HEADS = 4


DENOISER_DIT_MLP_RATIO = 4.0


def _choose_group_count(num_channels):
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


class SynchronizedBatchNorm2d(nn.GroupNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        del momentum, track_running_stats
        super().__init__(num_groups=_choose_group_count(num_features), num_channels=num_features, eps=eps, affine=affine)


class conv_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=(1, 1), group=1, bn_act=False, bias=False):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=group, bias=bias)
        self.bn = SynchronizedBatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class SpatialAttention(nn.Module):

    def __init__(self, in_ch, out_ch, droprate=0.15):
        super(SpatialAttention, self).__init__()
        self.conv_sh = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.bn_sh1 = SynchronizedBatchNorm2d(in_ch)
        self.bn_sh2 = SynchronizedBatchNorm2d(in_ch)
        self.fuse = conv_block(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.drop = droprate

    def forward(self, x):
        b, c, h, w = x.size()
        avg_h = self.bn_sh1(F.conv2d(F.avg_pool2d(x, [1, w]), self.conv_sh.weight))
        avg_w = self.bn_sh2(F.conv2d(F.avg_pool2d(x, [h, 1]), self.conv_sh.weight))
        att = torch.softmax(torch.matmul(avg_h, avg_w), 1)
        fusion = att * avg_h + att * avg_w
        out = F.dropout(self.fuse(fusion), p=self.drop, training=self.training)
        return F.relu(self.gamma * out + (1 - self.gamma) * x)


class ChannelWise(nn.Module):

    def __init__(self, channel):
        super(ChannelWise, self).__init__()
        self.conv_b = nn.Conv2d(channel * 2, channel // 8, 1)
        self.conv_c = nn.Conv2d(channel * 2, channel // 8, 1)
        self.conv_d = nn.Conv2d(channel * 2, channel, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_cat = torch.cat([F.avg_pool2d(x, 3, 1, 1), F.max_pool2d(x, 3, 1, 1)], dim=1)
        b, _, h, w = x_cat.size()
        f_b = self.conv_b(x_cat).view(b, -1, h * w).permute(0, 2, 1)
        f_c = self.conv_c(x_cat).view(b, -1, h * w)
        attn = self.softmax(torch.bmm(f_b, f_c))
        f_d = self.conv_d(x_cat).view(b, -1, h * w)
        f_e = torch.bmm(f_d, attn.permute(0, 2, 1)).view(b, -1, h, w)
        return self.alpha * f_e + x


class Attention(nn.Module):

    def __init__(self, size=224, stride=32, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.resolution = int(size / stride)
        self.dim = int(math.pow(size / stride, 2))
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(self.dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, self.dim), nn.Dropout(dropout))

    def forward(self, x1, x2):
        qkv1 = self.to_qkv(x1).chunk(3, dim=-1)
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv1)
        qkv2 = self.to_qkv(x2).chunk(3, dim=-1)
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv2)
        attn1 = self.attend(einsum('b h i d, b h j d -> b h i j', q1, k2) * self.scale)
        out1 = einsum('b h i j, b h j d -> b h i d', attn1, v2)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out1 = self.to_out(out1).view(out1.shape[0], -1, self.resolution, self.resolution)
        attn2 = self.attend(einsum('b h i d, b h j d -> b h i j', q2, k1) * self.scale)
        out2 = einsum('b h i j, b h j d -> b h i d', attn2, v1)
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out2 = self.to_out(out2).view(out2.shape[0], -1, self.resolution, self.resolution)
        return (out1, out2)


class PyrmidFusionNet(nn.Module):

    def __init__(self, ch_high, ch_low, ch_out, s, classes=2):
        super(PyrmidFusionNet, self).__init__()
        self.lateral_low = conv_block(ch_high, ch_low, 3, 1, bn_act=True, padding=1)
        self.conv_low = conv_block(ch_low, ch_out, 3, 1, bn_act=True, padding=1)
        self.sa = SpatialAttention(ch_out, ch_out)
        self.att = Attention(size=224, stride=s, heads=8, dim_head=64)
        self.conv_high = conv_block(ch_low, ch_out, 3, 1, bn_act=True, padding=1)
        self.ca = ChannelWise(ch_out)
        self.FRB = nn.Sequential(conv_block(2 * ch_low, ch_out, 1, 1, bn_act=True, padding=0), conv_block(ch_out, ch_out, 3, 1, bn_act=True, padding=1))
        self.apf = conv_block(ch_out, ch_out, 3, 1, padding=1, group=1, bn_act=True)

    def forward(self, x_h, x_l):
        lat_l = self.lateral_low(x_l)
        h_up = F.interpolate(x_h, size=lat_l.size()[2:], mode='bilinear', align_corners=False)
        concate = self.FRB(torch.cat([lat_l, h_up], 1))
        sa_in = self.sa(concate).view(concate.shape[0], concate.shape[1], -1)
        ca_in = self.ca(concate).view(concate.shape[0], concate.shape[1], -1)
        ca_out, sa_out = self.att(ca_in, sa_in)
        return self.apf(torch.mul(sa_out, self.conv_high(h_up)) + torch.mul(ca_out, self.conv_low(lat_l)))


class GlobalFeatureUpsample(nn.Module):

    def __init__(self, in_ch, out_ch, low_ch):
        super(GlobalFeatureUpsample, self).__init__()
        self.conv1 = conv_block(in_ch, out_ch, 1, 1, 0, bn_act=True)
        self.conv2 = nn.Sequential(conv_block(in_ch, out_ch, 1, 1, 0), nn.ReLU(True))
        self.conv3 = conv_block(out_ch, low_ch, 1, 1, 0, bn_act=True)

    def forward(self, x_g, y_h):
        x_g = self.conv1(x_g)
        y_h = nn.AdaptiveAvgPool2d(1)(self.conv2(y_h))
        return self.conv3(y_h.expand_as(x_g) * x_g)


class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        time = time.float()
        device = time.device
        half_dim = self.dim // 2
        if half_dim <= 1:
            return time.unsqueeze(-1)
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def modulate(features, shift, scale):
    return features * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTDenoiseBlock(nn.Module):

    def __init__(self, hidden_size=128, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f'hidden_size={hidden_size} must be divisible by num_heads={num_heads}.')
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, hidden_size))
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)

    def forward(self, tokens, condition):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation(condition).chunk(6, dim=1)
        attention_input = modulate(self.norm1(tokens), shift_msa, scale_msa)
        attention_output, _ = self.attention(attention_input, attention_input, attention_input, need_weights=False)
        tokens = tokens + gate_msa.unsqueeze(1) * attention_output
        mlp_input = modulate(self.norm2(tokens), shift_mlp, scale_mlp)
        tokens = tokens + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        return tokens


class DiTNoisePredictionHead(nn.Module):

    def __init__(self, channels=32, feature_size=56, patch_size=4, hidden_size=128, depth=2, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        if feature_size % patch_size != 0:
            raise ValueError(f'feature_size={feature_size} must be divisible by patch_size={patch_size}.')
        self.channels = channels
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.grid_size = feature_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_embedding = nn.Conv2d(channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.blocks = nn.ModuleList([DiTDenoiseBlock(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-06)
        self.unpatch = nn.ConvTranspose2d(hidden_size, channels, kernel_size=patch_size, stride=patch_size)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.unpatch.weight)
        if self.unpatch.bias is not None:
            nn.init.zeros_(self.unpatch.bias)

    def forward(self, features, condition):
        batch_size, _, height, width = features.shape
        token_map = self.patch_embedding(features)
        token_height, token_width = token_map.shape[-2:]
        tokens = token_map.flatten(2).transpose(1, 2)
        if tokens.size(1) == self.position_embedding.size(1):
            position_embedding = self.position_embedding
        else:
            position_map = self.position_embedding.transpose(1, 2).reshape(1, self.hidden_size, self.grid_size, self.grid_size)
            position_map = F.interpolate(position_map, size=(token_height, token_width), mode='bicubic', align_corners=False)
            position_embedding = position_map.flatten(2).transpose(1, 2)
        tokens = tokens + position_embedding
        for block in self.blocks:
            tokens = block(tokens, condition)
        tokens = self.final_norm(tokens)
        feature_map = tokens.transpose(1, 2).reshape(batch_size, self.hidden_size, token_height, token_width)
        feature_map = self.unpatch(feature_map)
        if feature_map.shape[-2:] != (height, width):
            feature_map = F.interpolate(feature_map, size=(height, width), mode='bilinear', align_corners=False)
        return feature_map


class FeatureDenoiseNet(nn.Module):

    def __init__(self, channels=32, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.GELU())
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_emb1 = nn.Linear(time_emb_dim, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_emb2 = nn.Linear(time_emb_dim, channels)
        self.act = nn.GELU()
        self.bn1 = SynchronizedBatchNorm2d(channels)
        self.bn2 = SynchronizedBatchNorm2d(channels)
        self.dit_noise_head = DiTNoisePredictionHead(channels=channels, feature_size=DENOISER_DIT_FEATURE_SIZE, patch_size=DENOISER_DIT_PATCH_SIZE, hidden_size=DENOISER_DIT_EMBED_DIM, depth=DENOISER_DIT_DEPTH, num_heads=DENOISER_DIT_NUM_HEADS, mlp_ratio=DENOISER_DIT_MLP_RATIO)

    def forward(self, x, time):
        t_emb = self.time_mlp(time)
        h = self.bn1(self.act(self.conv1(x)))
        h = h + self.time_emb1(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.bn2(self.act(self.conv2(h)))
        h = h + self.time_emb2(t_emb).unsqueeze(-1).unsqueeze(-1)
        return self.dit_noise_head(h, t_emb)


class FeatureDiffusion(nn.Module):

    def __init__(self, channels=32, timesteps=50):
        super().__init__()
        self.timesteps = timesteps
        self.denoise_net = FeatureDenoiseNet(channels=channels)
        betas = torch.linspace(0.0001, 0.02, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def forward(self, f):
        b = f.shape[0]
        if self.training:
            t = torch.randint(0, self.timesteps, (b,), device=f.device).long()
            noise = torch.randn_like(f)
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(b, 1, 1, 1)
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(b, 1, 1, 1)
            f_noisy = sqrt_alpha_t * f + sqrt_one_minus_alpha_t * noise
            noise_pred = self.denoise_net(f_noisy, t)
            diff_loss = F.mse_loss(noise_pred, noise)
            f_refined = (f_noisy - sqrt_one_minus_alpha_t * noise_pred) / (sqrt_alpha_t + 1e-08)
            return (f_refined, diff_loss)
        else:
            t = torch.zeros(b, device=f.device)
            noise_pred = self.denoise_net(f, t)
            sqrt_alpha_0 = self.sqrt_alphas_cumprod[0].view(1, 1, 1, 1)
            sqrt_one_minus_alpha_0 = self.sqrt_one_minus_alphas_cumprod[0].view(1, 1, 1, 1)
            f_refined = (f - sqrt_one_minus_alpha_0 * noise_pred) / (sqrt_alpha_0 + 1e-08)
            diff_loss = f.new_tensor(0.0)
            return (f_refined, diff_loss)
