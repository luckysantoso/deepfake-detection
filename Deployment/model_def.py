# Nama file: model_def.py
# Berisi semua definisi arsitektur model MAT dan komponennya.

# =============================================================
# BAGIAN 1: IMPOR PUSTAKA
# =============================================================
import os
import re
import math
import collections
from functools import partial
import logging
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
import numpy as np

# Coba impor pustaka opsional
try:
    import kornia
except ImportError:
    kornia = None
    logging.warning("Pustaka `kornia` tidak ditemukan. Komponen AGDA tidak akan berfungsi jika dipanggil.")

try:
    import timm
except ImportError:
    timm = None
    logging.warning("Pustaka `timm` tidak ditemukan. Backbone dari timm (ConvNeXt, ResNet) tidak akan berfungsi.")

# =============================================================
# BAGIAN 2: FUNGSI UTILITAS DAN PREPROCESSING
# =============================================================

# Parameter Global dan Utilitas Dasar
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def round_filters(filters, global_params):
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

def drop_connect(inputs, p, training):
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output

class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()
    def forward(self, input):
        return input

# Utilitas Konvolusi dengan Padding
def get_same_padding_conv2d(image_size=None):
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)

class Conv2dDynamicSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2dStaticSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()
    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

# Utilitas untuk Parameter dan Konfigurasi Backbone
class BlockDecoder(object):
    @staticmethod
    def _decode_block_string(block_string):
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))
        return BlockArgs(
            kernel_size=int(options['k']), num_repeat=int(options['r']),
            input_filters=int(options['i']), output_filters=int(options['o']),
            expand_ratio=int(options['e']), id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])
    @staticmethod
    def decode(string_list):
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

def efficientnet_params(model_name):
    params_dict = {
        'efficientnet-b0': (1.0, 1.0, 224, 0.2), 'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3), 'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4), 'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5), 'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5), 'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]

def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, image_size=None, num_classes=1000):
    blocks_args_str = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args_str)
    global_params = GlobalParams(
        batch_norm_momentum=0.99, batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate, drop_connect_rate=drop_connect_rate,
        num_classes=num_classes, width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient, depth_divisor=8,
        min_depth=None, image_size=image_size,
    )
    return blocks_args, global_params

def get_model_params(model_name, override_params):
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError(f'model name is not pre-defined: {model_name}')
    if override_params:
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params

# =============================================================
# BAGIAN 3: DEFINISI ARSITEKTUR BACKBONE
# =============================================================

# --- ConvNeXt Backbone ---
class ConvNeXtBackbone(nn.Module):
    def __init__(self, model_name='convnext_base.fb_in22k_ft_in1k', pretrained=True, num_classes=2):
        super().__init__()
        if timm is None:
            raise RuntimeError("Library `timm` diperlukan untuk ConvNeXtBackbone.")
        self.net = timm.create_model(
            model_name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3)
        )
        self.num_global_features = self.net.feature_info.channels()[-1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.num_global_features, num_classes)
        self.stage_map = {'s1': 0, 's2': 1, 's3': 2, 's4': 3}
    def forward(self, x):
        features_list = self.net(x)
        layers = {stage_name: features_list[idx] for stage_name, idx in self.stage_map.items()}
        global_features = features_list[-1]
        layers['final_conv'] = global_features
        layers['logits'] = self.fc(self.global_pool(global_features).flatten(1))
        return layers

# --- ResNet Backbone ---
class ResNetBackbone(nn.Module):
    def __init__(self, model_name='resnet152', pretrained=True, num_classes=2):
        super().__init__()
        if timm is None:
            raise RuntimeError("Library `timm` diperlukan untuk ResNetBackbone.")
        self.net = timm.create_model(
            model_name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3)
        )
        self.num_global_features = self.net.feature_info.channels()[-1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.num_global_features, num_classes)
        self.stage_map = {'s1': 0, 's2': 1, 's3': 2, 's4': 3}
    def forward(self, x):
        features_list = self.net(x)
        layers = {stage_name: features_list[idx] for stage_name, idx in self.stage_map.items()}
        global_features = features_list[-1]
        layers['final_conv'] = global_features
        layers['logits'] = self.fc(self.global_pool(global_features).flatten(1))
        return layers

# =============================================================
# BAGIAN 4: DEFINISI KOMPONEN DAN ARSITEKTUR MAT
# =============================================================

class AttentionMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionMap, self).__init__()
        self.register_buffer('mask', torch.zeros([1, 1, 24, 24]))
        if out_channels > 0:
            self.mask[0, 0, 2:-2, 2:-2] = 1
        self.num_attentions = out_channels
        self.conv_extract = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if out_channels > 0 else nn.Identity()

    def forward(self, x):
        if self.num_attentions == 0:
            return torch.ones([x.shape[0], 1, 1, 1], device=x.device)
        x = F.relu(self.bn1(self.conv_extract(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x = F.elu(x) + 1
        mask = self.mask.to(x.device)
        if x.shape[2:4] != mask.shape[2:4] and mask.sum() > 0:
            mask = F.interpolate(mask, (x.shape[2], x.shape[3]), mode='nearest')
        return x * mask

class AttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, features, attentions, norm=2):
        H, W = features.shape[-2:]
        if (attentions.shape[-2:] != (H,W)):
            attentions_resized = F.interpolate(attentions, size=(H, W), mode='bilinear', align_corners=True)
        else:
            attentions_resized = attentions
        
        if len(features.shape) == 4:
            feature_matrix = torch.einsum('bmhw,bchw->bmc', attentions_resized, features)
        elif len(features.shape) == 5:
            feature_matrix = torch.einsum('bmhw,bmchw->bmc', attentions_resized, features)
        else:
            raise ValueError(f"Unsupported feature shape: {features.shape}")

        if norm == 1:
            att_sum = torch.sum(attentions_resized, dim=(2,3), keepdim=True) + 1e-8
            normalized_attentions = attentions_resized / att_sum
            feature_matrix = torch.einsum('bmhw,bchw->bmc', normalized_attentions, features) if len(features.shape)==4 else torch.einsum('bmhw,bmchw->bmc', normalized_attentions, features)
        elif norm == 2:
            feature_matrix = F.normalize(feature_matrix, p=2, dim=-1)
        elif norm == 3:
            w = torch.sum(attentions_resized, dim=(2, 3)).unsqueeze(-1) + 1e-8
            feature_matrix = feature_matrix / w
        return feature_matrix

class Texture_Enhance_v2(nn.Module):
    def __init__(self, num_features, num_attentions):
        super().__init__()
        self.N_feat_per_map = num_features
        self.M = num_attentions
        self.output_features = self.N_feat_per_map
        self.output_features_d = self.N_feat_per_map
        MN = self.M * self.N_feat_per_map
        self.conv_extract = nn.Conv2d(num_features, self.N_feat_per_map, 3, padding=1)
        self.bn_extract = nn.BatchNorm2d(self.N_feat_per_map)
        self.conv0 = nn.Conv2d(MN, MN, 5, padding=2, groups=self.M)
        self.bn0 = nn.BatchNorm2d(MN)
        self.conv1 = nn.Conv2d(MN, MN, 3, padding=1, groups=self.M)
        self.bn1 = nn.BatchNorm2d(MN)
        self.conv2 = nn.Conv2d(MN*2, MN, 3, padding=1, groups=self.M)
        self.bn2 = nn.BatchNorm2d(MN)
        self.conv3 = nn.Conv2d(MN*3, MN, 3, padding=1, groups=self.M)
        self.bn3 = nn.BatchNorm2d(MN)
        self.bn4 = nn.BatchNorm2d(MN*4)
        self.conv_last = nn.Conv2d(MN*4, MN, 1, groups=self.M)
        self.bn_last = nn.BatchNorm2d(MN)

    def cat(self, a, b):
        B, C_a, H, W = a.shape
        a_r = a.view(B, self.M, -1, H, W)
        b_r = b.view(B, self.M, -1, H, W)
        return torch.cat([a_r, b_r], dim=2).view(B, -1, H, W)

    def forward(self, feature_maps_raw, attention_maps=(1,1)):
        B, N, H, W = feature_maps_raw.shape
        att_size = (int(H*attention_maps[0]), int(W*attention_maps[1])) if isinstance(attention_maps, tuple) else (attention_maps.shape[2], attention_maps.shape[3])

        base_feats = F.relu(self.bn_extract(self.conv_extract(feature_maps_raw)), inplace=True)
        feature_maps_d = F.adaptive_avg_pool2d(base_feats, att_size)
        feature_maps_hp = base_feats - F.interpolate(feature_maps_d, (H,W), mode='bilinear', align_corners=True) if (H>att_size[0] or W>att_size[1]) else base_feats

        if not isinstance(attention_maps, tuple):
            att_interp = torch.tanh(F.interpolate(attention_maps.detach(),(H,W),mode='bilinear',align_corners=True))
            current_fm = torch.cat([feature_maps_hp * att_interp[:, i:i+1] for i in range(self.M)], dim=1)
        else:
            current_fm = feature_maps_hp.repeat(1, self.M, 1, 1)

        fm0 = F.relu(self.bn0(self.conv0(current_fm)), inplace=True)
        fm1 = F.relu(self.bn1(self.conv1(fm0)), inplace=True)
        fm1_ = self.cat(fm0, fm1)
        fm2 = F.relu(self.bn2(self.conv2(fm1_)), inplace=True)
        fm2_ = self.cat(fm1_, fm2)
        fm3 = F.relu(self.bn3(self.conv3(fm2_)), inplace=True)
        fm3_ = self.cat(fm2_, fm3)
        final_conv_in = F.relu(self.bn4(fm3_), inplace=True)
        final_feats_grouped = F.relu(self.bn_last(self.conv_last(final_conv_in)), inplace=True)
        return final_feats_grouped.view(B, self.M, self.N_feat_per_map, H, W), feature_maps_d

class Auxiliary_Loss_v1(nn.Module):
    # Didefinisikan di sini, tetapi tidak akan digunakan saat inferensi
    def __init__(self, M, N_feat_per_map, C, alpha=0.05, margin=1, inner_margin=[0.01, 0.02]):
        super().__init__()
        # ... (Sisa implementasi tidak krusial untuk inferensi, jadi bisa dikosongkan)
        pass
    def forward(self, feature_map_d, attentions, y):
        return torch.tensor(0.0), None # Return loss nol saat inferensi

class MAT(nn.Module):
    def __init__(self, net='xception_model',feature_layer='b3',attention_layer='final_conv',num_classes=2, M=8,mid_dims=256,
                 dropout_rate=0.5,drop_final_rate=0.5, pretrained_backbone=False,
                 alpha=0.05,size=(380,380),margin=1,inner_margin=[0.01,0.02],
                 aux_loss_ver=1, texture_enhance_ver=2):
        super(MAT, self).__init__()
        self.num_classes, self.M = num_classes, M

        # Logika inisialisasi backbone
        if 'resnet' in net or 'resnext' in net:
            self.net = ResNetBackbone(model_name=net, pretrained=pretrained_backbone, num_classes=num_classes)
            self.feature_layer = 's2'
            self.attention_layer = 's3'
            self.GLOBAL_BRANCH_FEATURE_LAYER_NAME = 'final_conv'
        elif 'convnext' in net:
            self.net = ConvNeXtBackbone(model_name=net, pretrained=pretrained_backbone, num_classes=num_classes)
            self.feature_layer = 's2'
            self.attention_layer = 's3'
            self.GLOBAL_BRANCH_FEATURE_LAYER_NAME = 'final_conv'
        else:
            # Tambahkan placeholder untuk EfficientNet jika diperlukan, atau raise error
            raise ValueError(f"Unsupported backbone for deployment: {net}")

        # Dry run untuk mendapatkan shape
        with torch.no_grad():
            layers = self.net(torch.zeros(1, 3, *size))
        
        num_feat_backbone = layers[self.feature_layer].shape[1]
        att_in_channels = layers[self.attention_layer].shape[1]
        global_in_channels = layers[self.GLOBAL_BRANCH_FEATURE_LAYER_NAME].shape[1]
        
        # Inisialisasi komponen MAT
        self.attentions = AttentionMap(att_in_channels, self.M)
        self.atp = AttentionPooling()
        
        if texture_enhance_ver == 2:
            self.texture_enhance = Texture_Enhance_v2(num_feat_backbone, self.M)
        else:
            raise ValueError(f"Hanya TextureEnhance v2 yang didukung untuk deployment ini.")
            
        feat_per_map_te = self.texture_enhance.output_features
        
        self.projection_local = nn.Sequential(nn.Linear(self.M * feat_per_map_te, mid_dims), nn.Hardswish(), nn.Linear(mid_dims,mid_dims))
        self.project_final = nn.Linear(global_in_channels, mid_dims)
        self.ensemble_classifier_fc = nn.Sequential(nn.Linear(mid_dims*2, mid_dims), nn.Hardswish(), nn.Linear(mid_dims, num_classes))
        
        self.auxiliary_loss = Auxiliary_Loss_v1(M, self.texture_enhance.output_features_d, num_classes, alpha, margin, inner_margin)
        self.dropout = nn.Dropout2d(dropout_rate, inplace=True)
        self.dropout_final = nn.Dropout(drop_final_rate, inplace=True)

    def forward(self, x, return_attentions=False):
        # Forward pass disederhanakan untuk inferensi
        # Tambahkan argumen 'return_attentions' untuk kontrol
        
        layers = self.net(x)
        attention_maps = self.attentions(layers[self.attention_layer])
        enhanced_feats, _ = self.texture_enhance(layers[self.feature_layer], attention_maps)
        
        feat_matrix_pooled = self.atp(enhanced_feats, attention_maps).view(x.size(0), -1)
        projected_local = F.hardswish(self.projection_local(feat_matrix_pooled))
        
        global_feats = layers[self.GLOBAL_BRANCH_FEATURE_LAYER_NAME]
        summed_att = attention_maps.sum(dim=1, keepdim=True)
        pooled_global = self.atp(global_feats, summed_att, norm=1).squeeze(1)
        projected_global = F.hardswish(self.project_final(pooled_global))
        
        combined = torch.cat((projected_local, projected_global), dim=1)
        logits = self.ensemble_classifier_fc(combined)

        if return_attentions:
            return logits, attention_maps
        else:
            return logits