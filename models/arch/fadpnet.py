# Dependencies:
# torch>=1.12, einops, numpy, opencv-python
# For Mamba modules: please refer to MambaIR (https://github.com/csguoh/MambaIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.arch.blocks import LayerNorm as LayerNorm
from models.arch.blocks import flow_warp
from models.arch.mambair_arch import *
from einops import rearrange


###############------Channel_Attention_Block(CAB)------###############
class ChannelAttention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):   
        return self.cab(x)
###############------Channel_Attention_Block(CAB)------###############
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, input_resolution=None):
        super(FeedForward, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

####################-----lowpass-----#####################

class lowpass(nn.Module):
    def __init__(self,
                 dim,
                 d_state=8,
                 mlp_ratio=2.,
                 num_tokens=32,
                 inner_rank=32,
                 squeeze_factor=16,
                 ffn_expansion_factor=1.,
                 bias=False,
                 **kwargs):
        super().__init__()
        self.mlp_ratio = mlp_ratio
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1))  
        self.scale2 = nn.Parameter(torch.ones(1, dim, 1, 1)) 
        self.embeddingA = nn.Embedding(inner_rank, d_state)
        self.attn = ASSM(dim,d_state,num_tokens=num_tokens,inner_rank=inner_rank,mlp_ratio=mlp_ratio)
        self.norm1 = LayerNorm(dim,LayerNorm_type='WithBias') 
        self.norm2 = LayerNorm(dim,LayerNorm_type='WithBias') 
        self.conv_block = ChannelAttention(dim, squeeze_factor=squeeze_factor)
        self.ffd1 = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffd2 = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffd3 = FeedForward(dim, ffn_expansion_factor, bias)
        
    def forward(self, x):
    
        b, c, h, w =x.shape
        shortcut = x    
        x = self.norm1(x)
        x_norm = rearrange(x, 'b c h w -> b (h w) c')
        conv_x = self.conv_block(x)
        attn_out = self.attn(x_norm, (h,w), self.embeddingA)
        attn_out = rearrange(attn_out, 'b (h w) c -> b c h w', h=h, w=w)+shortcut
        x2 =  attn_out*self.scale+ conv_x*self.scale2
        x3 = self.norm2(x2)  
        x3= self.ffd1(x3) + x2
        x3= self.ffd2(x3) + x3
        x3= self.ffd3(x3) + x3
        return x3


####################-----highpass-----#################### 
        
class mixblock(nn.Module):
    def __init__(self, n_feats):
        super(mixblock, self).__init__()
        
        self.detail_branch = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        out = self.detail_branch(x) + x
        return out
##################################################################### 

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_groups=1, squeeze_factor=16):
        super(MultiScaleConv, self).__init__()
        out_channels = out_channels if out_channels else in_channels

        self.norm = LayerNorm(in_channels, LayerNorm_type='WithBias')
        self.conv7x7 = DepthwiseSeparableConv(in_channels, in_channels//2, kernel_size=7, groups=conv_groups)
        self.conv5x5 = DepthwiseSeparableConv(in_channels, in_channels//2, kernel_size=5, groups=conv_groups)
        self.conv = nn.Sequential(ChannelShuffle(groups=4),nn.GELU(),
                                  nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)) 
        
    def forward(self, x):
        feature7 = self.conv7x7(x)
        feature5 = self.conv5x5(x)
        combined = torch.concat([feature5, feature7],dim=1)
        combined = self.conv(combined)
        return  combined + x
        
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super(DepthwiseSeparableConv, self).__init__()
        padding = (kernel_size // 2) 
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.relu(x)
        x = self.pointwise(x)
        return x


class highpass(nn.Module):
    def __init__(self, in_channel, out_channel, num_layer=4, num_heads=4, ratio=1, LayerNorm_type='WithBias' ):
        super(highpass, self).__init__()
        internal_channel = in_channel * ratio
        
        #self.reduce_channels = nn.Conv2d(in_channel, in_channel//4, kernel_size=1, stride=1, bias=False)
        
        layers = []
        layers.append(nn.Conv2d(in_channel, in_channel//2, kernel_size=1, stride=1, padding=0,bias=False))
        for i in range(num_layer):
             layers.append(MultiScaleConv(in_channel//2, in_channel//2, conv_groups=in_channel//2))
        layers.append(nn.Conv2d(in_channel//2, in_channel, kernel_size=1, stride=1, padding=0,bias=False))
        self.layers = nn.Sequential(*layers)
        
        #self.expand_channels = nn.Conv2d(in_channel//4, in_channel, kernel_size=1, stride=1, bias=False)
        
        self.norm1 = LayerNorm(in_channel, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(in_channel, LayerNorm_type='WithBias')
        
        self.attn1 = Attention(out_channel,num_heads=num_heads)
        self.attn2 = Attention(out_channel,num_heads=num_heads)
        
        self.enhance_block1 = nn.Sequential(mixblock(out_channel),mixblock(out_channel))
        self.enhance_block2 = nn.Sequential(mixblock(out_channel),mixblock(out_channel))
        self.attn_out = dw_attention(out_channel) 
        self.conv = nn.Conv2d(out_channel*3, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
       
    def forward(self, x):
        norm1 = self.norm1(x)
        x1 = self.layers(norm1)+ x
        x2 = self.enhance_block1(x1)
        x3 = self.attn1(x2)+ x2
        norm2 = self.norm2(x3)
        out = self.enhance_block2(norm2)  
        out = self.attn2(out)
        return out + x

#####################################################################
        
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        if num_channels % self.groups != 0:
            raise ValueError("Number of channels must be divisible by groups")
        x = x.view(batch_size, self.groups, num_channels // self.groups, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, num_channels, height, width)
        return x


import torch
import torch.nn as nn
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=4, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        head_dim = dim // num_heads
        self.temp_gen = nn.Sequential(
             nn.Linear(dim, 2*dim),
             nn.GELU(),
             nn.Linear(2*dim, 1)
        )

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.pos_embed = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim // 2,dim, 1)
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b, c, h, w = x.shape
        
        global_features = x.mean(dim=(2, 3)) 
        temp = self.temp_gen(global_features).sigmoid().unsqueeze(-1).unsqueeze(-1)
        
      
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1) 

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * temp
        attn = attn.softmax(dim=-1)

        pos = self.pos_embed(x)
        
        out = attn @ v 
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out += pos.sigmoid()

        return self.project_out(out)

####################################################################
        
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True) 
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(combined)
        return self.sigmoid(attention)

###############################################################################
class Downsample(nn.Module):
    def __init__(self, in_channels, scale=2):
        super(Downsample, self).__init__()
        self.scale = scale
        layers = []
        num_layers = int(math.log(scale, 2))  
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.PixelUnshuffle(2))

        self.downsample = nn.Sequential(*layers)

    def forward(self, x):
        return self.downsample(x)
        
class Upsample(nn.Module):
    def __init__(self, in_channels, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale
        layers = []
        num_layers = int(math.log(scale, 2))  
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels *2, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.PixelShuffle(2))
            in_channels /=2

        self.upsample = nn.Sequential(*layers)

    def forward(self, x):
        return self.upsample(x)        

        
class ConvDownsample(nn.Module):
    def __init__(self, in_channels, scale=2, kernel_size=3, stride=2, padding=1):
        super(ConvDownsample, self).__init__()
        layers = []
        num_layers = int(math.log(scale, 2))
        for i in range(num_layers):  
            layers.append(nn.Conv2d(in_channels, in_channels*2, kernel_size, stride, padding))
            in_channels *=2 
        
        self.downsample = nn.Sequential(*layers)

    def forward(self, x):
        return self.downsample(x)
####################################################################################
class basicblock(nn.Module):
    def __init__(self,
                 n_feats,
                 num_blocks,
                 depth=[3, 3, 3],
                 LayerNorm_type='BiasFree',
                 ):
        super(basicblock, self).__init__()
        self.n_feats = n_feats

        self.down_4 = ConvDownsample(n_feats, 4)
        self.down_2 = ConvDownsample(n_feats, 2)

        self.down_low8 = Downsample(n_feats * 4)
        self.down_low4 = Downsample(n_feats * 2)
        self.down_low2 = Downsample(n_feats)

        self.out4_2 = Upsample(n_feats * 4)
        self.out2_1 = Upsample(n_feats * 2)

        self.highpass4_layers = nn.ModuleList(
            [highpass(n_feats * 4, n_feats * 4, num_layer=4, num_heads=8) for _ in range(depth[2])])
        self.highpass2_layers = nn.ModuleList(
            [highpass(n_feats * 2, n_feats * 2, num_layer=4, num_heads=4) for _ in range(depth[1])])
        self.highpass_layers = nn.ModuleList(
            [highpass(n_feats, n_feats, num_layer=4, num_heads=4) for _ in range(depth[0])])

        self.lowpass4_layers = nn.ModuleList([lowpass(n_feats * 4) for _ in range(num_blocks[2])])
        self.lowpass2_layers = nn.ModuleList([lowpass(n_feats * 2) for _ in range(num_blocks[1])])
        self.lowpass_layers = nn.ModuleList([lowpass(n_feats) for _ in range(num_blocks[0])])

        self.reduce_chan_level2 = nn.Conv2d(n_feats * 4, n_feats * 2, kernel_size=1)
        self.reduce_chan_level = nn.Conv2d(n_feats * 2, n_feats, kernel_size=1)

        self.offset = Predictor(n_feats)
        self.conv = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=False))
        self.conv_2 = nn.Sequential(nn.Conv2d(n_feats * 2, n_feats * 2, 3, 1, 1, bias=False))
        self.conv_4 = nn.Sequential(nn.Conv2d(n_feats * 4, n_feats * 4, 3, 1, 1, bias=False))

        self.conv = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=False))

    def forward(self, input):

        offsets = self.offset(input)
        x = input
        x4 = self.down_4(x)
        low_4 = self.down_low8(x4)
        high_4 = x4 - F.interpolate(low_4, size=x4.size()[-2:], mode='bilinear', align_corners=True)
        for highpass4_layer in self.highpass4_layers:
            high_4 = highpass4_layer(high_4)
        for lowpass4_layer in self.lowpass4_layers:
            low_4 = lowpass4_layer(low_4)
        low_4 = F.interpolate(low_4, size=x4.size()[-2:], mode='bilinear', align_corners=True)
        out4 = self.conv_4(low_4 + high_4) + x4
        out4 = self.out4_2(out4)

        x2 = self.down_2(x)
        low_2 = self.down_low4(x2)
        high_2 = x2 - F.interpolate(low_2, size=x2.size()[-2:], mode='bilinear', align_corners=True)
        for highpass2_layer in self.highpass2_layers:
            high_2 = highpass2_layer(high_2)
        for lowpass2_layer in self.lowpass2_layers:
            low_2 = lowpass2_layer(low_2)
        low_2 = F.interpolate(low_2, size=x2.size()[-2:], mode='bilinear', align_corners=True)
        out2 = self.conv_2(low_2 + high_2) + x2
        out2 = torch.concat([out4, out2], dim=1)
        out2 = self.reduce_chan_level2(out2)
        out2 = self.out2_1(out2)

        low = self.down_low2(x)
        high = x - F.interpolate(low, size=x.size()[-2:], mode='bilinear', align_corners=True)
        for highpass_layer in self.highpass_layers:
            high = highpass_layer(high)
        for lowpass_layer in self.lowpass_layers:
            low = lowpass_layer(low)
        low = F.interpolate(low, size=x.size()[-2:], mode='bilinear', align_corners=True)
        out = self.conv(low + high) + x
        out = torch.concat([out2, out], dim=1)
        out = self.reduce_chan_level(out)
        out = out + flow_warp(out, offsets.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border')

        return out + input


######################################################################
class dw_attention(nn.Module):
    def __init__(self, nin):
        super(dw_attention, self).__init__()
        self.conv_dws = nn.Conv2d(nin, nin, kernel_size=1, stride=1, padding=0, groups=nin)
        self.relu_dws = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_point = nn.Conv2d(nin, 1, kernel_size=1, stride=1, padding=0, groups=1)
        self.relu_point = nn.ReLU(inplace=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv_dws(x)
        out = self.relu_dws(out)
        out = self.maxpool(out)
        out = self.conv_point(out)
        out = self.relu_point(out)
        out = self.softmax(out)
        out = torch.mul(out, x)
        out = out + x

        return out


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, dim=40, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Sequential(nn.Conv2d(in_c, dim, kernel_size=3, stride=1, padding=1, bias=bias))

    def forward(self, x):
        x = self.proj(x)
        return x


class FADPNet_sr(nn.Module):
    def __init__(self,
                 in_chans=3,
                 ratio=1,
                 num_blocks=3,
                 use_chk=False,
                 norm_layer=LayerNorm,
                 bias=False,
                 **kwargs):
        super().__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans
        dim = 32
        self.use_chk = use_chk
        self.num_blocks = num_blocks
        self.patch_embed = OverlapPatchEmbed(num_in_ch, dim)
        self.main = basicblock(dim, num_blocks=[2, 2, 2], depth=[2, 3, 4])

        self.out_conv = nn.Conv2d(dim, 3, 3, 1, 1)

    def forward(self, x):
        inp_enc_level1 = self.patch_embed(x)
        out_dec_level1 = self.main(inp_enc_level1)
        out_dec_level1 = self.out_conv(out_dec_level1) + x

        return out_dec_level1

##################################################################################
class Norm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Predictor(nn.Module):
    """ Offsets Predictor
    """
    def __init__(self, dim):
        super().__init__()

        cdim = dim
        
        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim//4, 1),
            Norm(cdim//4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.out_offsets = nn.Sequential(
            nn.Conv2d(cdim//4, cdim//10, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(cdim//10, 2, 1),
        )       

    def forward(self, input_x):

        x = self.in_conv(input_x)
        offsets = self.out_offsets(x)
        offsets = offsets.tanh().mul(8)
        return offsets
