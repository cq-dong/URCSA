import torch
import torch.nn as nn
from antialias import Downsample as downsamp
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

class Mlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class query_Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Parameter(torch.ones((1, 10, dim)), requires_grad=True)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 10, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class query_SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = query_Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_embedding, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            # nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(out_channels // 2),
            # nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class Global_pred(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_heads=4, type='exp'):
        super(Global_pred, self).__init__()
        if type == 'exp':
            self.gamma_base = nn.Parameter(torch.ones((1)), requires_grad=False) # False in exposure correction
        else:
            self.gamma_base = nn.Parameter(torch.ones((1)), requires_grad=True)  
        self.color_base = nn.Parameter(torch.eye((3)), requires_grad=True)  # basic color matrix
        # main blocks
        self.conv_large = conv_embedding(in_channels, out_channels)
        self.generator = query_SABlock(dim=out_channels, num_heads=num_heads)
        self.gamma_linear = nn.Linear(out_channels, 1)
        self.color_linear = nn.Linear(out_channels, 1)

        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name == 'generator.attn.v.weight':
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        #print(self.gamma_base)
        x = self.conv_large(x)
        x = self.generator(x)
        gamma, color = x[:, 0].unsqueeze(1), x[:, 1:]
        gamma = self.gamma_linear(gamma).squeeze(-1) + self.gamma_base
        #print(self.gamma_base, self.gamma_linear(gamma))
        color = self.color_linear(color).squeeze(-1).view(-1, 3, 3) + self.color_base
        return gamma, color


class FusionLayer(nn.Module):
    def __init__(self, inchannel, outchannel, reduction=16):
        super(FusionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inchannel // reduction, inchannel, bias=False),
            nn.Sigmoid()
        )
        self.fusion   = ConvBlock(inchannel, inchannel, 1,1,0,bias=True)
        self.outlayer = ConvBlock(inchannel, outchannel, 1, 1, 0, bias=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        avg = self.fc(avg).view(b, c, 1, 1)
        max = self.max_pool(x).view(b, c)
        max = self.fc(max).view(b, c, 1, 1)
        fusion = self.fusion(avg+max) 
        fusion = x * fusion.expand_as(x)
        fusion = fusion + x
        fusion = self.outlayer(fusion)
        return fusion


class NewEncoderBlock(nn.Module):  
    def __init__(self, input_dim, out_dim, kernel_size, stride, padding):
        super(NewEncoderBlock, self).__init__()
        self.firstconv = ConvBlock(input_size=4,output_size=input_dim,kernel_size=3,stride=1,padding=1)
        self.prelu = nn.PReLU()
        codeim = out_dim // 2
        self.conv_Encoder = ConvBlock(input_dim, codeim, kernel_size, stride, padding, isuseBN=False)
        self.conv_Offset  = ConvBlock(codeim, codeim, kernel_size, stride, padding, isuseBN=False)
        self.conv_Decoder = ConvBlock(codeim, out_dim, kernel_size, stride, padding, isuseBN=False)

    def forward(self, x):
        firstconv = self.prelu(self.firstconv(x))
        code   = self.conv_Encoder(firstconv)
        offset = self.conv_Offset(code)
        code_add = code + offset
        out    = self.conv_Decoder(code_add)
        return out

class ResidualDownSample(nn.Module):###my_down
    def __init__(self,in_channel,bias=False):
        super(ResidualDownSample,self).__init__()
        self.prelu = nn.PReLU()
        
        self.conv1 = nn.Conv2d(in_channel,in_channel,3,1,1,bias=bias)
        self.downsamp = downsamp(channels=in_channel,filt_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channel,2*in_channel,1,stride=1,padding=0,bias=bias)
    def forward(self, x):
        out = self.prelu(self.conv1(x))
        # print("198: out ",out.shape)
        # out = self.downsamp(out)
        out=nn.MaxPool2d(kernel_size=2)(out)
        # print("200: out ",out.shape)
        out = self.conv2(out)
        # print("202: out ",out.shape)
        return out

# class ResidualDownSample(nn.Module):
#     def __init__(self,in_channel,bias=False):
#         super(ResidualDownSample,self).__init__()
#         self.prelu = nn.PReLU()
        
#         self.conv1 = nn.Conv2d(in_channel,in_channel,3,1,1,bias=bias)
#         self.downsamp = downsamp(channels=in_channel,filt_size=3,stride=2)
#         self.conv2 = nn.Conv2d(in_channel,2*in_channel,1,stride=1,padding=0,bias=bias)
#     def forward(self, x):
#         out = self.prelu(self.conv1(x))
#         out = self.downsamp(out)
#         out = self.conv2(out)
#         return out

class DownSample(nn.Module):
    def __init__(self, in_channel,scale_factor=2, stride=2,kernel_size=3):
        super(DownSample,self).__init__()
        self.scale_factor=scale_factor
        self.residualdownsample=ResidualDownSample(in_channel)

    def forward(self, x):
        out = self.residualdownsample(x)
        return out

class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out

class UpSample(nn.Module):
    def __init__(self, in_channel, scale_factor=2,stride=2,kernel_size=3):
        super(UpSample,self).__init__()
        self.scale_factor=scale_factor
        self.residualupsample=ResidualUpSample(in_channel)

    def forward(self, x):
        out = self.residualupsample(x)
        return out

class EncoderBlock(nn.Module):  
    def __init__(self, input_dim, out_dim,):
        super(EncoderBlock, self).__init__()
        hidden = input_dim // 4  # 2021-3-30 8->4
        self.prelu = nn.PReLU()
        
        self.SGblock = nn.Sequential(
                        ConvBlock(input_dim,input_dim,3,1,1,isuseBN=False),
                        nn.Conv2d(input_dim,hidden,1,1,0),
                        nn.Conv2d(hidden,out_dim,1,1,0,),
                        ConvBlock(out_dim,out_dim,3,1,1,isuseBN=False))
    def forward(self, x):
        out = self.SGblock(x)
        out = out + x
        return out
class low_block(nn.Module):
    def __init__(self, dim=64,device='cuda:0'):
        super(low_block, self).__init__()
        # self.out_fushion = FusionLayer(3*dim, 3*dim)  
        self.out_conv2 = nn.Conv2d(2*dim, dim, 3, 1, 1)
        self.prelu   = torch.nn.PReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.downsample   = DownSample(in_channel=dim,scale_factor=2)
        self.downsample2x = DownSample(in_channel=2*dim,scale_factor=2)
        # self.downsample   = DownSample(in_channel=dim,scale_factor=2)
        # self.downsample2x = DownSample(in_channel=2*dim,scale_factor=2)
        self.upsample2x   = UpSample(in_channel=2*dim,scale_factor=2)
        self.upsample4x   = UpSample(in_channel=4*dim,scale_factor=2)
        self.endecoder1x  = EncoderBlock(input_dim=dim,out_dim=dim)
        self.endecoder2x  = EncoderBlock(2*dim,2*dim)
        self.endecoder4x  = EncoderBlock(4*dim,4*dim)
    
        self.outlayer = ConvBlock(2*dim,2*dim, 1, 1, 0, bias=True)
        self.transformer_max=Transformer(dim=2*dim,depth=1,heads=8)
        self.transformer_avg=Transformer(dim=2*dim,depth=1,heads=8)

        self.para = torch.nn.Parameter(torch.ones((1,2*dim,1,1)).to(device).requires_grad_()/2)
    def attention(self,feature):
        #feature  [b,c,h,w]
        b,c,h,w=feature.shape
        H_avg = feature.sum(dim=2,keepdim=False)/h #[b,c,w]
        W_avg = feature.sum(dim=3,keepdim=False)/w #[b,c,h]
        H_max,_ = feature.max(dim=2,keepdim=False) #[b,c,w]
        W_max,_ = feature.max(dim=3,keepdim=False) #[b,c,h]
        ###transformer的输入格式  [batch ,xiangliang_num, xiangliang_dim]->[b,h/w,c]
        H_avg_atten=self.transformer_avg(H_avg.permute(0,2,1)).permute(0,2,1) #[b,c,w]-[b,w,c]-[b,c,w]
        W_avg_atten=self.transformer_avg(W_avg.permute(0,2,1)).permute(0,2,1) #[b,c,h]-[b,h,c]-[b,c,h]
        H_max_atten=self.transformer_max(H_max.permute(0,2,1)).permute(0,2,1) #[b,c,w]-[b,w,c]-[b,c,w]
        W_max_atten=self.transformer_max(W_max.permute(0,2,1)).permute(0,2,1) #[b,c,h]-[b,h,c]-[b,c,h]

        H_max_atten=H_max_atten.unsqueeze(dim=2) #[b,c,1,w]
        W_max_atten=W_max_atten.unsqueeze(dim=3) #[b,c,h,1]
        H_avg_atten=H_avg_atten.unsqueeze(dim=2) #[b,c,1,w]
        W_avg_atten=W_avg_atten.unsqueeze(dim=3) #[b,c,h,1]
        hw_max=(W_max_atten@H_max_atten) #[b,c,h,1]@[b,c,1,w]->[b,c,h,w]
        hw_avg=(W_avg_atten@H_avg_atten) #[b,c,h,1]@[b,c,1,w]->[b,c,h,w]
        ###可以考虑变成hwmax*W+hw_acg*(1-w),把w设置成为参数
        ###为保证para始终在（0，1）之间，是不是应该加clip
        para=nn.Sigmoid()(self.para)
        out=torch.mul(hw_avg,para)+torch.mul(hw_max,(1-para))
        # out=torch.mul(hw_avg,self.para.clip(0,1))+torch.mul(hw_max,(1-self.para.clip(0,1)))
        return out
        # return (hw_max+hw_avg)/2
    def forward(self,f_endecoder,f_endecoder2,mode='train'):
        # here is img1
        if mode=='train':
            fullres = f_endecoder
            halfres = self.downsample(fullres)
            quarres = self.downsample2x(halfres)  
            
            ende_fullres_out = self.endecoder1x(fullres)
            ende_halfres = self.endecoder2x(halfres)
            ende_quarres = self.endecoder4x(quarres)
            
            # ende_halfres_up = self.upsample2x(ende_halfres)
            ende_quarres_up = self.upsample4x(ende_quarres)+ende_halfres
            ende_quarres_up_up = self.upsample2x(ende_quarres_up)+ende_fullres_out
            
            

            # cat_all = torch.cat((ende_fullres_out,ende_halfres_up,ende_quarres_up_up),dim=1)
            # fusion_all = self.out_fushion(cat_all) 
            cat_all = torch.cat((ende_quarres_up_up,fullres),dim=1)
            fusion_all=self.attention(cat_all)
            fusion_all=self.outlayer(fusion_all)+cat_all
            fusion_out = self.out_conv2(fusion_all)  
            fusion_out = fusion_out+fullres

            # here is img2
            fullres2 = f_endecoder2
            halfres2 = self.downsample(fullres2)
            quarres2 = self.downsample2x(halfres2)  
            
            ende_fullres_out2 = self.endecoder1x(fullres2)
            ende_halfres2 = self.endecoder2x(halfres2)
            ende_quarres2 = self.endecoder4x(quarres2)
            
            # ende_halfres_up2 = self.upsample2x(ende_halfres2)
            ende_quarres_up2 = self.upsample4x(ende_quarres2)+ende_halfres2
            ende_quarres_up_up2 = self.upsample2x(ende_quarres_up2)+ende_fullres_out2
            # cat_all = torch.cat((ende_fullres_out,ende_halfres_up,ende_quarres_up_up),dim=1)
            # fusion_all = self.out_fushion(cat_all) 
            cat_all2 = torch.cat((ende_quarres_up_up2,fullres2),dim=1)
            fusion_all2=self.attention(cat_all2)
            fusion_all2=self.outlayer(fusion_all2)+cat_all2
            fusion_out2= self.out_conv2(fusion_all2)  
            fusion_out2 = fusion_out2+fullres2
            return fusion_out,fusion_out2
        else:
            fullres = f_endecoder
            halfres = self.downsample(fullres)
            quarres = self.downsample2x(halfres)  
            
            ende_fullres_out = self.endecoder1x(fullres)
            ende_halfres = self.endecoder2x(halfres)
            ende_quarres = self.endecoder4x(quarres)
            
            # ende_halfres_up = self.upsample2x(ende_halfres)
            ende_quarres_up = self.upsample4x(ende_quarres)+ende_halfres
            ende_quarres_up_up = self.upsample2x(ende_quarres_up)+ende_fullres_out
            
            

            # cat_all = torch.cat((ende_fullres_out,ende_halfres_up,ende_quarres_up_up),dim=1)
            # fusion_all = self.out_fushion(cat_all) 
            cat_all = torch.cat((ende_quarres_up_up,fullres),dim=1)
            fusion_all=self.attention(cat_all)
            fusion_all=self.outlayer(fusion_all)+cat_all
            fusion_out = self.out_conv2(fusion_all)  
            fusion_out = fusion_out+fullres
            return fusion_out,fusion_out


class lowlightnet3(nn.Module):
    def __init__(self, input_dim=3, dim=64):
        super(lowlightnet3, self).__init__()
        inNet_dim = input_dim + 1

        self.prelu   = torch.nn.PReLU()
        self.sigmoid = torch.nn.Sigmoid()


        self.out_fushion = FusionLayer(3*dim, 3*dim)  
        self.out_conv3 = nn.Conv2d(dim, 4, 3, 1, 1)
        self.out_conv4 = nn.Conv2d(4,3,1,1,0)

        self.firstconv = nn.Sequential(ConvBlock(input_size=4,output_size=dim,kernel_size=3,stride=1,padding=1),
                                       EncoderBlock(input_dim=dim,out_dim=dim))
        
        self.block=low_block()
    def forward(self, x_ori,x_ori2, tar=None,mode='train'):
        x = x_ori
        x_bright, _ = torch.max(x_ori, dim=1, keepdim=True)

        x_in = torch.cat((x, x_bright), 1)

        x2 = x_ori2
        x_bright2, _ = torch.max(x_ori2, dim=1, keepdim=True)

        x_in2 = torch.cat((x2, x_bright2), 1)

        if mode=='train':

            f_endecoder = self.firstconv(x_in)
            f_endecoder2 = self.firstconv(x_in2)
            # channel=3dim
            # here is 1st block out
            fusion_out,fusion_out_img2=self.block(f_endecoder,f_endecoder2)
            ###glaol_net,,,,,放在fusion_out = fusion_out+fullres前面还是后面有待确定
            # fusion_out=self.gloal(img_low=fullres,img_high=fusion_out)
            # here is 2nd block out
            fusion_out2,fusion_out2_img2=self.block(fusion_out,fusion_out_img2)
            ###glaol_net,,,,,放在fusion_out = fusion_out+fullres前面还是后面有待确定
            # fusion_out=self.gloal(img_low=fullres2,img_high=fusion_out2)
            # here is 3rd block out
            fusion_out3,fusion_out3_img3=self.block(fusion_out2,fusion_out2_img2)
            ###glaol_net,,,,,放在fusion_out = fusion_out+fullres前面还是后面有待确定
            # fusion_out=self.gloal(img_low=fullres3,img_high=fusion_out3)
            # real out
            out = self.prelu(fusion_out3) 
            out = self.out_conv3(out)  
            out = out  +  x_bright
            out = self.out_conv4(out)

            out2 = self.prelu(fusion_out3_img3) 
            out2 = self.out_conv3(out2)  
            out2 = out2  +  x_bright2
            out2 = self.out_conv4(out2)
            return out,out2
        else:
            f_endecoder = self.firstconv(x_in)
            fusion_out,fusion_out_img2=self.block(f_endecoder,f_endecoder,mode='test')
            ###glaol_net,,,,,放在fusion_out = fusion_out+fullres前面还是后面有待确定
            # fusion_out=self.gloal(img_low=fullres,img_high=fusion_out)
            # here is 2nd block out
            fusion_out2,fusion_out2_img2=self.block(fusion_out,fusion_out_img2,mode='test')
            ###glaol_net,,,,,放在fusion_out = fusion_out+fullres前面还是后面有待确定
            # fusion_out=self.gloal(img_low=fullres2,img_high=fusion_out2)
            # here is 3rd block out
            fusion_out3,fusion_out3_img3=self.block(fusion_out2,fusion_out2_img2,mode='test')
            ###glaol_net,,,,,放在fusion_out = fusion_out+fullres前面还是后面有待确定
            # fusion_out=self.gloal(img_low=fullres3,img_high=fusion_out3)
            # real out
            out = self.prelu(fusion_out3) 
            out = self.out_conv3(out)  
            out = out  +  x_bright
            out = self.out_conv4(out)

            # out2 = self.prelu(fusion_out3_img3) 
            # out2 = self.out_conv3(out2)  
            # out2 = out2  +  x_bright2
            # out2 = self.out_conv4(out2)
            return out,out



############################################################################################
# Base models
############################################################################################

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=False):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.deconv(x)
        return self.act(out)


class UpBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(UpBlock, self).__init__()

        self.conv1 = DeconvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2(hr)
        return hr_weight + h_residue


class DownBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(DownBlock, self).__init__()

        self.conv1 = ConvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        lr = self.conv1(x)
        hr = self.conv2(lr)
        residue = self.local_weight1(x) - hr
        l_residue = self.conv3(residue)
        lr_weight = self.local_weight2(lr)
        return lr_weight + l_residue


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_filter)

        self.act1 = torch.nn.PReLU()
        self.act2 = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.act2(out)

        return out
class MHSA(nn.Module):
    def __init__(self, num_heads, dim,bias=False):
        super().__init__()
        # Q, K, V 转换矩阵，这里假设输入和输出的特征维度相同
        self.q = nn.Linear(dim, dim,bias=bias)
        self.k = nn.Linear(dim, dim,bias=bias)
        self.v = nn.Linear(dim, dim,bias=bias)
        self.num_heads = num_heads
	
    def forward(self, x):
        B, N, C = x.shape
        # 生成转换矩阵并分多头
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        
        # 点积得到attention score
        attn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn = attn.softmax(dim=-1)
        
        # 乘上attention score并输出
        v = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        return v

class PreNorm(nn.Module):
    '''
    :param  dim 输入维度
            fn 前馈网络层，选择Multi-Head Attn和MLP二者之一
    '''
    def __init__(self, dim, fn):
        super().__init__()
        # LayerNorm: ( a - mean(last 2 dim) ) / sqrt( var(last 2 dim) )
        # 数据归一化的输入维度设定，以及保存前馈层
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    # 前向传播就是将数据归一化后传递给前馈层
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
        
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads):
        super().__init__()
        # 设定depth个encoder相连，并添加残差结构
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # self.layers.append(nn.ModuleList([PreNorm(dim, MHSA(num_heads = heads, dim=dim))]))
            self.layers.append(PreNorm(dim, MHSA(num_heads = heads, dim=dim)))
                # PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
    def forward(self, x):
        # 每次取出包含Norm-attention和Norm-mlp这两个的ModuleList，实现残差结构
        for attn in self.layers:
            x = attn.forward(x) + x
        return x

if __name__=='__main__':
    data=torch.randn(1,3,64,64).cuda()
    model = lowlightnet3().cuda()
    out = model(data)
    print(out.shape)