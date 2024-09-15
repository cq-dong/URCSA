from turtle import forward
import torch
import torch.nn as nn
import torchvision
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)

def print_network2(net):
    sum =0
    for name,param in net.named_parameters():
        mul=1
        for size_ in param.shape:
            mul*=size_
        sum+=mul
        print('%14s : %s' %(name,param.shape))
    print('Total params:', sum)
class Color_loss(nn.Module):
    def __init__(self):
        super(Color_loss, self).__init__()
    def forward(self,x,y):
        sqrt_x=torch.sqrt(torch.mul(x,x).sum())
        sqrt_y=torch.sqrt(torch.mul(y,y).sum())
        x_y=torch.mul(x,y).sum()
        return 1-x_y/(sqrt_x*sqrt_y)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class color_constency_loss(nn.Module):
    def __init__(self,):
        super(color_constency_loss, self).__init__()

    def forward(self, enhances):  
        plane_avg = enhances.mean((2, 3))  
        col_loss = torch.mean((plane_avg[:, 0] - plane_avg[:, 1]) ** 2
                              + (plane_avg[:, 1] - plane_avg[:, 2]) ** 2
                              + (plane_avg[:, 2] - plane_avg[:, 0]) ** 2)
        return col_loss

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        model = torchvision.models.vgg16(pretrained=False)  
        pre = torch.load('/root/dachuang097/train/vgg16.pth')
        model.load_state_dict(pre)
        blocks.append(model.features[:4].eval())
        blocks.append(model.features[4:9].eval())
        blocks.append(model.features[9:16].eval())
        blocks.append(model.features[16:23].eval())
        
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
   
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)) 
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
    
        if self.resize:
            input  = self.transform(input, mode='bilinear', size=(128, 128), align_corners=False) 
            target = self.transform(target, mode='bilinear', size=(128, 128), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss
#自似性损失(注意这块传入的是特征图)
class Self_loss(torch.nn.Module):
    def __init__(self):
        super(Self_loss, self).__init__()

    def forward(self,feature_img1,feature_img2,feature_out1,feature_out2):
        tem1=feature_img1-feature_img2
        tem2=feature_out1-feature_out2
        loss=torch.sqrt(((tem1-tem2)*(tem1-tem2)).mean())
        # print("loss182-self: ", loss.size())
        return loss
#临帧损失函数(传进相邻两帧增强后的图片)
class linzhen_pool(torch.nn.Module):
    def __init__(self, kernelsize=3):
        super(linzhen_pool, self).__init__()
        self.kernel_size =kernelsize

    def forward(self,img1,img2,max_scale=2):
        conv=nn.MaxPool2d(kernel_size=self.kernel_size,stride=self.kernel_size,ceil_mode=False)

        #得到rgb三个通道的最大池化结果，及池化不改变通道数
        tem1=conv(img1)
        tem2=conv(img2)
        if max_scale==4:
            tem1=conv(tem1)
            tem2=conv(tem2)
        # c,h,w=tem1.shape
        
        # 0.299R)+(0.587G)+(0.114*B)
        
        bright1=tem1[:,0,:,:]*0.299+tem1[:,1,:,:]*0.587+tem1[:,2,:,:]*0.114
        bright2=tem2[:,0,:,:]*0.299+tem2[:,1,:,:]*0.587+tem2[:,2,:,:]*0.114
        loss=(abs(bright1-bright2)).mean()
        return loss