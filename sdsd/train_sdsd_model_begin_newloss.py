import argparse
import itertools
import os
import time
from os import listdir
from os.path import join
import random
import cv2
import torch.utils.data as data
import numpy as np
import skimage
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageOps, ImageEnhance
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.data import DataLoader
import lib.pytorch_ssim as pytorch_ssim
from EDnet_loss import TVLoss, print_network, VGGPerceptualLoss,Self_loss,linzhen_pool
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
import skimage.filters.rank as sfr
from skimage.morphology import disk  #生成扁平的盘状结构元素，其主要参数是生成圆盘的半径
from sdsd_basemodel import lowlightnet3 as EDnet
#####################################################参数设置
def cfg():
    parser = argparse.ArgumentParser(description='low-video image enhancement by dachuang')
    parser.add_argument('--trainset', type=str, default='/root/autodl-tmp/SDSD_224/indoor/', help='location of trainset')
    parser.add_argument('--testset', type=str, default='/root/autodl-tmp/SDSDtest/', help='location of testset')

    parser.add_argument('--output', default='output', help='location to save output images')
    parser.add_argument('--modelname', default='sdsd_basemodel_begin_newloss', help='define model name')

    parser.add_argument('--deviceid', default='0', help='selecte which gpu device')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning Rate')
    parser.add_argument('--lr_decay', type=float, default=1.2, help='Every 50 epoch, lr decay')
    parser.add_argument('--batchSize', type=int, default=10,help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=20,help='number of epochs to train for')
    parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')

    parser.add_argument('--start_iter', type=int, default=0, help='Starting Epoch')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
    parser.add_argument('--patch_size', type=int, default=224, help='Size of cropped LR image')
    parser.add_argument('--patch_sizeH', type=int, default=128, help='Size of cropped LR image')
    parser.add_argument('--patch_sizeW', type=int, default=240, help='Size of cropped LR image')
    parser.add_argument('--loss_txt', type=str, default="./loss_txt.txt", help='记录各个损失')
    parser.add_argument('--output_100', type=str, default="/root/autodl-tmp/output_every_100/", help='每100次更新存储增强图片')

    opt = parser.parse_args()
    return opt
# Using model_LOL by default
#返回视频列表地址
def test_list(opt):
    new_test=['pair13','pair15','pair21','pair23','pair31','pair33','pair50','pair52','pair58','pair60','pair68','pair70']
    data_path=opt.test
    print("当前测试数据集地址为：",data_path)
    data_low=data_path+'/low/'
    data_high=data_path+'/high/'
    #pic_low:为二维列表，第一维度是第一帧，第二维度是第二帧
    pic_low=[]
    for index,data in enumerate(new_test):
        for indexj,d in enumerate(listdir(data_low+data)):
            if indexj<30:
                pic_low.append(data_low+data+'/'+str(d))
    pic_high=[]
    print(len(data_high+data))
    for index,data in enumerate(new_test):
        for indexj,d in enumerate(listdir(data_high+data)):
            if indexj<30:
                pic_high.append(data_high+data+'/'+str(d))
    # vedio_low=[pic_low[:-1],pic_low[1:]]
    # vedio_high=[pic_high[:-1],pic_high[1:]]
    print("testdata_num")
    print(len(pic_low),len(pic_high))
    return pic_low,pic_high
def all_list(opt):
    data_path=opt.trainset
    data_low=data_path+'/low/'
    data_high=data_path+'/high/'
    print(data_low,data_high)
    new_test=['pair13','pair15','pair21','pair23','pair31','pair33','pair50','pair52','pair58','pair60','pair68','pair70']
    #pic_low:为二维列表，第一维度是第一帧，第二维度是第二帧
    pic_low=[]
    for index,data in enumerate(listdir(data_low)):
        if data in new_test:
            continue
        for indexj,d in enumerate(listdir(data_low+data)):
            pic_low.append(data_low+data+'/'+str(d))
    pic_high=[]
    for index,data in enumerate(listdir(data_high)):
        if data in new_test:
            continue
        for indexj,d in enumerate(listdir(data_high+data)):
            pic_high.append(data_high+data+'/'+str(d))
    vedio_low=[pic_low[:-1],pic_low[1:]]
    vedio_high=[pic_high[:-1],pic_high[1:]]
    #若想实现前后帧的随机，可以在转置前shuffle对列表打乱,实现列表的打乱
    vedio_low=list(map(list, zip(*vedio_low)))
    random.seed(0)
    random.shuffle(vedio_low)
    vedio_low=list(map(list, zip(*vedio_low)))
    vedio_high=list(map(list, zip(*vedio_high)))
    random.seed(0)
    random.shuffle(vedio_high)
    vedio_high=list(map(list, zip(*vedio_high)))
    print("111")
    print(len(vedio_low[0]),len(vedio_low[1]),len(vedio_high[0]),len(vedio_high[1]))
    return vedio_low,vedio_high

####################################################合成亮通道
def max_box(image,kernel_size=15):
    max_image = sfr.maximum(image,disk(kernel_size))  #skimage.filters.rank.minimum()返回图像的局部最大值
    return max_image

def calculate_bright(image):
    if not isinstance(image,np.ndarray):
        raise ValueError("input image is not numpy type")  #手动抛出异常
    dark = np.maximum(image[0,:,:],image[1,:,:],image[2,:,:]).astype(np.float32) #取三个通道的最大值来获取亮通道
    dark = max_box(dark,kernel_size=15)
    return dark/255
#输入一个tensor ，C*H*W，然后返回合成后带有亮通道的tensor
def bright_channel(input):
    temout=np.array(input)
    bright=calculate_bright(temout)
    bright=np.reshape(bright,(1,bright.shape[0],bright.shape[1]))
    bright=torch.tensor(bright,dtype=float)
    input_now=torch.concat((input,bright),dim=0).float()
    return input_now
####################################################合成亮通道

#####################################################图片数据读取
def get_video_path(datapath):
    return listdir(datapath)
#判断图像
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"])
#根据地址读取图像
def load_img(filepath):

    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img
#####################################################图片数据读取


#####################################################图片数据增强与tenor
def transform(a,b):
    # transforms.Resize([224,224])注意由于get_patch函数里有尺寸调整部分，此处不在resize
    return transforms.Compose([transforms.Resize([a,b]),transforms.ToTensor(),
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
#这个可能不用
def get_patch(img_in1, img_in2,img_tar1,img_tar2, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in1.size
    # print(img_in.size)
    # (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in1,img_in2 = img_in1.crop((ty, tx, ty + tp, tx + tp)),img_in2.crop((ty, tx, ty + tp, tx + tp))
    img_tar1,img_tar2 = img_tar1.crop((ty, tx, ty + tp, tx + tp)),img_tar2.crop((ty, tx, ty + tp, tx + tp))
    # img_bic = img_bic.crop((ty, tx, ty + tp, tx + tp))

    # info_patch = {
    #    'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in1,img_in2, img_tar1,img_tar2
def augment(img_in1, img_in2,img_tar1,img_tar2, flip_h=False, rot=True, noise=False):
    info_aug = {'flip_h': True, 'flip_v': True, 'trans': True}

    if random.random() < 0.5 and flip_h:
        img_in1,img_in2 = ImageOps.flip(img_in1),ImageOps.flip(img_in2)
        img_tar1,img_tar2 = ImageOps.flip(img_tar1),ImageOps.flip(img_tar2)
        # img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in1,img_in2 = ImageOps.mirror(img_in1),ImageOps.mirror(img_in2)
            img_tar1,img_tar2 = ImageOps.mirror(img_tar1),ImageOps.mirror(img_tar2)
            # img_bic = ImageOps.mirror(img_bic)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in1,img_in2 = img_in1.rotate(180),img_in2.rotate(180)
            img_tar1,img_tar2 = img_tar1.rotate(180),img_tar2.rotate(180)
            # img_bic = img_bic.rotate(180)
            info_aug['trans'] = True
    if noise:
        # var=0.1 noise is already enormous
        # add noise is whole img added !!
        img_in1,img_in2 = np.asarray(img_in1),np.asarray(img_in2)
        img_in1 = skimage.util.random_noise(img_in1, mode='gaussian', clip=True,var=0.02).astype('uint8')
        img_in2 = skimage.util.random_noise(img_in2, mode='gaussian', clip=True,var=0.02).astype('uint8')
        img_in1,img_in2 = Image.fromarray(img_in1),Image.fromarray(img_in2)
    return img_in1,img_in2,img_tar1,img_tar2, info_aug
#####################################################图片数据增强与tenor
###添加L1正则化
def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is torch.nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)


class DatasetFromFolder(data.Dataset):
    def __init__(self, HR_list, LR_list, patch_size, upscale_factor, data_augmentation,transform=None,mode='train'):
        super(DatasetFromFolder, self).__init__()
        #print(listdir(HR_dir))
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.hr_image_filenames=HR_list
        self.lr_image_filenames=LR_list
        self.mode=mode
        # self.if_LAB = if_LAB
    def __getitem__(self, index):
        # print(self.lr_image_filenames)
        # print("index:",index,"  file_num: ",len(self.lr_image_filenames))
        target1 = load_img(self.hr_image_filenames[0][index])
        target2 = load_img(self.hr_image_filenames[1][index])
        # name = self.hr_image_filenames[index]
        # lr_name = name[:25]+'LR/'+name[28:-4]+'x4.png'
        # lr_name = name[:18] + 'LR_4x/' + name[21:]
        input1 = load_img(self.lr_image_filenames[0][index])
        input2 = load_img(self.lr_image_filenames[1][index])
        # target = ImageOps.equalize(target)
        # input_eq = ImageOps.equalize(input)
        # target = ImageOps.equalize(target)
        #         # input = ImageOps.equalize(input)
        # input1,input2, target1,target2 = get_patch(input1, input2,target1,target2,self.patch_size, self.upscale_factor)
        # augment:这个是加噪声和控制图像水平垂直反转的，这里不用了
        if self.data_augmentation:
            input1,input2, target1,target2, _ = augment(input1, input2,target1,target2)

        if self.transform:
            # print(np.shape(img_in))
            img_in1 = self.transform(input1)
            img_in2 = self.transform(input2)
            # print(np.shape(img_in))
            img_tar1 = self.transform(target1)
            img_tar2 = self.transform(target2)
        # if self.if_LAB:
        #     img_in = colors.rgb_to_lab(img_in)
        #     img_tar = colors.rgb_to_lab(img_tar)

        #合成亮通道。
        # img_in1=bright_channel(img_in1)
        # img_in2=bright_channel(img_in2)
        return img_in1, img_in2,img_tar1,img_tar2
    def __len__(self):
        # print("###########",len(self.hr_image_filenames[0]))
        if self.mode=='test':
            return len(self.hr_image_filenames[0])
        else:
            return len(self.hr_image_filenames[0])
def get_training_set(data_dir, upscale_factor, patch_size, data_augmentation,opt):
    lr_dir, hr_dir=all_list(opt)
    # 最后一百张留作测试
    return DatasetFromFolder(hr_dir, lr_dir, patch_size, upscale_factor, data_augmentation,
                              transform=transform(opt.patch_sizeH,opt.patch_sizeW))
def get_test_set(data_dir, upscale_factor, patch_size, data_augmentation,opt):
    lr_dir, hr_dir=test_list(opt)
    # 最后一百张留作测试
    return DatasetFromFolder(hr_dir, lr_dir, patch_size, upscale_factor, data_augmentation,
                              transform=transform(opt.patch_sizeH,opt.patch_sizeW),mode='test')
def log_metrics(logs, iter, end_str=" "):
    str_print = ''
    for key, value in logs.items():
        str_print = str_print + "%s: %.4f || " % (key, value)
    print(str_print, end=end_str)
#################################################保存模型参数
def checkpoint(model, epoch, opt):
    save_folder = os.path.join('/root/autodl-tmp/model/SDSD/',opt.modelname)   
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_out_path = os.path.join(save_folder,"{}_{}.pth".format(opt.modelname,epoch))  
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
###添加L1正则化
def L1_regularization(model,lamda=5e-5):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(abs(param))
    return lamda * regularization_loss
def get_test_resize_data(low_list,high_list):
    trans = transforms.ToTensor()
    resize_trans=transforms.Resize([512,960])
    size1=opt.patch_sizeH
    size2=opt.patch_sizeW
    channel_swap = (1, 2, 0)
    ll,hh=test_list(opt)
    device = torch.device('cuda:'+opt.deviceid)
    test_LL_list = ll
    test_NL_list = hh
    low_data=[]
    high_data=[]
    for i in range(test_NL_list.__len__()):
        LL = trans(Image.open(test_LL_list[i]).convert('RGB'))
        LL = resize_trans(LL)
        LL = LL.unsqueeze(0).to(device)
        low_data.append(LL)
    return low_data


#################################################训练时的模型评估
def eval(model, epoch, writer, txt_write, opt,low_list,high_list,low_data):

    print("==> Start testing")
    device = torch.device('cuda:'+opt.deviceid)
    tStart = time.time()
    trans = transforms.ToTensor()
    resize_trans=transforms.Resize([opt.patch_sizeH,opt.patch_sizeW])
    size1=opt.patch_sizeH
    size2=opt.patch_sizeW
    channel_swap = (1, 2, 0)
    model.eval()
    # Pay attention to the data structure
    # test_LL_folder = os.path.join( opt.testset,"low")
    # test_NL_folder = os.path.join( opt.testset,"high")
    #为增强的相邻帧设置存储位置
    test_est_folder1 = os.path.join(opt.output,opt.modelname,'eopch_%04d'% (epoch))
    try:
        os.stat(test_est_folder1)
    except:
        os.makedirs(test_est_folder1)
    ll,hh=test_list(opt)

    test_LL_folder = ll
    test_NL_folder = hh
    test_LL_list=test_LL_folder
    test_NL_list=test_NL_folder

    est_list = [join(test_est_folder1,x[-11:-9]+'_'+x[-7:]) for x in test_LL_folder if is_image_file(x)]
    for i in range(test_NL_list.__len__()):
        with torch.no_grad():
            if i%50==0:
                print(i)
            LL=low_data[i]
            prediction,_ = model(LL,LL,mode='test')
            prediction = prediction.data[0].cpu().numpy().transpose(channel_swap)
            prediction = prediction * 255.0
            prediction = prediction.clip(0, 255)
            Image.fromarray(np.uint8(prediction)).save(est_list[i])
    psnr_score = 0.0
    ssim_score = 0.0
    test_LL_list.sort()
    test_NL_list.sort()
    est_list.sort()
    print(test_LL_list.__len__(),test_NL_list.__len__())
    for i in range(test_LL_list.__len__()):
        if i%50==0:
            print(i)
        gt = Image.open(test_NL_list[i]).convert('RGB').resize((size1,size2))
        gt=np.array(gt)
        # gt=cv2.imread(test_NL_list[i])
        # gt=cv2.resize(gt,(size1,size2))
        est = Image.open(est_list[i]).convert('RGB').resize((size1,size2))
        est=np.array(est)
        # est=cv2.imread(est_list[i]).astype(np.uint8)
        # est=cv2.resize(est,(size1,size2))
        # gt = cv2.resize(gt, (size1, size2), interpolation=cv2.INTER_AREA)
        # gt = transforms.ToPILImage(gt)
        psnr_val = compare_psnr(gt, est, data_range=255)
        ssim_val = compare_ssim(gt, est, multichannel=True)
        psnr_score = psnr_score + psnr_val
        ssim_score = ssim_score + ssim_val
    psnr_score = psnr_score / (test_NL_list.__len__())
    ssim_score = ssim_score / (test_NL_list.__len__())  
    print(psnr_score,ssim_score,test_NL_list.__len__())  
    print("time: {:.2f} seconds ==> ".format(time.time() - tStart), end=" ")
    # print(len(listdir('./output/EDnet-vedio/512/')))
    return psnr_score, ssim_score
def logging(dircname):
    if not os.path.exists(dircname):
        os.makedirs(dircname)
    writer = SummaryWriter(dircname,  opt.modelname)
    return writer



######################################################保存增的图像
def tensor_to_img(tensor,path,j):
    if not os.path.exists(path):
        os.makedirs(path)
    
    toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    num=tensor.shape[0]
    for i in range(num):
        img =tensor[i]
        pic = toPIL(img)
        str_path=path+str(j).zfill(6)+str(i).zfill(2)+'.jpg'
        pic.save(str_path)
def test_to_img(tensor,est_dir,j):
    toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    num=tensor.shape[0]
    for i in range(num):
        img =tensor[i]
        pic = toPIL(img)
        str_path=est_dir[i+j]
        pic.save(str_path)
# def test(opt):
#     txt_name = 'metrics_'+opt.modelname+'.txt'
#     with open(txt_name, mode='w') as txt_write:
#         device = torch.device('cuda:'+opt.deviceid)
#         model = LLIENet(device).to(device)
#         model.load_state_dict(torch.load('./the_best_attation_9.pth'))
#         psnr_score, ssim_score = eval(model, epoch, writer, txt_write, opt)
#         print(psnr_score, ssim_score)
def main(opt):
    writer = logging(os.path.join('tensorboard',opt.modelname))
    txt_name = 'metrics_'+opt.modelname+'.txt'

    low_list,high_list=all_list(opt)
    test_lol,test_high=test_list(opt)
    
    low_data=get_test_resize_data(test_lol,test_high)
    
    
    
    with open(txt_name, mode='w') as txt_write:
        cuda = opt.gpu_mode
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        torch.manual_seed(opt.seed)
        if cuda:
            torch.cuda.manual_seed(opt.seed)
            cudnn.benchmark = True
        gpus_list = range(opt.gpus)

        # =============================#
        #   Prepare training data      #
        # =============================#
        print('===> Prepare training data')
        print('#### Now dataset is SDSD ####')
        train_set = get_training_set(opt.trainset, 1, opt.patch_size,True,opt=opt)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                          pin_memory=True, shuffle=True, drop_last=True)
        # =============================#
        #          Build model         #
        # =============================#
        print('===> Build model')
        
        device = torch.device('cuda:'+opt.deviceid)
        lighten = EDnet().to(device)
        
        # lighten.load_state_dict(torch.load('/root/autodl-tmp/model/models/base_model/base_model_572.pth'))
        # lighten = torch.load('./test_200.pth')
        # lighten.to(device) 
        print('---------- Networks architecture -------------')
        print_network(lighten)    
        print('----------------------------------------------')
        
        # =============================#
        #         Loss function        #
        # =============================#
        L1_criterion = nn.L1Loss()
        TV_loss = TVLoss()
        mse_loss = torch.nn.MSELoss()
        ssim = pytorch_ssim.SSIM()
        percep_loss = VGGPerceptualLoss()
        smooth_criterion = nn.SmoothL1Loss()

        loss_linzhen = linzhen_pool()
        loss_self =Self_loss()

        if cuda:
            gpus_list = range(opt.gpus)
            mse_loss = mse_loss.to(device)
            L1_criterion = L1_criterion.to(device)
            # TV_loss = TV_loss.to(device)
            ssim = ssim.to(device)
            percep_loss = percep_loss.to(device)
            smooth_criterion = smooth_criterion.to(device)
            

        # =============================#
        #         Optimizer            #
        # =============================#
        parameters = [lighten.parameters()]
        optimizer = optim.Adam(itertools.chain(*parameters), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

        # =============================#
        #         Training             #
        # =============================#
        int_num=1
        best_loss=9999999
        for epoch in range(opt.start_iter, opt.nEpochs + 1):
            print('===> training epoch %d' % epoch)
            epoch_loss = 0
            tStart_epoch = time.time()
            for iteration, batch in enumerate(training_data_loader):
                over_Iter = epoch * len(training_data_loader) + iteration 
                optimizer.zero_grad()

                LL_t1, LL_t2,NL_t1,NL_t2 = batch[0], batch[1],batch[2], batch[3]
                if cuda:
                    LL_t1, LL_t2 = LL_t1.to(device),LL_t2.to(device)
                    NL_t1,NL_t2 = NL_t1.to(device),NL_t2.to(device)
                int_num=int_num+1
                t0 = time.time()
                LL_t1_flatten = torch.flatten(LL_t1[:,:3,:,:])
                LL_t2_flatten = torch.flatten(LL_t2[:,:3,:,:])
                
                pred_t1,pred_t2 = lighten.forward(LL_t1,LL_t2,mode='train')
                pred_t_flatten1,pred_t_flatten2 = torch.flatten(pred_t1),torch.flatten(pred_t2)

                inner_loss1 = torch.dot(pred_t_flatten1, LL_t1_flatten) / (LL_t1.shape[0])/ (LL_t1.shape[1])/(LL_t1.shape[2])/(LL_t1.shape[3])
                inner_loss2 = torch.dot(pred_t_flatten2, LL_t2_flatten) / (LL_t2.shape[0])/ (LL_t2.shape[1])/(LL_t2.shape[2])/(LL_t2.shape[3])
                ssim_loss1 = 1 - ssim(pred_t1, NL_t1)
                ssim_loss2 = 1 - ssim(pred_t2, NL_t2)
                tv_loss1 = TV_loss(pred_t1)
                tv_loss2 = TV_loss(pred_t2)
                p_loss1 = percep_loss(pred_t1, NL_t1) 
                p_loss2 = percep_loss(pred_t2, NL_t2) 
                smoothloss1  = smooth_criterion(pred_t1, NL_t1)
                smoothloss2  = smooth_criterion(pred_t2, NL_t2)
                inner_loss=(inner_loss1+inner_loss2)/2
                p_loss=(p_loss1+p_loss2)/2
                ssim_loss=(ssim_loss1+ssim_loss2)/2
                smoothloss=1*(smoothloss1+smoothloss2)/2
                tv_loss=0.001*(tv_loss1+tv_loss2)/2
                L1_re=L1_regularization(lighten)

                linzhen_loss=1*loss_linzhen.forward(pred_t1,pred_t2,4)
                self_loss=1*loss_self.forward(NL_t1, NL_t2, pred_t1,pred_t2)

               
                loss = 1*ssim_loss +1*p_loss + 1*smoothloss +linzhen_loss+self_loss+tv_loss+inner_loss


                writer.add_scalar('ssim_loss', ssim_loss, over_Iter)
                writer.add_scalar('tv_loss', tv_loss*0.01, over_Iter)
                writer.add_scalar('perceptual_loss', p_loss, over_Iter)
                writer.add_scalar('smooth_loss', smoothloss, over_Iter)
                writer.add_scalar('inner_loss', inner_loss, over_Iter)
                writer.add_scalar('linzhen_loss', linzhen_loss, over_Iter)
                writer.add_scalar('self_loss', self_loss, over_Iter)
                writer.add_scalar('L1_re', L1_re, over_Iter)

                loss.backward()
                optimizer.step()
                t1 = time.time()
                if int_num % 300== 0:
                    tensor_to_img(pred_t1, opt.output_100, int_num)
                if best_loss > loss:
                    best_loss = loss
                    torch.save(lighten.state_dict(), './the_best_'+opt.modelname+'.pth')
                epoch_loss += loss

                if iteration % 5 == 0:
                    print("Epoch: %d/%d || Iter: %d/%d " % (epoch, opt.nEpochs, iteration, len(training_data_loader)),
                          end=" ==> ")
                    logs = {
                        "loss": loss.data,
                        "ssim_loss": ssim_loss.data,
                        # "tv_loss": tv_loss.data,
                        "percep_loss": p_loss.data,
                        "smooth_loss": smoothloss,
                        # "inner_loss": inner_loss,
                        "linzhen_loss": linzhen_loss.data,
                        "self_loss": self_loss.data,
                        "L1_re":L1_re
                    }
                    log_metrics( logs, over_Iter)
                    print("time: {:.4f} s".format(t1 - t0))

            writer.add_scalar('epoch_loss', float(epoch_loss/len(training_data_loader)), epoch)
            print("===> Epoch {} Complete: Avg. Loss: {:.4f}; ==> {:.2f} seconds".format(epoch, epoch_loss / len(
                training_data_loader), time.time() - tStart_epoch))
            if epoch % (opt.snapshots) == 0:
                file_checkpoint = checkpoint(lighten, epoch, opt)
                psnr_score, ssim_score = eval(lighten, epoch, writer, txt_write, opt,test_lol,test_high,low_data)
                
                logs = {
                    "psnr": psnr_score,
                    "ssim": ssim_score,
                }
                log_metrics(logs, epoch, end_str="\n")
                txt_write.writelines(["Epoch:",str(epoch)," psnr: ",str(psnr_score)," ssim: ",str(ssim_score)])
                txt_write.writelines(['\n'])
            if (epoch+1) %5 ==0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= opt.lr_decay
                print('G: Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
        print("======>>>Finished time: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
if __name__ =='__main__':
    opt=cfg()
    main(opt)
    # test(opt)