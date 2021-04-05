import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
from imutils import paths
from PIL import Image, ImageOps
from misc import utils
from pdb import set_trace as st

def OriImg_loader(path):
    RGBimg = Image.open(path).convert('RGB')
    HSVimg = Image.open(path).convert('HSV')
    # RGBimg = RGBimg.resize((380,380))
    # HSVimg = HSVimg.resize((380,380))
    RGBimg = RGBimg.resize((256,256))
    HSVimg = HSVimg.resize((256,256))
    return RGBimg, HSVimg

def OriImg_b4_loader(path):
    RGBimg = Image.open(path).convert('RGB')
    HSVimg = Image.open(path).convert('HSV')
    # RGBimg = RGBimg.resize((380,380))
    # HSVimg = HSVimg.resize((380,380))
    RGBimg = RGBimg.resize((300,300))
    HSVimg = HSVimg.resize((300,300))
    return RGBimg, HSVimg

def DepthImg_loader(path,imgsize=128):
    img = Image.open(path)
    re_img = img.resize((imgsize, imgsize), resample=Image.BICUBIC)
    re_img = ImageOps.grayscale(re_img)
    return re_img

def DepthImg_b4_loader(path,imgsize=150):
    img = Image.open(path)
    re_img = img.resize((imgsize, imgsize), resample=Image.BICUBIC)
    re_img = ImageOps.grayscale(re_img)
    return re_img    


class DatasetLoader(Dataset):
    def __init__(self, name, getreal, transform=None, oriimg_loader=OriImg_loader, depthimg_loader=DepthImg_loader, root='../../'):

        self.name = name
        self.root = os.path.expanduser(root)
        imgPaths = []
        imgs = []
        if name == 'CelebA':
            imgPaths = list(paths.list_images(self.root+"CelebA_Data/trainSquareCropped"))
            for path in imgPaths:                
                if "live" in path:# and getreal:
                    label = 0
                    depth_dir = path.replace("trainSquareCropped", "trainSquareCropped_depth")
                    if os.path.exists(depth_dir):
                        imgs.append((path, depth_dir, label))

                else:#if not getreal:
                    label = 1
                    depth_dir = path.replace("trainSquareCropped", "trainSquareCropped_depth")
                    if os.path.exists(depth_dir):
                        imgs.append((path, depth_dir, label))
        
        elif name == 'MSU':
            imgPaths = list(paths.list_images(self.root+"MSU_MFSD_similar/train"))
            for path in imgPaths:                
                if "live" in path:# and getreal:
                    label = 0
                    depth_dir = path.replace("MSU_MFSD_similar", "MSU_MFSD_similar_depth")
                    if os.path.exists(depth_dir):
                        imgs.append((path, depth_dir, label))
                else:#if not getreal:
                    label = 1
                    depth_dir = path.replace("MSU_MFSD_similar", "MSU_MFSD_similar_depth")
                    if os.path.exists(depth_dir):
                        imgs.append((path, depth_dir, label))
        
        elif name == 'OULU':
            imgPaths = list(paths.list_images(self.root+"Oulu_similar/Train_files"))
            for path in imgPaths:    
                pathTok = path.split("/")[-2][-1:]            
                if pathTok == "1":# and getreal:
                    label = 0
                    depth_dir = path.replace("Oulu_similar", "Oulu_similar_depth")
                    if os.path.exists(depth_dir):
                        imgs.append((path, depth_dir, label))
                else:#if not getreal:
                    label = 1
                    depth_dir = path.replace("Oulu_similar", "Oulu_similar_depth")
                    if os.path.exists(depth_dir):
                        imgs.append((path, depth_dir, label))

        elif name == 'idiap':
            imgPaths = list(paths.list_images(self.root+"Idiap_similar/train"))
            for path in imgPaths:    
                if "real" in path:
                    label = 0
                    depth_dir = path.replace("Idiap_similar", "Idiap_similar_depth")
                    if os.path.exists(depth_dir):
                        imgs.append((path, depth_dir, label))
                else:#if not getreal:
                    label = 1
                    depth_dir = path.replace("Idiap_similar", "Idiap_similar_depth")
                    if os.path.exists(depth_dir):
                        imgs.append((path, depth_dir, label))
        
        elif name == 'CASIA':
            imgPaths = list(paths.list_images(self.root+"CASIA-MFSD_similar/train_release"))
            for path in imgPaths:    
                pathTok = path.split('/')[-2]
                if pathTok == "1" or pathTok == "2" or pathTok == "HR_1":
                    label = 0
                    depth_dir = path.replace("CASIA-MFSD_similar", "CASIA-MFSD_similar_depth")
                    if os.path.exists(depth_dir):
                        imgs.append((path, depth_dir, label))
                else:#if not getreal:
                    label = 1
                    depth_dir = path.replace("CASIA-MFSD_similar", "CASIA-MFSD_similar_depth")
                    if os.path.exists(depth_dir):
                        imgs.append((path, depth_dir, label))
        '''
        elif name == 'Siw-m':
            imgPaths = list(paths.list_images("Siw-m_similar_er/train"))
            for path in imgPaths:                
                if "Live" in path:
                    label = 0
                else:
                    label = 1
                depth_dir = path.replace("Siw-m_similar_er", "Siw-m_similar_er_depth")
                if os.path.exists(depth_dir):
                    imgs.append((path, depth_dir, label))
        '''
        
        self.imgs = imgs
        self.transform = transform
        self.oriimg_loader = oriimg_loader
        self.depthimg_loader = depthimg_loader
        # self.depth_loader = depth_loader


    def __getitem__(self, index):
        ori_img_dir, depth_img_dir, label = self.imgs[index]
        # ori_img_dir_all = os.path.join(ori_img_dir)
        # depth_img_dir_all = os.path.join(depth_img_dir)

        ori_rgbimg, ori_hsvimg = self.oriimg_loader(ori_img_dir)
        depth_img = self.depthimg_loader(depth_img_dir)

        if self.transform is not None:
            ori_rgbimg = self.transform(ori_rgbimg)
            ori_hsvimg = self.transform(ori_hsvimg)
            depth_img = self.transform(depth_img)

            ori_catimg = torch.cat([ori_rgbimg,ori_hsvimg],0)
        return ori_catimg, depth_img, label

    def __len__(self):
        return len(self.imgs)


def get_dataset_loader(name, getreal, batch_size):

    # pre_process = transforms.Compose([transforms.ToTensor(),
    #                                   transforms.Normalize(
    #                                   mean=[0.485, 0.456, 0.406],
    #                                   std=[0.229, 0.224, 0.225])])      

    pre_process = transforms.Compose([transforms.ToTensor()])
  

    # dataset and data loader
    dataset = DatasetLoader(name=name,
                        getreal=getreal,
                        transform=pre_process
                        )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    return data_loader
