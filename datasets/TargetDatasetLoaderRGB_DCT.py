import os
import dct
import torch
import random
from imutils import paths
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
from PIL import Image
from misc import utils
from pdb import set_trace as st

def freq_domain(input):
    input = transforms.ToTensor()(input)#.unsqueeze_(0)    
    # input = pad_to(input, 8)
    input = dct.to_ycbcr(input * 255)
    input -= 128
    rst = dct.batch_dct(input)[0]
    rst =rst[0:3, :, :]
    return rst

def default_loader(path):
    RGBimg = Image.open(path).convert('RGB')
    # HSVimg = Image.open(path).convert('HSV')
    # RGBimg = RGBimg.resize((380,380))
    # HSVimg = HSVimg.resize((380,380))
    RGBimg = RGBimg.resize((256,256))
    DCTimg = freq_domain(RGBimg)
    # HSVimg = HSVimg.resize((256,256))
    return RGBimg, DCTimg

class DatasetLoader(Dataset):
    def __init__(self, name, transform=None, loader=default_loader, root='../../'):

        self.name = name
        self.root = os.path.expanduser(root)
        imgs = []
                
        if name == 'Siw-m':
            txt_path = "datasets/siw_metas/test_list.txt"
            imgPaths = list(paths.list_images(self.root+"Siw-m_similar_er"))
            random.Random(4).shuffle(imgPaths)
            with open(txt_path) as input_file:    
                foldLists = input_file.readlines()
                # img_paths = list(paths.list_images(cfg.IMG_PATH))
                # datas = [x.strip() for x in open(cfg.TEST_TXT_PATH)]
                lenOfDatas=len(imgPaths)
                for idx, img_path in enumerate(imgPaths):   
                    exist = False     
                    for line in foldLists:
                        testFolder = line.strip("\n")
                        if "Live" in img_path and "Test" in img_path:
                            label=0
                            exist = True
                            break
                        elif testFolder in img_path:
                            label=1
                            exist = True
                            break
                        else:
                            continue
                    # depth_dir = img_path.replace("Siw-m_similar_er", "Siw-m_similar_er_depth")
                    # if os.path.exists(depth_dir):
                    #     imgs.append((img_path, depth_dir, label))
                    if exist:
                        imgs.append((img_path, label))

        elif name == 'CelebA':
            imgPaths = list(paths.list_images(self.root+"CelebA_Data/testSquareCropped"))
            for path in imgPaths:
                if "live" in path:# and getreal:
                    label = 0
                else:#if not getreal:
                    label = 1
                imgs.append((path, label))
        
        elif name == 'MSU':
            imgPaths = list(paths.list_images(self.root+"MSU_MFSD_similar/test"))
            for path in imgPaths:
                if "live" in path:# and getreal:
                    label = 0
                else:#if not getreal:
                    label = 1
                imgs.append((path, label))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # fn = os.path.join(self.root, fn)
        rgbimg, dctimg = self.loader(fn)
        if self.transform is not None:
            rgbimg = self.transform(rgbimg)
            catimg = torch.cat([rgbimg,dctimg],0)

        return catimg, label

    def __len__(self):
        return len(self.imgs)


def get_tgtdataset_loader(name, batch_size):

    # pre_process = transforms.Compose([transforms.ToTensor(),
    #                                   transforms.Normalize(
    #                                   mean=[0.485, 0.456, 0.406],
    #                                   std=[0.229, 0.224, 0.225])])      

    pre_process = transforms.Compose([transforms.ToTensor()])  


    # dataset and data loader
    dataset = DatasetLoader(name=name,
                        transform=pre_process
                        )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True)

    return data_loader
