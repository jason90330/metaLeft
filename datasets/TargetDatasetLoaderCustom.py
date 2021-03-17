import os
import torch
import random
from imutils import paths
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
from PIL import Image
from misc import utils
from pdb import set_trace as st

def default_loader(path):
    RGBimg = Image.open(path).convert('RGB')
    HSVimg = Image.open(path).convert('HSV')
    RGBimg = RGBimg.resize((260,260))
    HSVimg = HSVimg.resize((260,260))
    return RGBimg, HSVimg


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
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # fn = os.path.join(self.root, fn)
        rgbimg, hsvimg = self.loader(fn)
        if self.transform is not None:
            rgbimg = self.transform(rgbimg)
            hsvimg = self.transform(hsvimg)

            catimg = torch.cat([rgbimg,hsvimg],0)

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
