import os
import os.path as osp
import argparse

import torch
from torch import nn
from tensorboardX import SummaryWriter
# from core.trainCustom import Train
from core.trainCustomEffAll import Train
# from core.trainCustomEffAllBalance import Train
# from core.trainCustomEffOptimize import Train
# from core import Test
from core.testCustom import Test
# from core.testArcface import Test
# from core import Train, Test
# from datasets.DatasetLoaderCustom import get_dataset_loader
# from datasets.DatasetLoaderBalance import get_dataset_loader
from datasets.DatasetLoaderRGB_DCT import get_dataset_loader
# from datasets.DatasetLoaderDCT import get_dataset_loader
from datasets.TargetDatasetLoaderCustom import get_tgtdataset_loader
# from datasets.DatasetLoader import get_dataset_loader
# from datasets.TargetDatasetLoader import get_tgtdataset_loader
from misc.utils import init_model, init_random_seed, mkdirs
from misc.saver import Saver
import models
import random
from pdb import set_trace as st

def main(args):

    if args.training_type is 'Train':
        # savefilename = osp.join(args.dataset1+args.dataset2+args.dataset3+'1')
        savefilename = osp.join(args.dataset1)
    elif  args.training_type is 'Test':    
        savefilename = osp.join(args.tstfile, args.tstdataset+'to'+args.tstdataset+args.snapshotnum) 

    summary_writer = SummaryWriter(osp.join(args.results_path, 'log'))# save current iter info.
    saver = Saver(args,savefilename)# print current iter info.
    saver.print_config()
    
    ##################### load seed#####################  
    args.seed = init_random_seed(args.manual_seed)

    #####################load datasets##################### 

    if args.training_type is 'Train':

        data_loader1_real = get_dataset_loader(name=args.dataset1, getreal=True, batch_size=args.batchsize)
        data_loader1_fake = get_dataset_loader(name=args.dataset1, getreal=False, batch_size=args.batchsize)

        data_loader2_real = get_dataset_loader(name=args.dataset2, getreal=True, batch_size=args.batchsize)
        data_loader2_fake = get_dataset_loader(name=args.dataset2, getreal=False, batch_size=args.batchsize)

        data_loader3_real = get_dataset_loader(name=args.dataset3, getreal=True, batch_size=args.batchsize)
        data_loader3_fake = get_dataset_loader(name=args.dataset3, getreal=False, batch_size=args.batchsize)

        # data_loader_target = get_tgtdataset_loader(name=args.dataset_target, batch_size=args.batchsize) 

    elif args.training_type is 'Test':

        data_loader_target = get_tgtdataset_loader(name=args.tstdataset, batch_size=args.test_batchsize) 


    ##################### load models##################### 

    FeatExtmodel = models.create(args.arch_FeatExt)  
    DepthEstmodel = models.create(args.arch_DepthEst)
    FeatEmbdmodel = models.create(args.arch_FeatEmbd,momentum=args.bn_momentum)


    if args.training_type is 'Train':
        FeatExt_restore = None
        DepthEst_restore = None
        FeatEmbd_restore = None

        FeatExtor = init_model(net=FeatExtmodel, init_type = args.init_type, restore=FeatExt_restore, parallel_reload=True)
        DepthEstor= init_model(net=DepthEstmodel, init_type = args.init_type, restore=DepthEst_restore, parallel_reload=True)
        FeatEmbder= init_model(net=FeatEmbdmodel, init_type = args.init_type, restore=FeatEmbd_restore, parallel_reload=False)

        print(">>> FeatExtor <<<")
        print(FeatExtor)
        print(">>> DepthEstor <<<")
        print(DepthEstor)
        print(">>> FeatEmbder <<<")
        print(FeatEmbder)    

        Train(args, FeatExtor, DepthEstor, FeatEmbder,
               data_loader1_real, data_loader1_fake,
               data_loader2_real, data_loader2_fake,
               data_loader3_real, data_loader3_fake,
            #    data_loader_target,
               summary_writer, saver) 

    elif args.training_type is 'Test':
        for modelIdx in range(1, args.test_model_num+1):
            # FeatExt_restore = osp.join(args.results_path, 'snapshots', args.dataset1, 'FeatExtor-'+str(modelIdx)+'.pt')
            # FeatEmbd_restore = osp.join(args.results_path, 'snapshots', args.dataset1, 'FeatEmbder-'+str(modelIdx)+'.pt')
            FeatExt_restore = osp.join(args.results_path, "model", 'FeatExtor-'+str(modelIdx)+'.pt')
            FeatEmbd_restore = osp.join(args.results_path, "model", 'FeatEmbder-'+str(modelIdx)+'.pt')
            # DepthEst_restore = None

            FeatExtor = init_model(net=FeatExtmodel, init_type = args.init_type, restore=FeatExt_restore, parallel_reload=True)
            FeatEmbder= init_model(net=FeatEmbdmodel, init_type = args.init_type, restore=FeatEmbd_restore, parallel_reload=False)
            # DepthEstor= init_model(net=DepthEstmodel, init_type = args.init_type, restore=DepthEst_restore, parallel_reload=True)

            Test(args, FeatExtor, FeatEmbder, data_loader_target, modelIdx)

    else:
        raise NotImplementedError('method type [%s] is not implemented' % args.training_type)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Meta_FA")

    # datasets 
        # OMI
    # parser.add_argument('--dataset1', type=str, default='OULU')
    # parser.add_argument('--dataset1', type=str, default='CelebA')
    # parser.add_argument('--dataset2', type=str, default='CelebA')
    # parser.add_argument('--dataset3', type=str, default='Siw-m')
    # parser.add_argument('--dataset_target', type=str, default='Siw-m')
    # parser.add_argument('--dataset_target', type=str, default='CelebA')

    # OIC
    parser.add_argument('--dataset1', type=str, default='OULU')
    parser.add_argument('--dataset2', type=str, default='idiap')
    parser.add_argument('--dataset3', type=str, default='CASIA')
    parser.add_argument('--dataset_target', type=str, default='MSU')
    
    #ICM    
    # parser.add_argument('--dataset1', type=str, default='idiap')
    # parser.add_argument('--dataset2', type=str, default='CASIA')
    # parser.add_argument('--dataset3', type=str, default='MSU')
    # parser.add_argument('--dataset_target', type=str, default='OULU')

    #OCM
    # parser.add_argument('--dataset1', type=str, default='OULU')
    # parser.add_argument('--dataset2', type=str, default='CASIA')
    # parser.add_argument('--dataset3', type=str, default='MSU')
    # parser.add_argument('--dataset_target', type=str, default='idiap')     
   

    # model
    # parser.add_argument('--arch_FeatExt', type=str, default='FeatExtractor')
    # parser.add_argument('--arch_DepthEst', type=str, default='DepthEstmator')
    # parser.add_argument('--arch_FeatEmbd', type=str, default='FeatEmbedder')
    parser.add_argument('--arch_FeatExt', type=str, default='Eff_FeatExtractor')
    parser.add_argument('--arch_DepthEst', type=str, default='Eff_DepthEstmator')
    parser.add_argument('--arch_FeatEmbd', type=str, default='Eff_FeatEmbedder')
    parser.add_argument('--eff_name', type=str, default='efficientnet-b0')
    parser.add_argument('--head_name', type=str, default='ArcFace')
    parser.add_argument('--focalWithWeight', type=bool, default=False)
    
    parser.add_argument('--init_type', type=str, default='xavier')
    parser.add_argument('--metatrainsize', type=int, default=2)
    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_lr', type=float, default= 0.0)
    parser.add_argument('--warm_iter', type=int, default=2000)
    # parser.add_argument('--lr_dep', type=float, default=1e-3)
    parser.add_argument('--meta_step_size', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--bn_momentum', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--optimizer', type=str, default='adam')

    # # # # training configs
    parser.add_argument('--training_type', type=str, default='Train')
    # parser.add_argument('--training_type', type=str, default='Test')
    # parser.add_argument('--results_path', type=str, default='./results/Train_CelebA_ori')
    # parser.add_argument('--results_path', type=str, default='./results/Train_CelebA_efficient_b0')
    # parser.add_argument('--results_path', type=str, default='./results/CelebA_efficient_b4_arc_focal')
    # parser.add_argument('--results_path', type=str, default='./results/CelebA_efficient_b4_arc_focal_norm')
    # parser.add_argument('--results_path', type=str, default='./results/CelebA_efficient_b4_optimize')
    # parser.add_argument('--results_path', type=str, default='./results/CelebA_efficient_b0_dct')
    # parser.add_argument('--results_path', type=str, default='./results/OULU_efficient_b0_dct')
    # parser.add_argument('--results_path', type=str, default='./results/OIC_efficient_b0_dct')
    # parser.add_argument('--results_path', type=str, default='./results/OIC_efficient_b0_woFocalW_equal')#worse than not equal
    parser.add_argument('--results_path', type=str, default='./results/OIC_efficient_b0_woFocalW_RGB_DCT')
    # parser.add_argument('--results_path', type=str, default='./results/Train_CelebA')
    # parser.add_argument('--results_path', type=str, default='./results/Train_CelebA_focal')
    # parser.add_argument('--results_path', type=str, default='./results/Train_CelebA_lambda')
    parser.add_argument('--batchsize', type=int, default=13)

    # parser.add_argument('--results_path', type=str, default='./results/Test_20191125/')
    # parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--tstfile', type=str, default='Train_CelebA_ori')
    # parser.add_argument('--tstdataset', type=str, default='Siw-m')
    # parser.add_argument('--tstdataset', type=str, default='OULU')
    parser.add_argument('--tstdataset', type=str, default='MSU')
    parser.add_argument('--tst_txt_name', type=str, default='testScore.txt')
    parser.add_argument('--snapshotnum', type=str, default='1')
    parser.add_argument('--test_batchsize', type=int, default=180)
    parser.add_argument('--test_model_num', type=int, default=1)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--tst_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--model_save_epoch', type=int, default=1)
    parser.add_argument('--manual_seed', type=int, default=666)

    parser.add_argument('--W_depth', type=int, default=10)
    parser.add_argument('--W_metatest', type=int, default=1)

    print(parser.parse_args())
    main(parser.parse_args())