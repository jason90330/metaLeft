import os
import os.path as osp
import argparse

import torch
from torch import nn
from tensorboardX import SummaryWriter
from core.trainCustom import Train
from core import Test
# from core import Train, Test
from datasets.DatasetLoaderCustom import get_dataset_loader
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
        savefilename = osp.join(args.dataset1+"_"+args.dataset2)
    elif  args.training_type is 'Test':    
        savefilename = osp.join(args.dataset1+"_"+args.dataset2+"_to_"+args.dataset_target+"_"+args.snapshotnum) 

    args.seed = init_random_seed(args.manual_seed)

    if args.training_type in ['Train','Test']:
        summary_writer = SummaryWriter(osp.join(args.results_path, 'log', savefilename))
        saver = Saver(args,savefilename)
        saver.print_config()

    ##################### load seed#####################  

    #####################load datasets##################### 

    if args.training_type is 'Train':

        data_loader1_real = get_dataset_loader(name=args.dataset1, getreal=True, batch_size=args.batchsize)
        data_loader1_fake = get_dataset_loader(name=args.dataset1, getreal=False, batch_size=args.batchsize)

        data_loader2_real = get_dataset_loader(name=args.dataset2, getreal=True, batch_size=args.batchsize)
        data_loader2_fake = get_dataset_loader(name=args.dataset2, getreal=False, batch_size=args.batchsize)

        # data_loader3_real = get_dataset_loader(name=args.dataset3, getreal=True, batch_size=args.batchsize)
        # data_loader3_fake = get_dataset_loader(name=args.dataset3, getreal=False, batch_size=args.batchsize)

        # data_loader_target = get_tgtdataset_loader(name=args.dataset_target, batch_size=args.batchsize) 

    elif args.training_type is 'Test':

        data_loader_target = get_tgtdataset_loader(name=args.dataset_target, batch_size=args.test_batchsize) 


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
            #    data_loader3_real, data_loader3_fake,
            #    data_loader_target,
               summary_writer, saver, savefilename) 

    elif args.training_type is 'Test':
        for modelIdx in range(1, args.test_model_num+1):
            FeatExt_restore = osp.join(args.results_path, 'snapshots', args.dataset1+"_"+args.dataset2, 'FeatExtor-'+str(modelIdx)+'.pt')
            FeatEmbd_restore = osp.join(args.results_path, 'snapshots', args.dataset1+"_"+args.dataset2, 'FeatEmbder-'+str(modelIdx)+'.pt')
            DepthEst_restore = None

            FeatExtor = init_model(net=FeatExtmodel, init_type = args.init_type, restore=FeatExt_restore, parallel_reload=True)
            FeatEmbder= init_model(net=FeatEmbdmodel, init_type = args.init_type, restore=FeatEmbd_restore, parallel_reload=False)
            DepthEstor= init_model(net=DepthEstmodel, init_type = args.init_type, restore=DepthEst_restore, parallel_reload=True)

            Test(args, FeatExtor, FeatEmbder, data_loader_target, modelIdx)

    else:
        raise NotImplementedError('method type [%s] is not implemented' % args.training_type)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Meta_FA")

    # datasets 
        # OMI
    parser.add_argument('--dataset1', type=str, default='CelebA')
    parser.add_argument('--dataset2', type=str, default='MSU')
    # parser.add_argument('--dataset3', type=str, default='Siw-m')
    parser.add_argument('--dataset_target', type=str, default='Siw-m')

        # OIC
    # parser.add_argument('--dataset1', type=str, default='OULU')
    # parser.add_argument('--dataset2', type=str, default='idiap')
    # parser.add_argument('--dataset3', type=str, default='CASIA')
    # parser.add_argument('--dataset_target', type=str, default='MSU')
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
    parser.add_argument('--arch_FeatExt', type=str, default='FeatExtractor')
    parser.add_argument('--arch_DepthEst', type=str, default='DepthEstmator')
    parser.add_argument('--arch_FeatEmbd', type=str, default='FeatEmbedder')
    
    parser.add_argument('--init_type', type=str, default='xavier')   
    parser.add_argument('--metatrainsize', type=int, default=1)   
    # optimizer
    parser.add_argument('--lr_meta', type=float, default=1e-3)
    parser.add_argument('--lr_dep', type=float, default=1e-3)
    parser.add_argument('--meta_step_size', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)    
    parser.add_argument('--bn_momentum', type=float, default=1)    
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--optimizer_meta', type=str, default='adam')

    # # # # training configs
    # parser.add_argument('--training_type', type=str, default='Train')
    parser.add_argument('--results_path', type=str, default='./results/Train_Lambda_CelebA_MSU_all_resAtt')
    # parser.add_argument('--results_path', type=str, default='./results/Train_Lambda_CelebA_MSU_no_equal_fc1')
    # parser.add_argument('--results_path', type=str, default='./results/Train_Lambda_CelebA_MSU_no_equal')#half resnet half ori paper
    # parser.add_argument('--results_path', type=str, default='./results/Train_CelebA_MSU')
    # parser.add_argument('--results_path', type=str, default='./results/Train_20210314')
    parser.add_argument('--batchsize', type=int, default=7)

    parser.add_argument('--training_type', type=str, default='Test')
    # parser.add_argument('--results_path', type=str, default='./results/Test_20191125/')
    # parser.add_argument('--batchsize', type=int, default=1)
    # parser.add_argument('--tstfile', type=str, default='Train_CelebA_MSU')
    parser.add_argument('--tstdataset', type=str, default='siw-m')    
    parser.add_argument('--snapshotnum', type=str, default='1')
    parser.add_argument('--test_batchsize', type=int, default=100)
    parser.add_argument('--test_model_num', type=int, default=1)
 

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--tst_step', type=int, default=200)
    parser.add_argument('--model_save_step', type=int, default=500)
    parser.add_argument('--model_save_epoch', type=int, default=1)
    parser.add_argument('--manual_seed', type=int, default=None)

    parser.add_argument('--W_depth', type=int, default=10)
    parser.add_argument('--W_metatest', type=int, default=1)


    print(parser.parse_args())
    main(parser.parse_args())

