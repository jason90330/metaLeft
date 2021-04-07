import itertools
import os
from collections import OrderedDict
import torchvision.utils as vutils
import torch
import torch.optim as optim
from torch import nn
from misc.utils import get_inf_iterator, mkdir
from misc import evaluate
from torch.nn import DataParallel
from loss.focal import FocalLoss
from loss.metrics import ArcFace, CosFace, SphereFace, Am_softmax
import random
import torch.autograd as autograd
from copy import deepcopy
from itertools import permutations, combinations
import models

from pdb import set_trace as st

def Train(args, FeatExtor_rgb, FeatExtor_dct, DepthEstor_rgb, DepthEstor_dct, ensembelModel,
        data_loader1_real, data_loader1_fake,
        data_loader2_real, data_loader2_fake,
        data_loader3_real, data_loader3_fake,
        # data_loader_target,
        summary_writer, Saver):
            
    ####################
    # 1. setup network #
    ####################
    # set train state for Dropout and BN layers
    FeatExtor_rgb.train()
    FeatExtor_dct.train()
    DepthEstor_rgb.train()
    DepthEstor_dct.train()
    ensembelModel.train()
    
    '''
    head_dict = {'ArcFace': ArcFace(in_features = 1000, out_features = 2, device_id = [0]),
                'CosFace': CosFace(in_features = 1000, out_features = 2, device_id = [0]),
                'SphereFace': SphereFace(in_features = 1000, out_features = 2, device_id = [0]),
                'Am_softmax': Am_softmax(in_features = 1000, out_features = 2, device_id = [0])}
    Head = head_dict[args.head_name]
    Head.train()
    '''


    FeatExtor_rgb = DataParallel(FeatExtor_rgb)    
    FeatExtor_dct = DataParallel(FeatExtor_dct)    
    DepthEstor_rgb = DataParallel(DepthEstor_rgb)    
    DepthEstor_dct = DataParallel(DepthEstor_dct)
    ensembelModel = DataParallel(ensembelModel)
 


    # setup criterion and optimizer
    criterionCls = FocalLoss(outNum = 2, withWeight = args.focalWithWeight)
    # criterionCls = nn.BCEWithLogitsLoss()
    criterionDepth = torch.nn.MSELoss()


    if args.optimizer is 'adam':
        optimizer_embd = optim.AdamW(itertools.chain(ensembelModel.parameters()),
                                   lr=args.lr,
                                   betas=(args.beta1, args.beta2),
                                   weight_decay=args.weight_decay)

        optimizer_extr = optim.AdamW(itertools.chain(FeatExtor_rgb.parameters(), FeatExtor_dct.parameters()),
                                   lr=args.lr,
                                   betas=(args.beta1, args.beta2),
                                   weight_decay=args.weight_decay)

        optimizer_dep = optim.AdamW(itertools.chain(DepthEstor_rgb.parameters(), DepthEstor_dct.parameters()),
                                   lr=args.lr,
                                   betas=(args.beta1, args.beta2),
                                   weight_decay=args.weight_decay)
        # optimizer_all = optim.Adam(itertools.chain(FeatExtor.parameters(), DepthEstor.parameters(), FeatEmbder.parameters()),
        #                            lr=args.lr_meta,
        #                            betas=(args.beta1, args.beta2))

    else:
        raise NotImplementedError('Not a suitable optimizer')
    

    # iternum = max(len(data_loader1_real),len(data_loader1_fake),
    #               len(data_loader2_real),len(data_loader2_fake), 
    #               len(data_loader3_real),len(data_loader3_fake))       
    # iternum = int(max(len(data_loader1_real),len(data_loader1_fake))/3)
    iternum = max(len(data_loader1_real), len(data_loader2_real), len(data_loader3_real))

    print('iternum={}'.format(iternum))

    ####################
    # 2. train network #
    ####################
    global_step = 0

    for epoch in range(args.epochs):

        data1 = get_inf_iterator(data_loader1_real)
        data2 = get_inf_iterator(data_loader2_real)
        data3 = get_inf_iterator(data_loader3_real)
        # data1_real = get_inf_iterator(data_loader1_real)
        # data1_fake = get_inf_iterator(data_loader1_fake)

        # data2_real = get_inf_iterator(data_loader2_real)
        # data2_fake = get_inf_iterator(data_loader2_fake)

        # data3_real = get_inf_iterator(data_loader3_real)
        # data3_fake = get_inf_iterator(data_loader3_fake)
            

        for step in range(iternum):
            if global_step<=args.warm_iter+1:
                optimizer_embd.param_groups[0]['lr'] = args.warmup_lr + global_step * (args.lr-args.warmup_lr)/args.warm_iter
                optimizer_extr.param_groups[0]['lr'] = args.warmup_lr + global_step * (args.lr-args.warmup_lr)/args.warm_iter
                optimizer_dep.param_groups[0]['lr'] = args.warmup_lr + global_step * (args.lr-args.warmup_lr)/args.warm_iter

            #============ one batch extraction ============#

            # cat_img1, depth_img1, lab1 = next(data1)

            # cat_img2, depth_img2, lab2 = next(data2)

            # cat_img3, depth_img3, lab3 = next(data3)

            rgb_img1, dct_img1, depth_img1, lab1 = next(data1)

            rgb_img2, dct_img2, depth_img2, lab2 = next(data2)

            rgb_img3, dct_img3, depth_img3, lab3 = next(data3)

            #============ one batch collection ============# 

            # cat_img1 = cat_img1.cuda()
            # depth_img1 = depth_img1.cuda()
            # lab1 = lab1.cuda()

            # cat_img2 = cat_img2.cuda()
            # depth_img2 = depth_img2.cuda()
            # lab2 = lab2.cuda()

            # cat_img3 = cat_img3.cuda()
            # depth_img3 = depth_img3.cuda()
            # lab3 = lab3.cuda()

            rgb_img1 = rgb_img1.cuda()
            dct_img1 = dct_img1.cuda()
            depth_img1 = depth_img1.cuda()
            lab1 = lab1.cuda()

            rgb_img2 = rgb_img2.cuda()
            dct_img2 = dct_img2.cuda()
            depth_img2 = depth_img2.cuda()
            lab2 = lab2.cuda()

            rgb_img3 = rgb_img3.cuda()
            dct_img3 = dct_img3.cuda()
            depth_img3 = depth_img3.cuda()
            lab3 = lab3.cuda()

           #============ doamin list augmentation ============# 
            catimglist = [[rgb_img1,dct_img1],[rgb_img2,dct_img2],[rgb_img3,dct_img3]]
            # catimglist_rgb = [rgb_img1,rgb_img2,rgb_img3]
            # catimglist_dct = [dct_img1,dct_img2,dct_img3]
            lablist = [lab1,lab2,lab3]
            deplist = [depth_img1,depth_img2,depth_img3]            

            domain_list = list(range(len(catimglist)))
            random.shuffle(domain_list) 
            
            meta_train_list = domain_list[:args.metatrainsize] 
            meta_test_list = domain_list[args.metatrainsize:]
            # print('metatrn={}, metatst={}'.format(meta_train_list, meta_test_list[0]))
 
            
            #============ meta training ============#

            Loss_dep_train = 0.0
            Loss_cls_train = 0.0

            adapted_state_dicts = []

            for index in meta_train_list:

                catimg_meta = catimglist[index]
                lab_meta = lablist[index]
                depGT_meta = deplist[index]

                catimg_meta_rgb, catimg_meta_dct = catimg_meta[0], catimg_meta[1]
                batchidx = list(range(len(catimg_meta_rgb)))
                random.shuffle(batchidx)
                img_rand_rgb = catimg_meta_rgb[batchidx,:]
                img_rand_dct = catimg_meta_dct[batchidx,:]
                lab_rand = lab_meta[batchidx]
                depGT_rand = depGT_meta[batchidx,:]

                feat_rgb = FeatExtor_rgb(img_rand_rgb)
                feat_dct = FeatExtor_dct(img_rand_dct)

                depth_rgb = DepthEstor_rgb(feat_rgb)
                depth_dct = DepthEstor_dct(feat_dct)
                Loss_dep = criterionDepth(depth_rgb, depGT_rand) + criterionDepth(depth_dct, depGT_rand)

                output = ensembelModel(feat_rgb, feat_dct)
                Loss_cls = criterionCls(output, lab_rand)
                Loss_cls_train+=Loss_cls
                Loss_dep_train+=Loss_dep
                
            # just update the training phase of (embedder)
            '''
            for name, param in FeatExtor.named_parameters():
                param.requires_grad = False
                print("\t", name)
            '''
            optimizer_extr.zero_grad()
            optimizer_embd.zero_grad()
            optimizer_dep.zero_grad()
            # optimizer_all.zero_grad()
            Loss_cls_train.backward(retain_graph=True)
            optimizer_embd.step()
            optimizer_embd.zero_grad()
            # optimizer_all.step()
            # optimizer_all.zero_grad()
            '''
            for name, param in FeatExtor.named_parameters():
                param.requires_grad = True
                print("\t", name)
            '''
            '''
            zero_param_grad(FeatEmbder.parameters())    
            grads_FeatEmbder = torch.autograd.grad(Loss_cls, FeatEmbder.parameters(), create_graph=True)
            fast_weights_FeatEmbder = FeatEmbder.cloned_state_dict()
    
            adapted_params = OrderedDict()
            for (key, val), grad in zip(FeatEmbder.named_parameters(), grads_FeatEmbder):
                adapted_params[key] = val - args.meta_step_size * grad
                fast_weights_FeatEmbder[key] = adapted_params[key]   

            adapted_state_dicts.append(fast_weights_FeatEmbder)
            '''
            #============ meta testing ============#    
            Loss_dep_test = 0.0
            Loss_cls_test = 0.0

            index = meta_test_list[0]

            catimg_meta = catimglist[index]
            catimg_meta_rgb, catimg_meta_dct = catimg_meta[0], catimg_meta[1]
            lab_meta = lablist[index]
            depGT_meta = deplist[index]

            batchidx = list(range(len(catimg_meta_rgb)))
            random.shuffle(batchidx)
            
            img_rand_rgb = catimg_meta_rgb[batchidx,:]
            img_rand_dct = catimg_meta_dct[batchidx,:]
            lab_rand = lab_meta[batchidx]
            depGT_rand = depGT_meta[batchidx,:]

            feat_rgb = FeatExtor_rgb(img_rand_rgb)
            feat_dct = FeatExtor_dct(img_rand_dct)
            depth_rgb = DepthEstor_rgb(feat_rgb)
            depth_dct = DepthEstor_dct(feat_dct)
            Loss_dep = criterionDepth(depth_rgb, depGT_rand) + criterionDepth(depth_dct, depGT_rand)
            
            output = ensembelModel(feat_rgb, feat_dct)
            Loss_cls_test = criterionCls(output, lab_rand)

            # feat = FeatExtor(img_rand) # use the old weight
            # depth_Pre = DepthEstor(feat) # use the old weight
            # Loss_dep = criterionDepth(depth_Pre, depGT_rand)

            # use the updated embedder to caculate
            '''
            pred = FeatEmbder(feat)
            Loss_cls_test = criterionCls(pred.squeeze(), lab_rand)
            '''
            #=arc
            # embd = FeatEmbder(feat)
            # thetas = Head(embd, lab_rand)
            # Loss_cls_test = criterionCls(thetas, lab_rand)
            # update training phase of (extractor)
            
            '''
            for name, param in FeatEmbder.named_parameters():
                param.requires_grad = False
                print("\t", name)
            
            with torch.autograd.set_detect_anomaly(True):
                Loss_cls_train.backward()
            '''
            '''=debug
            optimizer_extr.step()
            optimizer_extr.zero_grad()
            '''
            # optimizer_all.zero_grad()
            # need to reopen the require_grad
            '''
            for name, param in FeatEmbder.named_parameters():
                param.requires_grad = True
                print("\t", name)
            '''

            ''' do not need to forward pass the feat one by one
            for n_scr in range(len(meta_train_list)):# will use the new embeder weight
                a_dict = adapted_state_dicts[n_scr]

                pred = FeatEmbder(feat, a_dict)
                Loss_cls = criterionCls(pred.squeeze(), lab_rand)

                Loss_cls_test+=Loss_cls
            '''
            Loss_dep_test = Loss_dep

            Loss_dep_train_ave = Loss_dep_train/len(meta_train_list)   
            Loss_dep_test = Loss_dep_test  

            # Loss_meta_train =  Loss_cls_train+args.W_depth*Loss_dep_train  
            '''
            Loss_meta_train =  args.W_depth*Loss_dep_train  
            Loss_meta_test =  Loss_cls_test+args.W_depth*Loss_dep_test                 

            Loss_all = Loss_meta_train + args.W_metatest*Loss_meta_test
            Loss_all = Loss_meta_train + args.W_metatest*Loss_meta_test
            Loss_all.backward()
            '''
            Loss_meta_train =  args.W_depth*Loss_dep_train  
            Loss_meta_train.backward()

            Loss_meta_test =  args.W_metatest * Loss_cls_test+args.W_depth * Loss_dep_test                 
            Loss_meta_test.backward()
            # optimizer_all.zero_grad()
            optimizer_extr.step()
            optimizer_embd.step()
            optimizer_dep.step()
            # optimizer_all.step()        
                                   


            if (step+1) % args.log_step == 0:
                errors = OrderedDict([
                                    ('Loss_meta_train', Loss_meta_train.item()),
                                    ('Loss_meta_test', Loss_meta_test.item()),
                                    ('Loss_cls_train', Loss_cls_train.item()),
                                    ('Loss_cls_test', Loss_cls_test.item()),
                                    ('Loss_dep_train_ave', Loss_dep_train_ave.item()),
                                    ('Loss_dep_test', Loss_dep_test.item()),
                                    ])
                Saver.print_current_errors((epoch+1), (step+1), (iternum), errors)


            #============ tensorboard the log info ============#
            info = {
                'Loss_meta_train': Loss_meta_train.item(),                                                                                     
                'Loss_meta_test': Loss_meta_test.item(),                                                                                     
                'Loss_cls_train': Loss_cls_train.item(),  
                'Loss_cls_test': Loss_cls_test.item(),                                                                                                                                                                                                                                                          
                'Loss_dep_train_ave': Loss_dep_train_ave.item(),                                                                                                                                                                                                                                                          
                'Loss_dep_test': Loss_dep_test.item(),                                                                                                                                                                                                                                                          
                    }           
            for tag, value in info.items():
                summary_writer.add_scalar(tag, value, global_step) 


            global_step+=1


            #############################
            # 2.4 save model parameters #
            #############################
            '''
            if ((step + 1) % args.model_save_step == 0):
                model_save_path = os.path.join(args.results_path, "model")     
                mkdir(model_save_path) 

                torch.save(FeatExtor.state_dict(), os.path.join(model_save_path,
                    "FeatExtor-{}-{}.pt".format(epoch+1, step+1)))
                torch.save(FeatEmbder.state_dict(), os.path.join(model_save_path,
                    "FeatEmbder-{}-{}.pt".format(epoch+1, step+1)))
                torch.save(DepthEstor.state_dict(), os.path.join(model_save_path,
                    "DepthEstor-{}-{}.pt".format(epoch+1, step+1)))
            '''


        # if ((epoch + 1) % args.model_save_epoch == 0):
        #     model_save_path = os.path.join(args.results_path, "model")      
        #     mkdir(model_save_path) 

        #     torch.save(FeatExtor.state_dict(), os.path.join(model_save_path,
        #         "FeatExtor-{}.pt".format(epoch+1)))
        #     torch.save(FeatEmbder.state_dict(), os.path.join(model_save_path,
        #         "FeatEmbder-{}.pt".format(epoch+1)))
        #     torch.save(DepthEstor.state_dict(), os.path.join(model_save_path,
        #         "DepthEstor-{}.pt".format(epoch+1)))
        #     torch.save(Head.state_dict(), os.path.join(model_save_path,
        #         "Head-{}.pt".format(epoch+1)))
        if ((epoch + 1) % args.model_save_epoch == 0):
            model_save_path = os.path.join(args.results_path, "model")      
            mkdir(model_save_path) 

            torch.save(FeatExtor_rgb.state_dict(), os.path.join(model_save_path,
                "FeatExtor_rgb-{}.pt".format(epoch+1)))
            torch.save(FeatExtor_dct.state_dict(), os.path.join(model_save_path,
                "FeatExtor_dct-{}.pt".format(epoch+1)))            
            torch.save(DepthEstor_rgb.state_dict(), os.path.join(model_save_path,
                "DepthEstor_rgb-{}.pt".format(epoch+1)))
            torch.save(DepthEstor_dct.state_dict(), os.path.join(model_save_path,
                "DepthEstor_dct-{}.pt".format(epoch+1)))
            torch.save(ensembelModel.state_dict(), os.path.join(model_save_path,
                "ensembelEmbd-{}.pt".format(epoch+1)))                

    '''
    torch.save(FeatExtor.state_dict(), os.path.join(model_save_path,
        "FeatExtor-final.pt"))
    torch.save(FeatEmbder.state_dict(), os.path.join(model_save_path,
        "FeatEmbder-final.pt"))    
    torch.save(DepthEstor.state_dict(), os.path.join(model_save_path,
        "DepthEstor-final.pt"))
    torch.save(Head.state_dict(), os.path.join(model_save_path,
                "Head-final{}.pt".format(epoch+1)))
    '''

def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
