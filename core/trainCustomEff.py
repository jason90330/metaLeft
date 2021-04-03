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

def Train(args, FeatExtor, DepthEstor, FeatEmbder,
        data_loader1_real, data_loader1_fake,
        # data_loader2_real, data_loader2_fake,
        # data_loader3_real, data_loader3_fake,
        # data_loader_target,
        summary_writer, Saver):
            
    ####################
    # 1. setup network #
    ####################
    # set train state for Dropout and BN layers
    FeatExtor.train()
    DepthEstor.train()
    FeatEmbder.train()
    
    head_dict = {'ArcFace': ArcFace(in_features = 1000, out_features = 2, device_id = [0,1]),
                'CosFace': CosFace(in_features = 1000, out_features = 2, device_id = [0,1]),
                'SphereFace': SphereFace(in_features = 1000, out_features = 2, device_id = [0,1]),
                'Am_softmax': Am_softmax(in_features = 1000, out_features = 2, device_id = [0,1])}
    Head = head_dict[args.head_name]
    Head.train()
    


    FeatExtor = DataParallel(FeatExtor)    
    DepthEstor = DataParallel(DepthEstor)    
    # FeatEmbder = DataParallel(FeatEmbder)    
 


    # setup criterion and optimizer
    criterionCls = FocalLoss(outNum = 2)
    # criterionCls = nn.BCEWithLogitsLoss()
    criterionDepth = torch.nn.MSELoss()


    if args.optimizer is 'adam':
        optimizer_embd = optim.AdamW(itertools.chain(FeatEmbder.parameters()),
                                   lr=args.lr,
                                   betas=(args.beta1, args.beta2),
                                   weight_decay=args.weight_decay)

        optimizer_extr = optim.AdamW(itertools.chain(FeatExtor.parameters()),
                                   lr=args.lr,
                                   betas=(args.beta1, args.beta2),
                                   weight_decay=args.weight_decay)

        optimizer_dep = optim.AdamW(itertools.chain(DepthEstor.parameters()),
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
    iternum = int(max(len(data_loader1_real),len(data_loader1_fake))/3)

    print('iternum={}'.format(iternum))

    ####################
    # 2. train network #
    ####################
    global_step = 0

    for epoch in range(args.epochs):

        data1_real = get_inf_iterator(data_loader1_real)
        data1_fake = get_inf_iterator(data_loader1_fake)

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

            cat_img1_real, depth_img1_real, lab1_real = next(data1_real)
            cat_img1_fake, depth_img1_fake, lab1_fake = next(data1_fake)

            cat_img2_real, depth_img2_real, lab2_real = next(data1_real)
            cat_img2_fake, depth_img2_fake, lab2_fake = next(data1_fake)

            cat_img3_real, depth_img3_real, lab3_real = next(data1_real)
            cat_img3_fake, depth_img3_fake, lab3_fake = next(data1_fake)

            #============ one batch collection ============# 

            catimg1 = torch.cat([cat_img1_real,cat_img1_fake],0).cuda()
            depth_img1 = torch.cat([depth_img1_real,depth_img1_fake],0).cuda()
            lab1 = torch.cat([lab1_real,lab1_fake],0).float().cuda()

            catimg2 = torch.cat([cat_img2_real,cat_img2_fake],0).cuda()
            depth_img2 = torch.cat([depth_img2_real,depth_img2_fake],0).cuda()
            lab2 = torch.cat([lab2_real,lab2_fake],0).float().cuda()

            catimg3 = torch.cat([cat_img3_real,cat_img3_fake],0).cuda()
            depth_img3 = torch.cat([depth_img3_real,depth_img3_fake],0).cuda()
            lab3 = torch.cat([lab3_real,lab3_fake],0).float().cuda()

            catimg = torch.cat([catimg1,catimg2,catimg3],0)
            depth_GT = torch.cat([depth_img1,depth_img2,depth_img3],0)
            label = torch.cat([lab1,lab2,lab3],0)                   

           #============ doamin list augmentation ============# 
            catimglist = [catimg1,catimg2,catimg3]
            lablist = [lab1,lab2,lab3]
            deplist = [depth_img1,depth_img2,depth_img3]            

            domain_list = list(range(len(catimglist)))
            random.shuffle(domain_list) 
            
            meta_train_list = domain_list[:args.metatrainsize] 
            meta_test_list = domain_list[args.metatrainsize:]
            print('metatrn={}, metatst={}'.format(meta_train_list, meta_test_list[0]))
 
            
            #============ meta training ============#

            Loss_dep_train = 0.0
            Loss_cls_train = 0.0

            adapted_state_dicts = []

            for index in meta_train_list:

                catimg_meta = catimglist[index]
                lab_meta = lablist[index]
                depGT_meta = deplist[index]

                batchidx = list(range(len(catimg_meta)))
                random.shuffle(batchidx)
                
                img_rand = catimg_meta[batchidx,:]
                lab_rand = lab_meta[batchidx]
                depGT_rand = depGT_meta[batchidx,:]

                '''
                feat = FeatExtor(img_rand)
                pred = FeatEmbder(feat)          
                depth_Pre = DepthEstor(feat)

                Loss_cls = criterionCls(pred.squeeze(), lab_rand)
                Loss_dep = criterionDepth(depth_Pre, depGT_rand)

                Loss_dep_train+=Loss_dep
                Loss_cls_train+=Loss_cls
                '''
                #=arc
                feat = FeatExtor(img_rand)
                embd = FeatEmbder(feat)          
                depth_Pre = DepthEstor(feat)

                thetas = Head(embd, lab_rand)
                Loss_cls = criterionCls(thetas, lab_rand)
                Loss_dep = criterionDepth(depth_Pre, depGT_rand)

                Loss_dep_train+=Loss_dep
                Loss_cls_train+=Loss_cls
                
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
            lab_meta = lablist[index]
            depGT_meta = deplist[index]

            batchidx = list(range(len(catimg_meta)))
            random.shuffle(batchidx)
            
            img_rand = catimg_meta[batchidx,:]
            lab_rand = lab_meta[batchidx]
            depGT_rand = depGT_meta[batchidx,:]

            feat = FeatExtor(img_rand) # use the old weight
            depth_Pre = DepthEstor(feat) # use the old weight
            Loss_dep = criterionDepth(depth_Pre, depGT_rand)

            # use the updated embedder to caculate
            '''
            pred = FeatEmbder(feat)
            Loss_cls_test = criterionCls(pred.squeeze(), lab_rand)
            '''
            #=arc
            embd = FeatEmbder(feat)
            thetas = Head(embd, lab_rand)
            Loss_cls_test = criterionCls(thetas, lab_rand)
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
                Saver.print_current_errors((epoch+1), (step+1), errors)


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


        if ((epoch + 1) % args.model_save_epoch == 0):
            model_save_path = os.path.join(args.results_path, "model")      
            mkdir(model_save_path) 

            torch.save(FeatExtor.state_dict(), os.path.join(model_save_path,
                "FeatExtor-{}.pt".format(epoch+1)))
            torch.save(FeatEmbder.state_dict(), os.path.join(model_save_path,
                "FeatEmbder-{}.pt".format(epoch+1)))
            torch.save(DepthEstor.state_dict(), os.path.join(model_save_path,
                "DepthEstor-{}.pt".format(epoch+1)))
            torch.save(Head.state_dict(), os.path.join(model_save_path,
                "Head-{}.pt".format(epoch+1)))


    torch.save(FeatExtor.state_dict(), os.path.join(model_save_path,
        "FeatExtor-final.pt"))
    torch.save(FeatEmbder.state_dict(), os.path.join(model_save_path,
        "FeatEmbder-final.pt"))    
    torch.save(DepthEstor.state_dict(), os.path.join(model_save_path,
        "DepthEstor-final.pt"))
    torch.save(Head.state_dict(), os.path.join(model_save_path,
                "Head-final{}.pt".format(epoch+1)))


def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
