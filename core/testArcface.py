import os
from collections import OrderedDict
import torchvision.utils as vutils
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os.path as osp
plt.get_backend()
from sklearn.metrics import roc_curve, auc, confusion_matrix
from torch import nn
from misc.utils import get_inf_iterator, mkdir
from misc import evaluate
from torch.nn import DataParallel
from torch.nn import Parameter
import numpy as np
import h5py
import math
import torch.nn.functional as F


from pdb import set_trace as st

class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel. 
                    if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """
    def __init__(self, in_features, out_features, device_id, s = 64.0, m = 0.50, easy_margin = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s = s
        self.m = m
        
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------        
        cosine = F.linear(F.normalize(input), F.normalize(self.weight.cuda()))

        # --------------------------- convert label to one-hot ---------------------------
        # torch.cuda.set_device(1)
        one_hot = torch.zeros(cosine.size()).cuda()
        one_hot.scatter_(1, label.view(-1, 1).long().cuda(), 1)

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * cosine) + ((1.0 - one_hot) * cosine)
        output *= 64

        return output

def Test(args, FeatExtor, FeatEmbder, 
       data_loader_target, modelIdx):

    # print("***The type of norm is: {}".format(normtype))
    if not os.path.isdir(args.results_path + "/txt"):
        os.mkdir(args.results_path + "/txt")

    ####################
    # 1. setup network #
    ####################
    # set train state for Dropout and BN layers
    FeatExtor.eval()
    FeatEmbder.eval()
    Head = ArcFace(in_features=1000, out_features=2, device_id=[0,1])
    head_path = osp.join(args.results_path, "model", 'Head-'+str(modelIdx)+'.pt')
    checkpoint_head = torch.load(head_path, map_location={'cuda:1':'cuda:0'})
    Head.load_state_dict(checkpoint_head)
    Head.eval()

    if torch.cuda.device_count() > 1:
        FeatEmbder = DataParallel(FeatEmbder)    
        FeatExtor = DataParallel(FeatExtor)    

    score_list = []
    label_list = []
    pred_list = []

    idx = 0

    with torch.no_grad():
        for (catimages, labels) in data_loader_target:
            images = catimages.cuda()
            # labels = labels.long().squeeze().cuda()

            # _,feat  = FeatExtor(images)
            feat  = FeatExtor(images)
            embd  = FeatEmbder(feat)
            theta = Head(embd, torch.zeros(len(labels)))
            probablity = torch.nn.functional.softmax(theta, dim=1).cpu().detach().numpy().copy()
            
            # score = torch.sigmoid(label_pred).cpu().detach().numpy()
            score = probablity[:,1:]#np.squeeze(score, 1)

            pred = np.round(score)
            pred = np.array(pred, dtype=int)
            pred = np.squeeze(pred, 1)

            pred_list = pred_list + pred.tolist()
            labels = labels.tolist()

            score_list = score_list + score.tolist()
            label_list = label_list + labels

            print('SampleNum:{} in total:{}, score:{}'.format(idx,len(data_loader_target), score.squeeze()))

            idx+=1
    fpr, tpr, _ = roc_curve(label_list, score_list)
    fnr = 1-tpr
    if args.dataset1 == args.tstdataset: # inter dataset
        roc_auc = auc(fpr, tpr) #x,y
        plot_score(args, modelIdx, fpr, tpr, fnr, roc_auc, cross_data = False, log = False)
        plot_score(args, modelIdx, fpr, tpr, fnr, roc_auc, cross_data = False, log = True)
    elif args.dataset1 != args.tstdataset: # cross dataset
        roc_auc = auc(fpr, fnr) #x,y
        plot_score(args, modelIdx, fpr, tpr, fnr, roc_auc, cross_data = True, log = False)
        plot_score(args, modelIdx, fpr, tpr, fnr, roc_auc, cross_data = False, log = False)# to record auc

    # plot_score()
    model_eval(args, label_list, pred_list, score_list, modelIdx)

def plot_score(args, modelIdx,fpr, tpr, fnr, roc_auc, cross_data = False, log = False):
    fig = plt.figure()
    lw = 2

    if not cross_data:
        if log:
            plt.xscale("log")
        elif not log:
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.ylabel('True Living Rate')
    elif cross_data:
        if log:
            plt.xscale("log")
        elif not log:
            plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')#(x0,x1), (y0,y1)
        plt.plot(fpr, fnr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.ylabel('False Fake Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Living Rate')

    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #fig.savefig('/tmp/roc.png')
    curve_save_path = os.path.join(args.results_path, "curve")
    mkdir(curve_save_path)
    if log:
        if cross_data:
            plt.savefig("%s/ROC_cross_log_%s_%03d.png" %(curve_save_path, args.tstdataset, modelIdx))
        elif not cross_data:
            plt.savefig("%s/ROC_log_%s_%03d.png" %(curve_save_path, args.tstdataset, modelIdx))
    elif not log:
        if cross_data:
            plt.savefig("%s/ROC_cross_%s_%03d.png" %(curve_save_path, args.tstdataset, modelIdx))
        elif not cross_data:
            plt.savefig("%s/ROC_%s_%03d.png" %(curve_save_path, args.tstdataset, modelIdx))

def model_eval(args, actual, pred, predsDecimal, modelIdx):
    with open(args.results_path + "/txt/"+args.tstdataset+"_"+args.tst_txt_name,"a+") as f:
        # calculate eer
        
        fpr, tpr, threshold = roc_curve(actual,predsDecimal)          
        fnr = 1-tpr
        diff = np.absolute(fnr - fpr)
        idx = np.nanargmin(diff)
        # print(threshold[idx])
        eer = np.mean((fpr[idx],fnr[idx]))        

        avg = np.add(fpr, fnr)
        idx = np.nanargmin(avg)
        hter = np.mean((fpr[idx],fnr[idx])) 

        fpr_at_10e_m3_idx = np.argmin(np.abs(fpr-10e-3))
        tpr_cor_10e_m3 = tpr[fpr_at_10e_m3_idx+1]

        fpr_at_5e_m3_idx = np.argmin(np.abs(fpr-5e-3))
        print(fpr[-1])
        tpr_cor_5e_m3 = tpr[fpr_at_5e_m3_idx+1]

        fpr_at_10e_m4_idx = np.argmin(np.abs(fpr-10e-4))
        tpr_cor_10e_m4 = tpr[fpr_at_10e_m4_idx+1]

        actual = list(map(lambda el:[el], actual))
        pred = list(map(lambda el:[el], pred))
        
        cm = confusion_matrix(actual, pred)
        TP = cm[0][0]
        TN = cm[1][1]
        FP = cm[1][0]
        FN = cm[0][1]
        accuracy = ((TP+TN))/(TP+FN+FP+TN)
        precision = (TP)/(TP+FP)
        recall = (TP)/(TP+FN)
        f_measure = (2*recall*precision)/(recall+precision)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)		
        error_rate = 1 - accuracy
        apcer = FP/(TN+FP)
        bpcer = FN/(FN+TP)
        acer = (apcer+bpcer)/2
        f.write("="*60)
        f.write('\nModel %03d \n'%(modelIdx))
        f.write('TP:%d, TN:%d,  FP:%d,  FN:%d\n' %(TP,TN,FP,FN))
        f.write('accuracy:%f\n'%(accuracy))
        f.write('precision:%f\n'%(precision))
        f.write('recall:%f\n'%(recall))
        f.write('f_measure:%f\n'%(f_measure))
        f.write('sensitivity:%f\n'%(sensitivity))
        f.write('specificity:%f\n'%(specificity))
        f.write('error_rate:%f\n'%(error_rate))
        f.write('apcer:%f\n'%(apcer))
        f.write('bpcer:%f\n'%(bpcer))
        f.write('acer:%f\n'%(acer))
        f.write('eer:%f\n'%(eer))
        f.write('hter:%f\n'%(hter))
        f.write('TPR@FPR=10E-3:%f\n'%(tpr_cor_10e_m3))
        f.write('TPR@FPR=5E-3:%f\n'%(tpr_cor_5e_m3))
        f.write('TPR@FPR=10E-4:%f\n\n'%(tpr_cor_10e_m4))
