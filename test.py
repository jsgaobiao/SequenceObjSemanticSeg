#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_sposs_encoder_params, convgru_sposs_decoder_params
from data.MovingMNIST import MovingMNIST
from data.sposs import SemanticPOSS
import torch
import cv2
import time
import yaml
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse

TIMESTAMP = time.strftime("%Y%m%d_%H%M_", time.localtime(time.time()))
parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('-cgru',
                    '--convgru',
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('-m',
                    '--model_path',
                    type=str,
                    required=True,
                    help='model path (trained model for testing)')
parser.add_argument('-d',
                    '--dataset',
                    default='../../Data/MovingMNIST/',
                    type=str,
                    help='dataset dir')
parser.add_argument('-c',
                    '--config',
                    help='config file to use',
                    default="sposs",
                    type=str)
parser.add_argument('-v',
                    '--vis_dir',
                    default="vis",
                    type=str,
                    help='visualization results')
parser.add_argument('--batch_size',
                    default=1,
                    type=int,
                    help='mini-batch size')
# TODO：目前输入输出序列的长度是固定参数，要改为可变值
parser.add_argument('-frames_input',
                    default=10,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=10,
                    type=int,
                    help='sum of predict frames')          
parser.add_argument('-sa',
                    '--sample',
                    default=10,
                    type=int,
                    help='select some samples for visualization')                 
args = parser.parse_args()

# 为CPU/GPU设置随机种子
random_seed = 666
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
# 固定CUDA的随机种子    
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# save_dir = './save_model/' + TIMESTAMP

# open dataset config file
try:
    print("Opening dataset config file %s" % args.config)
    CONFIG = yaml.safe_load(open(os.path.join("data", args.config+".yaml"), 'r'))
except Exception as e:
    print(e)
    print("Error opening dataset yaml file.")
    quit()

if not os.path.isdir(args.dataset):
    raise ValueError("dataset folder [{0}] doesn't exist! Exiting...".format(args.dataset))

testFolder = SemanticPOSS(root=args.dataset,
                        sequences=CONFIG["split"]["train"],
                        labels=CONFIG["labels"],
                        color_map=CONFIG["color_map"],
                        learning_map=CONFIG["learning_map"],
                        learning_map_inv=CONFIG["learning_map_inv"],
                        sensor=CONFIG["sensor"],
                        gt=True)
testLoader = torch.utils.data.DataLoader(testFolder,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=4,
                                        pin_memory=True,
                                        drop_last=True)

if args.convlstm:
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params
if args.convgru:
    encoder_params = convgru_sposs_encoder_params
    decoder_params = convgru_sposs_decoder_params
else:
    encoder_params = convgru_sposs_encoder_params
    decoder_params = convgru_sposs_decoder_params

def normalization(data, imax=-1):
    if imax != -1:
        _range = imax - np.min(data)
    else:
        _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def SaveVis(model_dir, id, _inputs, _labels, _preds):
    # 取value大的维度作为预测的obj mask/background
    preds = _preds.data.max(2, keepdim=True)[1]
    inputs = _inputs.cpu().numpy()
    labels = _labels.cpu().numpy()
    preds = preds.cpu().numpy()
    # 根据inputs计算range image
    range_img = np.zeros((inputs.shape[0], inputs.shape[1], 1, inputs.shape[3], inputs.shape[4]))
    range_img[:,:,0,:,:] = np.sqrt(inputs[:,:,0,:,:]**2 + inputs[:,:,1,:,:]**2 + inputs[:,:,2,:,:]**2)
    range_img = normalization(range_img, 15)
    # Batch, Seq, Channel, H, W
    # print(inputs.min(), inputs.max(), inputs.dtype, inputs.shape)
    # print(range_img.min(), range_img.max(), range_img.dtype, range_img.shape)
    # print(labels.min(), labels.max(), labels.dtype, labels.shape)
    # print(preds.min(), preds.max(), preds.dtype, preds.shape)
    
    save_dir = os.path.join(args.vis_dir, model_dir, str(id))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    vis_dim = (range_img.shape[4], range_img.shape[3]*3)
    # 距离图像
    for i in range(range_img.shape[1]):
        resized_ri = cv2.resize((np.minimum(range_img[0,i,0,:,:]*255, 255)).astype(np.uint8), vis_dim, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(save_dir, "input_"+str(i)+".png"), resized_ri)
    # 真值的obj mask
    for i in range(labels.shape[1]):
        resized_lab = cv2.resize((labels[0,i,0,:,:]*255).astype(np.uint8), vis_dim, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(save_dir, "label_"+str(i)+".png"), resized_lab)
    # 预测的obj mask
    for i in range(preds.shape[1]):
        resized_pre = cv2.resize((preds[0,i,0,:,:]*255).astype(np.uint8), vis_dim, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(save_dir, "pred_"+str(i+inputs.shape[1]-preds.shape[1])+".png"), resized_pre)
    # 距离图像+真值obj mask
    for i in range(range_img.shape[1]):
        resized_ri = cv2.resize((np.minimum(range_img[0,i,0,:,:]*255, 255)).astype(np.uint8), vis_dim, interpolation = cv2.INTER_LINEAR)
        resized_lab = cv2.resize((labels[0,i,0,:,:]).astype(np.uint8), vis_dim, interpolation = cv2.INTER_LINEAR)
        # obj mask位置涂成红色
        resized_lab = np.where(resized_lab != 0, 255, resized_ri)
        resized_ri = np.expand_dims(resized_ri, axis=2)
        resized_lab = np.expand_dims(resized_lab, axis=2)
        resized_ri_lab = np.concatenate((resized_ri, resized_ri, resized_lab), axis=2).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, "input_label_mask_"+str(i)+".png"), resized_ri_lab)

def test():
    '''
    main function to run the testing
    '''
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = ED(encoder, decoder)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    # 加载待测试模型
    if os.path.exists(args.model_path):
        # load existing model
        print('==> loading existing model ' + args.model_path)
        model_info = torch.load(args.model_path)
        net.load_state_dict(model_info['state_dict'])
        model_dir = args.model_path.split('/')[-2]
    else:
        raise Exception("Invalid model path!")

    # 创建存储可视化图片的路径
    if not os.path.isdir(args.vis_dir):
        os.makedirs(args.vis_dir)

    class_weights = torch.FloatTensor([1.0, 100.0]).cuda()    
    lossfunction = nn.CrossEntropyLoss(weight=class_weights).cuda()
    # to track the testing loss as the model testing
    test_losses = []
    # to track the average testing loss per epoch as the model testing
    avg_test_losses = []
    
    ######################
      # test the model #
    ######################
    with torch.no_grad():
        net.eval()  # 将module设置为 eval mode，只影响dropout和batchNorm
        # tqdm 进度条
        t = tqdm(testLoader, total=len(testLoader))
        for i, (seq_len, scan_seq, label_seq, mask_seq, label_id) in enumerate(t):
            # 序列长度不固定，至少前2帧用来输入，固定预测后3帧
            inputs = scan_seq.to(device)[:,:-3,...]   # B,S,C,H,W
            label = mask_seq.to(device)[:,(seq_len-3):,...]     # B,S,C,H,W    
            pred = net(inputs)
            SaveVis(model_dir, i, scan_seq.to(device), mask_seq.to(device), pred)
            seq_number, batch_size, input_channel, height, width = pred.size() 
            pred = pred.reshape(-1, input_channel, height, width)  # reshape to B*S,C,H,W
            seq_number, batch_size, input_channel, height, width = label.size() 
            label = label.reshape(-1, height, width) # reshape to B*S,H,W
            label = label.to(device=device, dtype=torch.long)
            loss = lossfunction(pred, label)
            loss_aver = loss.item() / (label.shape[0])
            # record test loss
            test_losses.append(loss_aver)
            t.set_postfix({
                'test_loss': '{:.6f}'.format(loss_aver),
                'cnt': '{:02d}'.format(i)
            })
            if i >= args.sample:
                break

    torch.cuda.empty_cache()
    # print test statistics
    # calculate average loss over an epoch
    test_loss = np.average(test_losses)
    avg_test_losses.append(test_loss)

    # epoch_len = len(str(args.epochs))

    test_losses = []


if __name__ == "__main__":
    test()
