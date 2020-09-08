#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_sposs_encoder_params, convgru_sposs_decoder_params
from data.MovingMNIST import MovingMNIST
from data.sposs import SemanticPOSS
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
import cv2
import time
import yaml
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse
from utils import get_visualization_example

TIMESTAMP = time.strftime("%Y%m%d_", time.localtime(time.time()))
parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('-cgru',
                    '--convgru',
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('-d',
                    '--dataset',
                    help='dataset path',
                    type=str)
parser.add_argument('-c',
                    '--config',
                    help='config file to use',
                    default="sposs",
                    type=str)
# TODO：目前输入输出序列的长度是固定参数，要改为可变值
# parser.add_argument('-frames_input',
#                     default=5,
#                     type=int,
#                     help='sum of input frames')
# parser.add_argument('-frames_output',
#                     default=5,
#                     type=int,
#                     help='sum of predict frames')
parser.add_argument('--mname', "-mn", default="", type=str, help='brief description of the model')
parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('-epochs', default=500, type=int, help='sum of epochs')
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

save_dir = './save_model/' + TIMESTAMP + args.mname

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

trainFolder = SemanticPOSS(root=args.dataset,
                        sequences=CONFIG["split"]["train"],
                        labels=CONFIG["labels"],
                        color_map=CONFIG["color_map"],
                        learning_map=CONFIG["learning_map"],
                        learning_map_inv=CONFIG["learning_map_inv"],
                        sensor=CONFIG["sensor"],
                        gt=True)
validFolder = SemanticPOSS(root=args.dataset,
                        sequences=CONFIG["split"]["valid"],
                        labels=CONFIG["labels"],
                        color_map=CONFIG["color_map"],
                        learning_map=CONFIG["learning_map"],
                        learning_map_inv=CONFIG["learning_map_inv"],
                        sensor=CONFIG["sensor"],
                        gt=True)

trainLoader = torch.utils.data.DataLoader(trainFolder,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=8,
                                        pin_memory=True,
                                        drop_last=True)
validLoader = torch.utils.data.DataLoader(validFolder,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=8,
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


def train():
    '''
    main function to run the training
    '''
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = ED(encoder, decoder)
    run_dir = './runs/' + TIMESTAMP + args.mname
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=200, verbose=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoin.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0

    class_weights = torch.FloatTensor([1.0, 15.0]).cuda()    
    lossfunction = nn.CrossEntropyLoss(weight=class_weights).cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=5,
                                                      verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    min_train_loss = np.inf
    for epoch in range(cur_epoch, args.epochs + 1):
        print(time.strftime("now time: %Y%m%d_%H:%M", time.localtime(time.time())))
        ###################
        # train the model #
        ###################
        # tqdm 进度条
        t = tqdm(trainLoader, total=len(trainLoader))
        for i, (seq_len, scan_seq, _, mask_seq, _) in enumerate(t):
            # 序列长度不固定，至少前2帧用来输入，固定预测后3帧
            inputs = scan_seq.to(device)[:,:-3,...]   # B,S,C,H,W
            label = mask_seq.to(device)[:,(seq_len-3):,...]     # B,S,C,H,W    
            optimizer.zero_grad()
            net.train()         # 将module设置为 training mode，只影响dropout和batchNorm
            pred = net(inputs)  # B,S,C,H,W

            # 在tensorboard中绘制可视化结果
            if i % 100 == 0:
                grid_ri_lab, grid_pred = get_visualization_example(scan_seq.to(device), mask_seq.to(device), pred, device)
                tb.add_image('visualization/train/rangeImage_gtMask', grid_ri_lab, global_step=epoch)
                tb.add_image('visualization/train/prediction', grid_pred, global_step=epoch)
                
            seq_number, batch_size, input_channel, height, width = pred.size() 
            pred = pred.reshape(-1, input_channel, height, width)  # reshape to B*S,C,H,W
            seq_number, batch_size, input_channel, height, width = label.size() 
            label = label.reshape(-1, height, width) # reshape to B*S,H,W
            label = label.to(device=device, dtype=torch.long)
            # 计算loss
            loss = lossfunction(pred, label)
            loss_aver = loss.item() / (label.shape[0] * batch_size) 
            train_losses.append(loss_aver)
            loss.backward()
            # 防止梯度爆炸，进行梯度裁剪，指定clip_value之后，裁剪的范围就是[-clip_value, clip_value]
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=30.0)
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
        tb.add_scalar('TrainLoss', np.average(train_losses), epoch)
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            # 将module设置为 eval模式, 只影响dropout和batchNorm
            net.eval()
            # tqdm 进度条
            t = tqdm(validLoader, total=len(validLoader))
            for i, (seq_len, scan_seq, _, mask_seq, _) in enumerate(t):
                if i == 300:    # 限制 validate 数量
                    break
                # 序列长度不固定，至少前2帧用来输入，固定预测后3帧
                inputs = scan_seq.to(device)[:,:-3,...]   # B,S,C,H,W
                label = mask_seq.to(device)[:,(seq_len-3):,...]     # B,S,C,H,W    
                pred = net(inputs)

                # 在tensorboard中绘制可视化结果
                if i % 100 == 0:
                    grid_ri_lab, grid_pred = get_visualization_example(scan_seq.to(device), mask_seq.to(device), pred, device)
                    tb.add_image('visualization/valid/rangeImage_gtMask', grid_ri_lab, global_step=epoch)
                    tb.add_image('visualization/valid/prediction', grid_pred, global_step=epoch)
                    
                seq_number, batch_size, input_channel, height, width = pred.size() 
                pred = pred.reshape(-1, input_channel, height, width)  # reshape to B*S,C,H,W
                seq_number, batch_size, input_channel, height, width = label.size() 
                label = label.reshape(-1, height, width) # reshape to B*S,H,W
                label = label.to(device=device, dtype=torch.long)
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / (label.shape[0] * batch_size)
                # record validation loss
                valid_losses.append(loss_aver)
                #print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })
                # get_visualization_example(inputs, label, pred)

        tb.add_scalar('ValidLoss', np.average(valid_losses), epoch)
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(args.epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        # 保存train loss最低的模型
        if (train_loss < min_train_loss):
            torch.save(model_dict, save_dir + "/" + "best_train_checkpoint.pth.tar")
            min_train_loss = train_loss
        # 保存valid loss最低的模型
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # end for

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)


if __name__ == "__main__":
    train()