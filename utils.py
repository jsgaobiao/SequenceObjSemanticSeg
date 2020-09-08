from torch import nn
from collections import OrderedDict
import cv2
import torch
import torchvision
import os
import numpy as np

def normalization(data, imax=-1):
    if imax != -1:
        _range = imax - np.min(data)
    else:
        _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def get_visualization_example(scan_seq, gtmask_seq, mask_seq, device):
    # 最多可视化的图片数量
    MAX_VISUALIZE_IMG = 10
    # 取value大的维度作为预测的obj mask/background
    preds = mask_seq.data.max(2, keepdim=True)[1]
    inputs = scan_seq.cpu().numpy()
    labels = gtmask_seq.cpu().numpy()
    preds = preds.cpu().numpy()
    if (inputs.shape[1] > MAX_VISUALIZE_IMG):
        inputs = inputs[:,inputs.shape[1]-MAX_VISUALIZE_IMG:,...]
    if (labels.shape[1] > MAX_VISUALIZE_IMG):
        labels = labels[:,labels.shape[1]-MAX_VISUALIZE_IMG:,...]
    if (preds.shape[1] > MAX_VISUALIZE_IMG):
        preds = preds[:,:MAX_VISUALIZE_IMG,...]
    # 根据inputs计算range image
    range_img = np.zeros((inputs.shape[0], inputs.shape[1], 1, inputs.shape[3], inputs.shape[4]))
    range_img[:,:,0,:,:] = np.sqrt(inputs[:,:,0,:,:]**2 + inputs[:,:,1,:,:]**2 + inputs[:,:,2,:,:]**2)
    range_img = normalization(range_img, 15)
    
    vis_dim = (range_img.shape[4], range_img.shape[3]*3)
    # 预测的obj mask
    resized_pre = np.zeros((preds.shape[1], 1, vis_dim[1], vis_dim[0]), dtype=np.uint8)
    for i in range(preds.shape[1]):
        resized_pre[i,0] = cv2.resize((preds[0,i,0,:,:]*255).astype(np.uint8), vis_dim, interpolation = cv2.INTER_LINEAR) # H,W

    # 距离图像+真值obj mask        # B,C,H,W
    resized_ri_lab = np.zeros((range_img.shape[1], 3, vis_dim[1], vis_dim[0]), dtype=np.uint8)
    for i in range(range_img.shape[1]):
        resized_ri = cv2.resize((np.minimum(range_img[0,i,0,:,:]*255, 255)).astype(np.uint8), vis_dim, interpolation = cv2.INTER_LINEAR)
        resized_lab = cv2.resize((labels[0,i,0,:,:]).astype(np.uint8), vis_dim, interpolation = cv2.INTER_LINEAR)
        # obj mask位置涂成红色
        resized_lab = np.where(resized_lab != 0, 255, resized_ri)
        resized_ri = np.expand_dims(resized_ri, axis=0)
        resized_lab = np.expand_dims(resized_lab, axis=0)
        resized_ri_lab[i] = np.concatenate((resized_lab, resized_ri, resized_ri), axis=0).astype(np.uint8) # C,H,W

    ret_grid_pred = torchvision.utils.make_grid(torch.from_numpy(resized_pre).to(device), nrow=10, padding=2)
    ret_grid_ri_lab = torchvision.utils.make_grid(torch.from_numpy(resized_ri_lab).to(device), nrow=10, padding=2)
    return ret_grid_ri_lab, ret_grid_pred
        
def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))
