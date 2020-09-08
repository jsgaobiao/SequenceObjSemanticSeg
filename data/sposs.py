import os
import numpy as np
import torch
from numba import jit
from torch.utils.data import Dataset
# from common.laserscan import LaserScan, SemLaserScan

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']

# __init__：创建文件目录列表，根据train/val/test划分数据
# __getitem__：根据seq文件信息读取序列数据

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


class SemanticPOSS(Dataset):
  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               sensor,              # sensor to parse scans from
            #    max_points=150000,   # max number of points present in dataset
               gt=True):            # send ground truth?
    # save deats
    self.root = root
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    # self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]
    self.sensor_img_means = np.array(sensor["img_means"], dtype=np.float32)
    self.sensor_img_stds = np.array(sensor["img_stds"], dtype=np.float32)
    # self.sensor_fov_up = sensor["fov_up"]
    # self.sensor_fov_down = sensor["fov_down"]
    # self.max_points = max_points
    self.gt = gt

    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)
    self.nclasses = len(self.learning_map_inv)

    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("SPOSS dataset folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("SPOSS dataset folder doesn't exist! Exiting...")

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))
    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))
    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))
    # make sure sequences is a list
    assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.scan_files = []
    self.label_files = []
    self.seq_dirs = []

    # fill in with names, checking that all sequences are complete
    for seq in self.sequences:
      # to string
      seq = '{0:02d}'.format(int(seq))
      print("parsing seq {}".format(seq))

      # get paths for each
      scan_path = os.path.join(self.root, seq, "velodyne")
      label_path = os.path.join(self.root, seq, "labels")
      seq_path = os.path.join(self.root, seq, "seq")

      # get files
      scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
      label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(label_path)) for f in fn if is_label(f)]
      # get sequence subdirs (seq序列的列表)
      seq_stat = np.loadtxt(os.path.join(seq_path, "statistic.txt"))
      seq_dirs = [os.path.join(os.path.expanduser(seq_path), str(sd)) for sd in range(len(seq_stat))]
      # print("seq_dirs: \n")
      # print(seq_dirs[:10])
        
      # check all scans have labels
      if self.gt:
        assert(len(scan_files) == len(label_files))

      # extend list
      self.scan_files.extend(scan_files)
      self.label_files.extend(label_files)
      self.seq_dirs.extend(seq_dirs)

    # sort for correspondance
    self.scan_files.sort()
    self.label_files.sort()

    print("Using {} scans from sequences {}".format(len(self.scan_files), self.sequences))


  def __getitem__(self, index):
    seq_info = np.loadtxt(os.path.join(self.seq_dirs[index], "seqInfo.txt"))
    # DEBUG
    if (seq_info.shape[0] > 8):
      seq_info = seq_info[:8]

    mask_seq = np.zeros((len(seq_info), 40, 256), dtype=np.int16)
    scan_seq = np.zeros((len(seq_info), 40, 256, 4), dtype=np.float32)
    label_seq = np.zeros((len(seq_info), 40, 256), dtype=np.int32)
    label_id = 0
    
    # 遍历seq的每一帧
    for i, dat in zip(range(len(seq_info)), seq_info):
      dat = [int(x) for x in dat]

      # DEBUG
      if (dat[4] == 1800):
        dat[2] -= 1
        dat[4] -= 1
      # DEBUG

      frame_id = dat[0]
      label_id = dat[5]
      mask_file = os.path.join(self.seq_dirs[index], str(frame_id)+"_"+str(label_id)+".mask")
      _dir = self.seq_dirs[index].split("/seq/")[0]
      scan_file = os.path.join(_dir, "velodyne", "{:0>6d}".format(frame_id)+".bin")
      if self.gt:
        label_file = os.path.join(_dir, "labels", "{:0>6d}".format(frame_id)+".label")

      # 读取mask
      mask = np.fromfile(mask_file, dtype=np.int16)
      mask = mask.reshape(dat[3]-dat[1]+1, dat[4]-dat[2]+1)
      # 读取scan,label
      scan = np.fromfile(scan_file, dtype=np.float32)
      # scan = scan.reshape(-1, 4)
      scan = scan.reshape(self.sensor_img_H, self.sensor_img_W, 4)
      scan = (scan[:,:,:] - self.sensor_img_means[1:5]) / self.sensor_img_stds[1:5]
      label = np.fromfile(label_file, dtype=np.int32)
      label = np.array(list(map(lambda x: (x & ((1 << 16) - 1)), label)), dtype=np.int32)
      label = label.reshape(self.sensor_img_H, self.sensor_img_W)
      
      # 在距离图像和label中截取物体所在的小区域
      sub_scan = scan[dat[1]:dat[3]+1, dat[2]:dat[4]+1, :]
      sub_label = label[dat[1]:dat[3]+1, dat[2]:dat[4]+1]

      # 加入物体跟踪序列中
      mask_seq[i] = mask
      scan_seq[i] = sub_scan
      label_seq[i] = sub_label

    # S,H,W,C --> S,C,H,W
    scan_seq = np.array(scan_seq).transpose((0,3,1,2))
    label_seq = np.expand_dims(np.array(label_seq), axis=1)
    mask_seq = np.expand_dims(np.array(mask_seq), axis=1)
    # return
    return len(seq_info), scan_seq, label_seq, mask_seq, label_id

  def __len__(self):
    # print(len(self.seq_dirs))
    # return 5
    return len(self.seq_dirs)

#   @staticmethod
#   def map(label, mapdict):
#     # put label from original values to xentropy
#     # or vice-versa, depending on dictionary values
#     # make learning map a lookup table
#     maxkey = 0
#     for key, data in mapdict.items():
#       if isinstance(data, list):
#         nel = len(data)
#       else:
#         nel = 1
#       if key > maxkey:
#         maxkey = key
#     # +100 hack making lut bigger just in case there are unknown labels
#     if nel > 1:
#       lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
#     else:
#       lut = np.zeros((maxkey + 100), dtype=np.int32)
#     for key, data in mapdict.items():
#       try:
#         lut[key] = data
#       except IndexError:
#         print("Wrong key ", key)
#     # do the mapping
#     return lut[label]


# class Parser():
#   # standard conv, BN, relu
#   def __init__(self,
#                root,              # directory for data
#                train_sequences,   # sequences to train
#                valid_sequences,   # sequences to validate.
#                test_sequences,    # sequences to test (if none, don't get)
#                labels,            # labels in data
#                color_map,         # color for each label
#                learning_map,      # mapping for training labels
#                learning_map_inv,  # recover labels from xentropy
#                sensor,            # sensor to use
#                max_points,        # max points in each scan in entire dataset
#                batch_size,        # batch size for train and val
#                workers,           # threads to load data
#                gt=True,           # get gt?
#                shuffle_train=True):  # shuffle training set?
#     super(Parser, self).__init__()

#     # if I am training, get the dataset
#     self.root = root
#     self.train_sequences = train_sequences
#     self.valid_sequences = valid_sequences
#     self.test_sequences = test_sequences
#     self.labels = labels
#     self.color_map = color_map
#     self.learning_map = learning_map
#     self.learning_map_inv = learning_map_inv
#     self.sensor = sensor
#     self.max_points = max_points
#     self.batch_size = batch_size
#     self.workers = workers
#     self.gt = gt
#     self.shuffle_train = shuffle_train

#     # number of classes that matters is the one for xentropy
#     self.nclasses = len(self.learning_map_inv)

#     # Data loading code
#     self.train_dataset = SemanticKitti(root=self.root,
#                                        sequences=self.train_sequences,
#                                        labels=self.labels,
#                                        color_map=self.color_map,
#                                        learning_map=self.learning_map,
#                                        learning_map_inv=self.learning_map_inv,
#                                        sensor=self.sensor,
#                                        max_points=max_points,
#                                        gt=self.gt)

#     self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
#                                                    batch_size=self.batch_size,
#                                                    shuffle=self.shuffle_train,
#                                                    num_workers=self.workers,
#                                                    pin_memory=True,
#                                                    drop_last=True)
#     assert len(self.trainloader) > 0
#     self.trainiter = iter(self.trainloader)

#     self.valid_dataset = SemanticKitti(root=self.root,
#                                        sequences=self.valid_sequences,
#                                        labels=self.labels,
#                                        color_map=self.color_map,
#                                        learning_map=self.learning_map,
#                                        learning_map_inv=self.learning_map_inv,
#                                        sensor=self.sensor,
#                                        max_points=max_points,
#                                        gt=self.gt)

#     self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
#                                                    batch_size=self.batch_size,
#                                                    shuffle=False,
#                                                    num_workers=self.workers,
#                                                    pin_memory=True,
#                                                    drop_last=True)
#     assert len(self.validloader) > 0
#     self.validiter = iter(self.validloader)

#     if self.test_sequences:
#       self.test_dataset = SemanticKitti(root=self.root,
#                                         sequences=self.test_sequences,
#                                         labels=self.labels,
#                                         color_map=self.color_map,
#                                         learning_map=self.learning_map,
#                                         learning_map_inv=self.learning_map_inv,
#                                         sensor=self.sensor,
#                                         max_points=max_points,
#                                         gt=False)

#       self.testloader = torch.utils.data.DataLoader(self.test_dataset,
#                                                     batch_size=self.batch_size,
#                                                     shuffle=False,
#                                                     num_workers=self.workers,
#                                                     pin_memory=True,
#                                                     drop_last=True)
#       assert len(self.testloader) > 0
#       self.testiter = iter(self.testloader)

#   def get_train_batch(self):
#     scans = self.trainiter.next()
#     return scans

#   def get_train_set(self):
#     return self.trainloader

#   def get_valid_batch(self):
#     scans = self.validiter.next()
#     return scans

#   def get_valid_set(self):
#     return self.validloader

#   def get_test_batch(self):
#     scans = self.testiter.next()
#     return scans

#   def get_test_set(self):
#     return self.testloader

#   def get_train_size(self):
#     return len(self.trainloader)

#   def get_valid_size(self):
#     return len(self.validloader)

#   def get_test_size(self):
#     return len(self.testloader)

#   def get_n_classes(self):
#     return self.nclasses

#   def get_original_class_string(self, idx):
#     return self.labels[idx]

#   def get_xentropy_class_string(self, idx):
#     return self.labels[self.learning_map_inv[idx]]

#   def to_original(self, label):
#     # put label in original values
#     return SemanticKitti.map(label, self.learning_map_inv)

#   def to_xentropy(self, label):
#     # put label in xentropy values
#     return SemanticKitti.map(label, self.learning_map)

#   def to_color(self, label):
#     # put label in original values
#     label = SemanticKitti.map(label, self.learning_map_inv)
#     # put label in color
#     return SemanticKitti.map(label, self.color_map)
