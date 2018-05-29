from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from torchviz import make_dot, make_dot_from_trace

def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", help =
                        "Image / Directory containing images to perform detection upon",
                        default = "train_data/imgs", type = str)
    parser.add_argument("--labels", help="Directory containing groundtruth",
                        default="train_data/labels")
    parser.add_argument("--det", help =
                        "Image / Directory to store detections to",
                        default = "det")
    parser.add_argument("--bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", help = "Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", help = "NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = "Config file",
                        default = "cfg/yolov3.cfg")
    parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile",
                        default = "yolov3.weights")
    parser.add_argument("--reso", help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416")
    parser.add_argument('--classes', help='file with class id to string mapping',
                        default='data/coco.names')

    return parser.parse_args()

args = arg_parse()
images = osp.realpath(args.images)
labels = osp.realpath(args.labels)
batch_size = int(args.bs)
classes = args.classes
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()
dev = 'cuda' if CUDA else 'cpu'

num_classes = 80
classes = load_classes(classes)

#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

#Set the model in evaluation mode
model.train()

read_dir = time.time()
#Detection phase

try:
    imlist = [osp.join(images, img) for img in os.listdir(images)]
except FileNotFoundError:
    print ("No directory with the name {}".format(images))
    exit()

data = [(x, osp.join(labels, osp.splitext(osp.basename(x))[0] + '.txt')) for x in imlist]
for im_name, label_name in data:
    assert osp.isfile(im_name), "No file with name {}".format(im_name)
    assert osp.isfile(label_name), "No file with name {}".format(label_name)

train_data = data

if not os.path.exists(args.det):
    os.makedirs(args.det)

num_train_batches = (len(train_data) - 1) // batch_size + 1
train_batches = [zip(*(train_data[i*batch_size:(i+1)*batch_size])) for i in range(num_train_batches)]

start_det_loop = time.time()
for i, (im_names, label_names) in enumerate(train_batches):
    start = time.time()

    # load the image
    batch_imgs = [cv2.imread(x) for x in im_names]
    tr_imgs = list(map(prep_image, batch_imgs, [inp_dim for x in range(len(im_names))]))
    tr_imgs = torch.tensor(torch.cat(tr_imgs), device=dev)
    # load the labels
    batch_labels = list(map(lambda x: np.fromfile(x, sep=' ', dtype=np.float32).reshape(-1, 5), label_names))
    batch_labels = list(map(lambda x: x[:, [1, 2, 3, 4, 0]], batch_labels))
    # transform

    model.zero_grad()
    prediction_all = model(Variable(tr_imgs))

    prediction = write_results(prediction_all, confidence, num_classes, nms_conf = nms_thesh)
    loss = model.loss_function(prediction_all, batch_labels, inp_dim)

    #graph = make_dot(loss, params=dict(model.named_parameters()))
    #graph.render('/tmp/graph')
    end = time.time()
    loss.backward()
    b_end = time.time()

    prediction = [] if type(prediction) == int else prediction

    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        objs = [classes[int(x[-1])] for x in prediction if int(x[0]) == im_num]
        gt = list(map(lambda x: classes[int(x)], batch_labels[im_num][:, -1]))
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} gradients computation in {1:6.3f} seconds".format(image.split("/")[-1], (b_end - end)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", ", ".join(objs)))
        print("{0:20s} {1:s}".format("Objects marked:", ", ".join(gt)))
        print("{0:20s} {1:f}".format("Loss:", loss))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()

output_recast = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Average time_per_img", (output_recast - start_det_loop)/len(imlist)))
print("----------------------------------------------------------")

torch.cuda.empty_cache()
