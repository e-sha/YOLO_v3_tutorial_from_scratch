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
                        default = "imgs", type = str)
    parser.add_argument("--labels", help="Directory containing groundtruth",
                        default="labels")
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

    return parser.parse_args()

args = arg_parse()
images = osp.realpath(args.images)
labels = osp.realpath(args.labels)
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()
dev = 'gpu' if CUDA else 'cpu'

num_classes = 80
classes = load_classes("data/coco.names")

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
model.eval()

read_dir = time.time()
#Detection phase

try:
    imlist = [osp.join(images, img) for img in os.listdir(images)]
except FileNotFoundError:
    print ("No directory with the name {}".format(images))
    exit()

data = [(x, osp.join(labels, osp.splitext(osp.basename(x))[0] + '.txt')) for x in imlist]
for im_name, label_name in data:
    assert(isfile(im_name), "No file with name {}".format(im_name))
    assert(isfile(label_name), "No file with name {}".format(label_name))

train_data = data

if not os.path.exists(args.det):
    os.makedirs(args.det)

num_train_batches = (len(train_data) - 1) // batch_size + 1
train_batches = [train_data[i*batch_size:(i+1)*batch_size] for i in range(num_train_batches)]

output = []
start_det_loop = time.time()
for i, batch in enumerate(train_batches):
    start = time.time()

    # load the image
    im_names, label_names = batch
    batch_imgs = [cv2.imread(x) for x in im_names]
    tr_imgs = list(map(prep_image, batch_imgs, [inp_dim for x in range(len(im_names))]))
    tr_imgs = torch.cat(tr_imgs)
    # load the labels
    batch_labels = list(map(lambda x: np.fromfile(x, sep=' ').shape(-1, 5), label_names))
    # transform

    if CUDA:
        batch = batch.cuda()
    model.zero_grad()
    prediction_all = model(Variable(tr_imgs))

    prediction = write_results(prediction_all, confidence, num_classes, nms_conf = nms_thesh)
    if isinstance(prediction, int):
        gt_pred = np.zeros((0, 5))
        box2img = np.zeros(0)
    else:
        gt_pred = prediction[:, [1, 2, 3, 4, 7]]
        box2img = prediction[:, 0]
    # convert gt corners to box in (center, size) format
    gt_pred[:, :2] = (gt_pred[:, :2] + gt_pred[:, 2:4]) / 2
    gt_pred[:, 2:4] = 2 * (gt_pred[:, 2:4] - gt_pred[:, :2])
    y = [gt_pred[box2img == i] / inp_dim for i in range(batch.size()[0])]
    loss = model.loss_function(prediction_all, y, inp_dim, loaded_ims[i])

    #graph = make_dot(loss, params=dict(model.named_parameters()))
    #graph.render('/tmp/graph')
    end = time.time()
    loss.backward()
    b_end = time.time()

    if type(prediction) == int:
        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} gradients computation in {1:6.3f} seconds".format(image.split("/")[-1], (b_end - end)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("{0:20s} {1:f}".format("Loss:", loss))
            print("----------------------------------------------------------")
        continue

    prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist

    output.append(prediction)

    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output[-1] if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} gradients computation in {1:6.3f} seconds".format(image.split("/")[-1], (b_end - end)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("{0:20s} {1:f}".format("Loss:", loss))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()

if len(output) == 0:
    print ("No detections were made")
    exit()

output = torch.cat(output)

im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)

output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2

output[:,1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

draw = time.time()

def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

list(map(lambda x: write(x, loaded_ims), output))

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))

list(map(cv2.imwrite, det_names, loaded_ims))

end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")

torch.cuda.empty_cache()

