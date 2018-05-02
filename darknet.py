from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *



def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors



def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        #check the type of block
        #create a new module for the block
        #append to module_list

        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            #Check the activation.
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

            #If it's an upsampling layer
            #We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]

        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer

        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)


            elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])

                #Get the number of classes
                num_classes = int (module["classes"])

                #Transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:              #if no collector has been intialised.
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    def forward_dims(self, inp_dim):
        modules = self.blocks[1:]
        outputs = {-1: inp_dim}

        cur = inp_dim
        for i, module in enumerate(modules):
            module_type = module['type']
            if module_type == 'convolutional':
                if not bool(module['pad']):
                    cur -= int(module['size']) - 1
                cur /= int(module['stride'])
            elif module_type == 'upsample':
                cur *= int(module['stride'])
            elif module_type == 'route':
                layers = [int(a) for  a in module['layers']]
                if layers[0] < 0:
                    layers[0] += i
                cur = outputs[layers[0]]
            elif module_type == 'shortcut':
                # does not change size
                pass
            elif module_type == 'yolo':
                # does not change size
                pass
            outputs[i] = cur
        return outputs

    def loss_function(self, detections, y, CUDA, inp_dim):
        modules = self.blocks[1:]
        det_corners = det2corners(detections)

        # size of the tensor at each layer
        tensor_size = self.forward_dims(inp_dim)

        yolo_modules = [(i, x) for i, x in enumerate(modules) if x['type'] == 'yolo']
        num_yolo_layers = len(yolo_modules)
        # assume that all yolo modules share the same parameters values
        ignore_thresh = [float(x[1]['ignore_thresh']) for x in yolo_modules]
        assert(np.unique(ignore_thresh).size == 1) # yolo modules share parameters
        ignore_thresh = ignore_thresh[0]
        truth_thresh = [float(x[1]['truth_thresh']) for x in yolo_modules]
        assert(np.unique(ignore_thresh).size == 1) # yolo modules share parameters
        truth_thresh = truth_thresh[0]

        # load anchors
        anchors = [self.module_list[i][0].anchors for i, _ in yolo_modules]
        num_anchors = np.array([len(a) for a in anchors])
        anchors_cs = np.cumsum(num_anchors)
        anchors = np.array([(a[0], a[1]) for layer_a in anchors for a in layer_a])
        anchors = anchors[np.newaxis, :]

        OBJ_IDX = 4

        for i, img_y in enumerate(y):
            img_corners = det_corners[i]
            # convert boxes from relative to pixel coordinates
            img_gt = (img_y.view((-1, 2, 2)) * img_dim).view((-1, 4))
            # convert boxes to corners
            gt_corners = det2corners(img_gt.unsqeeze(0)).sqeeze(0)
            # compute IoU matrix. Rows correspond to detected corners,
            # columns correspond to groundtruth corners
            iou = bbox_iou(img_corners, gt_corners)
            det_iou, gt_idx = torch.maximum(iou, dim=1)
            # move all objectness score toward zero
            delta_obj = -detections[:, OBJ_IDX]
            # do not penalize objectness score for detections with huge IoU value
            mask = det_iou > ignore_thresh
            delta_obj[mask] = 0
            # move objectness score to 1 for detections with the huge IoU value
            mask = det_iou > truth_tresh
            delta_obj[mask] = 1 - detections[:, OBJ_IDX]

            # find a cell that has best match with the truth box
            tmp_gt = img_gt[:, new_axis, 2:4]
            inter = np.prod(np.minimum(anchors, img_gt), axis=2)
            union = np.prod(tmp_gt, axis=2) + np.prod(anchors, axis=2) - inter
            iou = inter / union
            anchor_idx = np.argmax(iou, axis=1)
            for j, img_gt in zip(anchor_idx, img_gt):
                # index of the yolo layer with the best anchor
                layer_idx = np.searchsorted(num_anchors, j, side='right')
                # index of the module with the best anchor
                module_idx = yolo_modules[layer_idx][0]
                # get index of the cell corresponding to the truth box
                ts = tensor_size[module_idx]
                stride = inp_dim / ts
                cell_x = img_gt[0] // stride
                cell_y = img_gt[1] // stride
                # index of the bounding box in detections
                box_idx = 0
                for k in range(module_idx):
                    ts_k = tensor_size[yolo_modules[k][0]]
                    box_idx += num_anchors[k] * ts_k * ts_k
                box_idx += (ts * cell_x + cell_y) * num_anchors[layer_idx] + num_anchors[layer_idx] + j - anchors[cs]
                # compute loss corresponding to the box

        loss = None
        return loss

    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype = np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            #If module_type is convolutional load weights
            #Otherwise ignore.

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]


                if (batch_normalize):
                    bn = model[1]

                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    #Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    #Number of biases
                    num_biases = conv.bias.numel()

                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


