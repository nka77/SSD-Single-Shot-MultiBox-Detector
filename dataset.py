import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# .clamp_(max=1, min=0)

#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    # TODO handle out of image index

    boxes = []
    # tmp_img = np.ones((100,100))
    for i, out_layer_size in enumerate(layers):
        ssize = small_scale[i]
        lsize = large_scale[i]
        # print("out_layer_size:{}".format(out_layer_size))

        for i_height in range(out_layer_size):
            for i_width in range(out_layer_size):
                x_center, y_center = (i_width+0.5)/out_layer_size, (i_height+0.5)/out_layer_size 
                sizes = [[ssize,ssize],
                        [lsize,lsize],
                        [np.round(lsize*np.sqrt(2),2),np.round(lsize/np.sqrt(2),2)],
                        [np.round(lsize/np.sqrt(2),2),np.round(lsize*np.sqrt(2),2)]]
                for size in sizes:
                    box_width, box_height = size[0], size[1]
                    width, height = box_width/2, box_height/2
                    x_min, y_min = x_center-width, y_center-height
                    x_max, y_max = x_center+width, y_center+height
                    box = np.array([x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max])
                    box = np.clip(box, 0., 1.)
                    boxes.append(box)
                    # tmp_img = cv2.rectangle(tmp_img, (int(x_min*100), int(y_min*100)), (int(x_max*100), int(y_max*100)), (255,0,0), 1)

    # plt.imshow(tmp_img)
    # plt.savefig("default_boxes.png")
    # print("Total boxes", len(boxes))
    return np.array(boxes)

# default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])

    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    

#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxs_default -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min, y_min, x_max, y_max)
    
    if ious.max() < threshold:
        ious_true = (ious == ious.max())
    else:
        ious_true = ious>threshold

    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence

    gx, gy = x_min + (x_max - x_min)/2, y_min + (y_max - y_min)/2
    gw, gh = (x_max - x_min), (y_max - y_min)

    for idx, is_true in enumerate(ious_true):
        if is_true:
            px, py, pw, ph = boxs_default[idx][:4]
            tx = (gx - px)/pw
            ty = (gy - py)/ph
            tw = np.log(gw/pw)
            th = np.log(gh/ph)
            ann_box[idx, :] = tx, ty, tw, th
            ann_confidence[idx, -1] = 0
            ann_confidence[idx, cat_id] = 1
            print("box default", px, py, pw, ph)
            print("compute IOU", idx, gx, gy, gw, gh, x_min, y_min, x_max, y_max)


class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        

        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        # self.img_names[index] = "01109.jpg"
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        image = cv2.imread(img_name)
        # print(img_name)

        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        with open(ann_name) as f:
            ann_data = f.readline()
            while ((ann_data != None) and (len(ann_data) > 2)):
                print(ann_data)
                ann_data = ann_data.split(' ')
                class_id =  int(ann_data[0])
                ann_data = [float(x) for x in ann_data[1:5]]
                x_min, y_min, width, height = ann_data[0], ann_data[1], ann_data[2], ann_data[3]
                x_max, y_max = x_min + width, y_min + height
                im_h,im_w, _ = image.shape
                
                x_max /= im_w
                x_min /= im_w
                y_max /= im_h
                y_min /= im_h
                print("xmin xmax", x_min, y_min, x_max, y_max)

                #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
                # [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
                match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
                ann_data = f.readline()

        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
            # – Use the entire original input image.
            # – Sample a patch so that the minimum jaccard overlap with the objects is 0.1, 0.3,
            # 0.5, 0.7, or 0.9.
            # – Randomly sample a patch.
         
        image = cv2.resize(image, (320, 320))
        image = np.transpose(image, (2, 0, 1))
        return image, ann_box, ann_confidence


# class_num = 4
# batch_size = 1

# boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])
# dataset = COCO("CMPT733-Lab3-Workspace/data/train/images/", "CMPT733-Lab3-Workspace/data/train/annotations/", class_num, boxs_default, train = True, image_size=320)
# dataset_test = COCO("CMPT733-Lab3-Workspace/data/train/images/", "CMPT733-Lab3-Workspace/data/train/annotations/", class_num, boxs_default, train = False, image_size=320)

# # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

# for i, data in enumerate(dataloader_test, 0):
#         images_, ann_box_, ann_confidence_ = data
#         images = images_
#         ann_box = ann_box_
#         ann_confidence = ann_confidence_