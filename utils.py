import numpy as np
import cv2
from dataset import iou


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    class_num = class_num-1
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO: when ann conf was max and not > threshold??
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                
                px, py, pw, ph = boxs_default[i][:4]
                tx, ty, tw, th = ann_box[i]
                gx = (tx*pw + px)*image.shape[0]
                gy = (ty*ph + py)*image.shape[1]
                gw = (pw*np.exp(tw))*image.shape[0]
                gh = (ph*np.exp(th))*image.shape[1]
                print("idx in visualize", i, px, py, pw, ph)
                x1, y1 = int(gx - gw/2), int(gy - gh/2)
                x2, y2 = int(gx + gw/2), int(gy + gh/2)
                print(gx, gy, gh, gw, x1, y1, x2, y2)

                start_point = (x1, y1) 
                end_point = (x2, y2) 
                image1 = cv2.rectangle(image1, start_point, end_point, color=colors[j], thickness=2)

                px, py, pw, ph = boxs_default[i][:4]
                x1, y1 = int((px - pw/2)*image.shape[0]), int((py - ph/2)*image.shape[1])
                x2, y2 = int((px + pw/2)*image.shape[0]), int((py + ph/2)*image.shape[1])
                start_point = (x1, y1) 
                end_point = (x2, y2) 
                image2 = cv2.rectangle(image2, start_point, end_point, color=colors[j], thickness=2)

    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #image3: draw network-predicted bounding boxes on image3
                px, py, pw, ph = pred_box[i][:4]
                x1, y1 = int((px - pw/2)*image.shape[0]), int((py - ph/2)*image.shape[1])
                x2, y2 = int((px + pw/2)*image.shape[0]), int((py + ph/2)*image.shape[1])
                start_point = (x1, y1) 
                end_point = (x2, y2) 
                image3 = cv2.rectangle(image3, start_point, end_point, color=colors[j], thickness=2)
                
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                px, py, pw, ph = boxs_default[i][:4]
                x1, y1 = int((px - pw/2)*image.shape[0]), int((py - ph/2)*image.shape[1])
                x2, y2 = int((px + pw/2)*image.shape[0]), int((py + ph/2)*image.shape[1])
                start_point = (x1, y1) 
                end_point = (x2, y2) 
                image4 = cv2.rectangle(image4, start_point, end_point, color=colors[j], thickness=2)

    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    cv2.imwrite("visualize_pred.png", image)



def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.5, threshold=0.5):
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    
    all_boxes = box_
    non_overlap_boxes = []

    while len(all_boxes) > 0:
        max_conf_idx = np.argmax(confidence_[])
    
    #TODO: non maximum suppression
    return