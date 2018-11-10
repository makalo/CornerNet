import sys
sys.path.append('./utils')
import cv2
import math
import numpy as np
import random
import string
import tensorflow as tf
from config import cfg
from transform import random_crop, draw_gaussian, gaussian_radius,crop_image,crop_image, normalize_, color_jittering_, lighting_,full_image_crop,resize_image,clip_detections
from init_data import MSCOCO
class Image_data():
    def __init__(self,split):
        self.coco=MSCOCO(split)
        self.data_rng   = cfg.data_rng
        self.num_image  = len(self.coco.get_all_img())
        self.categories   = cfg.categories
        self.input_size   = cfg.input_size
        self.output_size  = cfg.output_sizes[0]

        self.border        = cfg.border
        #self.lighting      = cfg.lighting
        self.rand_crop     = cfg.rand_crop
        print(self.rand_crop)
        self.rand_color    = cfg.rand_color
        self.rand_scales   = cfg.rand_scales
        self.gaussian_bump = cfg.gaussian_bump
        self.gaussian_iou  = cfg.gaussian_iou
        self.gaussian_rad  = cfg.gaussian_radius
    def read_from_disk(self,queue):
        # allocating memory
        max_tag_len = 128
        image       = np.zeros((self.input_size[0], self.input_size[1],3), dtype=np.float32)
        heatmaps_tl = np.zeros((self.output_size[0], self.output_size[1],self.categories), dtype=np.float32)
        heatmaps_br = np.zeros((self.output_size[0], self.output_size[1],self.categories), dtype=np.float32)
        offsets_tl    = np.zeros((max_tag_len, 2), dtype=np.float32)
        offsets_br    = np.zeros((max_tag_len, 2), dtype=np.float32)
        tags_tl     = np.zeros((max_tag_len), dtype=np.int64)
        tags_br     = np.zeros((max_tag_len), dtype=np.int64)
        tags_mask   = np.zeros((max_tag_len), dtype=np.float32)
        boxes       = np.zeros((max_tag_len,4), dtype=np.int64)
        ratio       = np.ones((max_tag_len,2), dtype=np.float32)
        tag_lens    = 0

        # reading image
        image=self.coco.read_img(queue[0])

        # reading detections
        detections = self.coco.detections(queue[0])

        # cropping an image randomly
        if self.rand_crop:
            image, detections = random_crop(image, detections, self.rand_scales, self.input_size, border=self.border)
        else:
            image, detections = full_image_crop(image, detections)

        image, detections = resize_image(image, detections, self.input_size)
        detections = clip_detections(image, detections)

        width_ratio  = self.output_size[1] / self.input_size[1]
        height_ratio = self.output_size[0] / self.input_size[0]

        # flipping an image randomly
        if np.random.uniform() > 0.5:
            image[:] = image[:, ::-1, :]
            width    = image.shape[1]
            detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1


        image = image.astype(np.float32) / 255.
        # if rand_color:
        #     color_jittering_(data_rng, image)
        #     if lighting:
        #         lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)

        #normalize_(image, self.coco.mean, self.coco.std)

        for ind, detection in enumerate(detections):
            category = int(detection[-1]) - 1

            xtl_ori, ytl_ori = detection[0], detection[1]
            xbr_ori, ybr_ori = detection[2], detection[3]

            fxtl = (xtl_ori * width_ratio)
            fytl = (ytl_ori * height_ratio)
            fxbr = (xbr_ori * width_ratio)
            fybr = (ybr_ori * height_ratio)


            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)
            ybr = int(fybr)


            if self.gaussian_bump:
                width  = detection[2] - detection[0]
                height = detection[3] - detection[1]

                width  = math.ceil(width * width_ratio)
                height = math.ceil(height * height_ratio)

                if self.gaussian_rad == -1:
                    radius = gaussian_radius((height, width), self.gaussian_iou)
                    radius = max(0, int(radius))
                else:
                    radius = self.gaussian_rad

                draw_gaussian(heatmaps_tl[:,:,category], [xtl, ytl], radius)
                draw_gaussian(heatmaps_br[:,:,category], [xbr, ybr], radius)
            else:
                heatmaps_tl[ytl, xtl, category] = 1
                heatmaps_br[ybr, xbr, category] = 1

            tag_ind = tag_lens
            offsets_tl[tag_ind, :] = [fxtl - xtl, fytl - ytl]
            offsets_br[tag_ind, :] = [fxbr - xbr, fybr - ybr]
            tags_tl[tag_ind] = ytl * self.output_size[1] + xtl
            tags_br[tag_ind] = ybr * self.output_size[1] + xbr
            boxes[tag_ind] = [xtl_ori,ytl_ori,xbr_ori,ybr_ori]
            ratio[tag_ind] = [width_ratio,height_ratio]
            tag_lens += 1
        tags_mask[:tag_lens] = 1
        return image, tags_tl, tags_br,heatmaps_tl, heatmaps_br, tags_mask, offsets_tl, offsets_br,boxes,ratio

    def get_single_data(self,queue):
        images, tags_tl, tags_br,heatmaps_tl, heatmaps_br, tags_mask, offsets_tl, offsets_br,boxes,ratio=tf.py_func(self.read_from_disk,[queue],
            [tf.float32,tf.int64,tf.int64,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.int64,tf.float32])
        return images, tags_tl, tags_br,heatmaps_tl, heatmaps_br, tags_mask, offsets_tl, offsets_br,boxes,ratio
    def inupt_producer(self):
        quene_train=tf.train.slice_input_producer([self.coco.get_all_img()],shuffle=True)
        self.images, self.tags_tl, self.tags_br,self.heatmaps_tl, self.heatmaps_br, self.tags_mask, self.offsets_tl, self.offsets_br,self.boxes,self.ratio=self.get_single_data(quene_train)

    def get_batch_data(self,batch_size):
        images, tags_tl, tags_br,heatmaps_tl, heatmaps_br, tags_mask, offsets_tl, offsets_br,boxes,ratio=tf.train.shuffle_batch([self.images,
            self.tags_tl, self.tags_br,self.heatmaps_tl, self.heatmaps_br, self.tags_mask, self.offsets_tl, self.offsets_br,self.boxes,self.ratio],
            batch_size=batch_size,shapes=[(self.input_size[0], self.input_size[1],3),(128),(128),
            (self.output_size[0], self.output_size[1],self.categories),(self.output_size[0], self.output_size[1],self.categories),
            (128),(128,2),(128,2),(128,4),(128,2)],capacity=100,min_after_dequeue=batch_size,num_threads=16)
        return images, tags_tl, tags_br,heatmaps_tl, heatmaps_br, tags_mask, offsets_tl, offsets_br,boxes,ratio

if __name__=='__main__':
    data=Image_data('trainval')
    data.inupt_producer()
    images, tags_tl, tags_br,heatmaps_tl, heatmaps_br, tags_mask, offsets_tl, offsets_br,boxes,ratio=data.get_batch_data(2)
    sess=tf.InteractiveSession()
    coord = tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(12):
        images_, tags_tl_, tags_br_,heatmaps_tl_, heatmaps_br_, tags_mask_, offsets_tl_, offsets_br_,boxes_= sess.run([images, tags_tl, tags_br,heatmaps_tl, heatmaps_br, tags_mask, offsets_tl, offsets_br,boxes])
        for j in range(2):
            #print(images_.shape,tags_tl_.shape,heatmaps_tl_.shape,tags_mask_.shape,offsets_tl_.shape)
            img=(images_[j]*255).astype(np.uint8)
            heat_1=np.max(heatmaps_tl_[j],axis=-1)
            heat_cat=np.zeros_like(heat_1)
            heat_cat[np.where(heat_1==1)]=1
            heat_arg=np.argmax(heat_cat,-1)
            print(heat_arg)
            # print('kkkkkkk')
            # print(np.where(heat_1==1))
            # print(np.where(heat_1>1))
            # print('ppppp')
            heat_1=heat_1*255
            heat_1=heat_1.astype(np.uint8)
            heat_1=np.stack([heat_1,heat_1,heat_1],-1)

            heat_2=np.max(heatmaps_br_[j],axis=-1)*255
            heat_2=heat_2.astype(np.uint8)
            heat_2=np.stack([heat_2,heat_2,heat_2],-1)

            heat=heat_1+heat_2
            heat=cv2.resize(heat,(511,511))
            norm=cv2.addWeighted(img,0.5,heat,0.5,0)
            box=boxes_[j]
            for b in box:
                cv2.rectangle(norm ,(b[0],b[1]),(b[2],b[3]),(225,225,0),1)
            cv2.imshow('img',norm)
            cv2.waitKey(0)
    coord.request_stop()
    coord.join(threads)
    sess.close()