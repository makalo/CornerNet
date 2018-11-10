import tensorflow as tf
import numpy as np
import cv2
import os
from net.network import NetWork
from utils.init_data import MSCOCO
from tqdm import tqdm
from config import cfg
from utils.transform import crop_image
from module.forward_module import rescale_dets,soft_nms_merge
class Test():
    def __init__(self):
        self.coco=MSCOCO('minival')
        self.net=NetWork()
        self.top_k=cfg.top_k
        self.ae_threshold=cfg.ae_threshold
        self.test_scales=cfg.test_scales
        self.weight_exp=cfg.weight_exp
        self.merge_bbox=cfg.merge_bbox
        self.categories=cfg.categories
        self.nms_threshold=cfg.nms_threshold
        self.max_per_image=cfg.max_per_image
        self.result_dir=cfg.result_dir
    def test(self,sess):
        debug_dir = os.path.join(result_dir, "debug")
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        img_names=self.coco.get_all_img()
        num=len(img_names)
        for img_name in tqdm(img_names):
            img=self.coco.read_img(img_name)
            height, width = img.shape[0:2]
            detections=[]
            for scale in test_scales:
                new_height = int(height * scale)
                new_width  = int(width * scale)
                new_center = np.array([new_height // 2, new_width // 2])

                inp_height = new_height | 127
                inp_width  = new_width  | 127

                images  = np.zeros((1, inp_height, inp_width, 3), dtype=np.float32)
                ratios  = np.zeros((1, 2), dtype=np.float32)
                borders = np.zeros((1, 4), dtype=np.float32)
                sizes   = np.zeros((1, 2), dtype=np.float32)

                out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
                height_ratio = out_height / inp_height
                width_ratio  = out_width  / inp_width

                resized_image = cv2.resize(image, (new_width, new_height))
                resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])

                resized_image = resized_image / 255.
                #normalize_(resized_image, db.mean, db.std)

                images[0]  = resized_image
                borders[0] = border
                sizes[0]   = [int(height * scale), int(width * scale)]
                ratios[0]  = [height_ratio, width_ratio]

                images = np.concatenate((images, images[:, :, ::-1, :]), axis=0)
                images = tf.convert_to_tensor(images)
                is_training=tf.convert_to_tensor(False)
                outs=self.net.corner_net(images,is_training=is_training)
                dets_tensor=self.net.decode(*outs[-6:])
                dets=sess.run(dets_tensor)

                dets   = dets.reshape(2, -1, 8)
                dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
                dets   = dets.reshape(1, -1, 8)

                dets=rescale_dets(dets, ratios, borders, sizes)
                dets[:, :, 0:4] /= scale
                detections.append(dets)

            detections = np.concatenate(detections, axis=1)
            classes    = detections[..., -1]
            classes    = classes[0]
            detections = detections[0]

            # reject detections with negative scores
            keep_inds  = (detections[:, 4] > -1)
            detections = detections[keep_inds]
            classes    = classes[keep_inds]

            top_bboxes[image_id] = {}
            for j in range(categories):
                keep_inds = (classes == j)
                top_bboxes[image_id][j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
                if merge_bbox:
                    top_bboxes[image_id][j + 1]=soft_nms_merge(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=2, weight_exp=weight_exp)
                else:
                    top_bboxes[image_id][j + 1]=soft_nms(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm)
                top_bboxes[image_id][j + 1] = top_bboxes[image_id][j + 1][:, 0:5]

            scores = np.hstack([
                top_bboxes[image_id][j][:, -1]
                for j in range(1, categories + 1)
            ])
            if len(scores) > max_per_image:
                kth    = len(scores) - max_per_image
                thresh = np.partition(scores, kth)[kth]
                for j in range(1, categories + 1):
                    keep_inds = (top_bboxes[image_id][j][:, -1] >= thresh)
                    top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]

            if debug:
                image=self.coco.read_img(img_name)

                bboxes = {}
                for j in range(1, categories + 1):
                    keep_inds = (top_bboxes[image_id][j][:, -1] > 0.5)
                    cat_name  = self.coco.class_name(j)
                    cat_size  = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    color     = np.random.random((3, )) * 0.6 + 0.4
                    color     = color * 255
                    color     = color.astype(np.int32).tolist()
                    for bbox in top_bboxes[image_id][j][keep_inds]:
                        bbox  = bbox[0:4].astype(np.int32)
                        if bbox[1] - cat_size[1] - 2 < 0:
                            cv2.rectangle(image,
                                (bbox[0], bbox[1] + 2),
                                (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                                color, -1
                            )
                            cv2.putText(image, cat_name,
                                (bbox[0], bbox[1] + cat_size[1] + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
                            )
                        else:
                            cv2.rectangle(image,
                                (bbox[0], bbox[1] - cat_size[1] - 2),
                                (bbox[0] + cat_size[0], bbox[1] - 2),
                                color, -1
                            )
                            cv2.putText(image, cat_name,
                                (bbox[0], bbox[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
                            )
                        cv2.rectangle(image,
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            color, 2
                        )
                debug_file = os.path.join(debug_dir, {}.format(img_name))

        # result_json = os.path.join(result_dir, "results.json")
        # detections  = db.convert_to_coco(top_bboxes)
        # with open(result_json, "w") as f:
        #     json.dump(detections, f)

        # cls_ids   = list(range(1, categories + 1))
        # image_ids = [db.image_ids(ind) for ind in db_inds]
        # db.evaluate(result_json, cls_ids, image_ids)
        return 0
class Debug():
    def __init__(self):
        self.top_k=cfg.top_k
        self.ae_threshold=cfg.ae_threshold
        self.test_scales=cfg.test_scales
        self.weight_exp=cfg.weight_exp
        self.merge_bbox=cfg.merge_bbox
        self.categories=cfg.categories
        self.nms_threshold=cfg.nms_threshold
        self.max_per_image=cfg.max_per_image
        self.debug_dir=cfg.debug_dir
    def test_debug(self,image,detections,debug_boxes,boxes,ratio,coco,step):
        detections   = detections.reshape(-1, 8)
        detections[:, 0:4:2] /= ratio[0]
        detections[:, 1:4:2] /= ratio[1]
        debug_boxes=debug_boxes.reshape(-1,4)
        debug_boxes[:,0:4:2] /= ratio[0]
        debug_boxes[:,1:4:2] /= ratio[1]

        classes    = detections[..., -1].astype(np.int64)

        # reject detections with negative scores
        keep_inds  = (detections[:, 4] > -1)
        detections = detections[keep_inds]
        classes    = classes[keep_inds]

        top_bboxes = {}
        for j in range(self.categories):
            keep_inds = (classes == j)
            top_bboxes[j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
            if self.merge_bbox:
                top_bboxes[j + 1]=soft_nms_merge(top_bboxes[j + 1], Nt=0.5, method=2, weight_exp=8)
            else:
                top_bboxes[j + 1]=soft_nms(top_bboxes[j + 1], Nt=0.5, method=2)
            top_bboxes[j + 1] = top_bboxes[j + 1][:, 0:5]

        scores = np.hstack([
            top_bboxes[j][:, -1]
            for j in range(1, self.categories + 1)
        ])
        if len(scores) > self.max_per_image:
            kth    = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.categories + 1):
                keep_inds = (top_bboxes[j][:, -1] >= thresh)
                top_bboxes[j] = top_bboxes[j][keep_inds]
                # if len(top_bboxes[j])!=0:
                #     print(top_bboxes[j].shape)


        image=(image*255).astype(np.uint8)

        bboxes = {}
        for j in range(1, self.categories + 1):
            #if step>10000:
            keep_inds = (top_bboxes[j][:, -1] > 0.5)
            top_bboxes[j]=top_bboxes[j][keep_inds]
            cat_name  = coco.class_name(j)
            cat_size  = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            color     = np.random.random((3, )) * 0.6 + 0.4
            color     = color * 255
            color     = color.astype(np.int32).tolist()
            for bbox in top_bboxes[j]:
                bbox  = bbox[0:4].astype(np.int32)
                if bbox[1] - cat_size[1] - 2 < 0:
                    cv2.rectangle(image,
                        (bbox[0], bbox[1] + 2),
                        (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                        color, -1
                    )
                    cv2.putText(image, cat_name,
                        (bbox[0], bbox[1] + cat_size[1] + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
                    )
                else:
                    cv2.rectangle(image,
                        (bbox[0], bbox[1] - cat_size[1] - 2),
                        (bbox[0] + cat_size[0], bbox[1] - 2),
                        color, -1
                    )
                    cv2.putText(image, cat_name,
                        (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
                    )
                cv2.rectangle(image,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    color, 2
                )
            for b in boxes:
                cv2.rectangle(image ,(b[0],b[1]),(b[2],b[3]),(0,0,255),1)
        for i in range(len(debug_boxes)):
            color     = np.random.random((3, )) * 0.6 + 0.4
            color     = color * 255
            color     = color.astype(np.int32).tolist()
            cv2.circle(image,(debug_boxes[i][0],debug_boxes[i][1]),2,color,2)
            cv2.circle(image,(debug_boxes[i][2],debug_boxes[i][3]),2,color,2)
        cv2.imwrite(os.path.join(self.debug_dir,str(step)+'.jpg'),image)


