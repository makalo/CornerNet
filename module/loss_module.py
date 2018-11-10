import numpy as np
import tensorflow as tf
def focal_loss(preds,gt):
    print(gt.get_shape().as_list())
    zeros=tf.zeros_like(gt)
    ones=tf.ones_like(gt)
    num_pos=tf.reduce_sum(tf.where(tf.equal(gt,1),ones,zeros))
    loss=0
    #loss=tf.reduce_mean(tf.log(preds))
    for pre in preds:
        pos_weight=tf.where(tf.equal(gt,1),ones-pre,zeros)
        neg_weight=tf.where(tf.less(gt,1),pre,zeros)
        pos_loss=tf.reduce_sum(tf.log(pre) * tf.pow(pos_weight,2))
        neg_loss=tf.reduce_sum(tf.pow((1-gt),4)*tf.pow(neg_weight,2)*tf.log((1-pre)))
        loss=loss-(pos_loss+neg_loss)/(num_pos+tf.convert_to_tensor(1e-4))
    return loss
def tag_loss(tag0, tag1, mask):
    #pull
    print(tag0.get_shape().as_list())
    tag0=tf.squeeze(tag0,axis=-1)
    tag1=tf.squeeze(tag1,axis=-1)
    zeros=tf.zeros_like(mask)
    ones=tf.ones_like(mask)
    num  = tf.reduce_sum(mask)


    tag_mean = (tag0 + tag1) / 2

    tag0 = tf.pow((tag0 - tag_mean) , 2) / (num + tf.convert_to_tensor(1e-4))
    #tag0_mask=tf.where(tf.equal(mask,1),ones,zeros)
    tag0 = tf.reduce_sum(tag0*mask)

    tag1 = tf.pow((tag1 - tag_mean), 2) / (num + tf.convert_to_tensor(1e-4))
    #tag1_mask=tf.where(tf.equal(mask,1),ones,zeros)
    tag1 = tf.reduce_sum(tag1*mask)
    pull = tag0 + tag1
    #push
    dist_mask=tf.reshape(mask,(tf.shape(mask)[0],1,tf.shape(mask)[1]))+tf.reshape(mask,(tf.shape(mask)[0],tf.shape(mask)[1],1))
    dist_zeros=tf.zeros_like(dist_mask)
    dist_ones=tf.ones_like(dist_mask)
    dist_mask=tf.where(tf.equal(dist_mask,2),dist_ones,dist_zeros)
    num2=num*(num-1)
    dist=tf.reshape(tag_mean,(tf.shape(tag_mean)[0],1,tf.shape(tag_mean)[1]))-tf.reshape(tag_mean,(tf.shape(tag_mean)[0],tf.shape(tag_mean)[1],1))
    #dist=-tf.pow(dist,2)
    dist=1-tf.abs(dist)
    dist=tf.nn.relu(dist)
    dist=dist-1 / (num + tf.convert_to_tensor(1e-4))
    dist=dist / (num2 + tf.convert_to_tensor(1e-4))
    dist=tf.multiply(dist_mask,dist)
    push=tf.reduce_sum(dist)
    return pull, push

def offset_loss(offset, gt_offset, mask):
    num  = tf.reduce_sum(mask)
    mask = tf.stack((mask,mask),-1)
    offset_loss = smooth_l1_loss(offset, gt_offset)
    offset_loss = offset_loss / (num + tf.convert_to_tensor(1e-4))
    offset_loss=tf.reduce_sum(tf.multiply(offset_loss,mask))
    return offset_loss
def smooth_l1_loss(pred,targets,sigma=1):
    # diff = pred -targets
    # abs_diff = tf.abs(diff)
    # smoothL1_sign =tf.to_float(tf.less(abs_diff, 1))
    # loss = tf.pow(diff, 2) * 0.5 * smoothL1_sign + (abs_diff - 0.5) * (1. - smoothL1_sign)
    # return loss
    sigma2 = sigma * sigma

    diff = tf.subtract(pred, targets)

    smooth_l1_sign = tf.cast(tf.less(tf.abs(diff), 1.0 / sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(tf.multiply(diff, diff), 0.5 * sigma2)
    smooth_l1_option2 = tf.subtract(tf.abs(diff), 0.5 / sigma2)
    smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))
    return smooth_l1_result



