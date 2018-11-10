import tensorflow as tf
import math
import numpy as np
def nms(heat):
    hmax=tf.nn.max_pool(heat,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME')
    mask=tf.cast(tf.equal(hmax,heat),tf.float32)
    return mask*heat
def top_k(heat,k=100):
    batch,h,w,c=heat.get_shape().as_list()
    heat=tf.reshape(heat,(batch,-1))
    k_value,k_index=tf.nn.top_k(heat,k)
    k_class=k_index//(h*w)
    k_position=k_index%(h*w)
    k_y=k_position//w#0 is also a cata
    k_x=k_position%w
    return k_value,k_position,k_class,k_y,k_x
def map_to_vector(feature_map,inds,transpose=True):
    #B*128*128*1
    #print(feature_map.get_shape().as_list())

    #assert tf.shape(inds)[1]==128
    def sub_map(value,select):
        value=tf.transpose(value,(1,0))
        sub_vector=tf.map_fn(fn=lambda x:tf.gather(x,select),elems=value,dtype=tf.float32)
        return tf.transpose(sub_vector,(1,0))
    if transpose:
        assert len(feature_map.get_shape().as_list())==4
        inter_vector=tf.reshape(feature_map,(feature_map.get_shape().as_list()[0],-1,feature_map.get_shape().as_list()[-1]))
    else:
        assert len(feature_map.get_shape().as_list())==3
        inter_vector=feature_map
    vector=tf.map_fn(fn=lambda p:sub_map(p[0],p[1]),elems=[inter_vector,inds],dtype=tf.float32)
    return vector
def expand_copy(feature_map,k,inter=False):
    feature_map=tf.expand_dims(feature_map,axis=-1)
    temp=feature_map
    for i in range(k-1):
        temp=tf.concat([temp,feature_map],-1)
    if inter:
        feature_map=tf.transpose(temp,(0,2,1))
    else:
        feature_map=temp
    assert feature_map.get_shape().as_list()[1]==feature_map.get_shape().as_list()[2]
    return feature_map

def rescale_dets(detections, ratios, borders, sizes):#may be problem
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs=xs/ ratios[:, 1][:, None, None]#change the shape
    ys=ys/ ratios[:, 0][:, None, None]
    xs=xs- borders[:, 2][:, None, None]
    ys=ys- borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)
    detections[..., 0:4:2], detections[..., 1:4:2]=xs, ys
    return detections
def soft_nms_merge(boxes,sigma=0.5, Nt=0.5, threshold=0.01, method=2, weight_exp=6):
    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0
    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]

        mx1 = boxes[i, 0] * boxes[i, 5]
        my1 = boxes[i, 1] * boxes[i, 5]
        mx2 = boxes[i, 2] * boxes[i, 6]
        my2 = boxes[i, 3] * boxes[i, 6]
        mts = boxes[i, 5]
        mbs = boxes[i, 6]

        # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    mw  = (1 - weight) ** weight_exp
                    mx1 = mx1 + boxes[pos, 0] * boxes[pos, 5] * mw
                    my1 = my1 + boxes[pos, 1] * boxes[pos, 5] * mw
                    mx2 = mx2 + boxes[pos, 2] * boxes[pos, 6] * mw
                    my2 = my2 + boxes[pos, 3] * boxes[pos, 6] * mw
                    mts = mts + boxes[pos, 5] * mw
                    mbs = mbs + boxes[pos, 6] * mw

                    boxes[pos, 4] = weight*boxes[pos, 4]

                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

        boxes[i, 0] = mx1 / mts
        boxes[i, 1] = my1 / mts
        boxes[i, 2] = mx2 / mbs
        boxes[i, 3] = my2 / mbs

    #keep = [i for i in range(N)]
    return boxes[0:N,:]