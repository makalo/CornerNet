import tensorflow as tf
import os
from utils.get_data import Image_data
from net.network import NetWork
from config import cfg
import numpy as np
import cv2
import time
from test import Debug
class Train():
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES']='0,1'
        self.net=NetWork()
        self.data_train=Image_data('trainval')
        self.data_train.inupt_producer()
        # self.data_test=Image_data('minival')
        # self.data_test.inupt_producer()
        self.gpus=[0,1]
        self.batch_i=cfg.batch_size
        self.batch_size=self.batch_i*len(self.gpus)
        self.save_pre_every=int(self.data_train.num_image/self.batch_size)+1
        self.num_steps=int(self.save_pre_every*cfg.epoch_num+1)
        self.lr=cfg.learning_rate
        self.snapshot_dir=cfg.snapshot_dir
        self.snapshot_file=cfg.snapshot_file
        self.decay_rate=cfg.decay_rate
        self.decay_step=cfg.decay_step

    def train_mult(self):
        coord = tf.train.Coordinator()
        images, tags_tl, tags_br,heatmaps_tl, heatmaps_br, tags_mask, offsets_tl, offsets_br,boxes,ratio=self.data_train.get_batch_data(self.batch_size)
        tower_grads = []
        steps=tf.Variable(0,name='global_step',trainable=False)
        lr=tf.train.exponential_decay(self.lr,steps,self.decay_step,self.decay_rate,staircase= True, name= 'learning_rate')
        optim=tf.train.AdamOptimizer(learning_rate=lr)
        #optim= tf.train.MomentumOptimizer(0.000025,0.9)
        reuse1 = False
        for i in range(len(self.gpus)):
            with tf.device('/gpu:%d'%i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    if i == 0:
                        reuse1 = False
                    else:
                        reuse1 = True

                    next_imgs=images[i*self.batch_i:(i+1)*self.batch_i]
                    next_tags_tl=tags_tl[i*self.batch_i:(i+1)*self.batch_i]
                    next_tags_br=tags_br[i*self.batch_i:(i+1)*self.batch_i]
                    next_heatmaps_tl=heatmaps_tl[i*self.batch_i:(i+1)*self.batch_i]
                    next_heatmaps_br=heatmaps_br[i*self.batch_i:(i+1)*self.batch_i]
                    next_tags_mask=tags_mask[i*self.batch_i:(i+1)*self.batch_i]
                    next_offsets_tl=offsets_tl[i*self.batch_i:(i+1)*self.batch_i]
                    next_offsets_br=offsets_br[i*self.batch_i:(i+1)*self.batch_i]
                    with tf.variable_scope('', reuse=reuse1):
                        outs,test_outs=self.net.corner_net(next_imgs,next_tags_tl,next_tags_br,is_training=True)
                    dets_tensor,debug_boxes=self.net.decode(*test_outs)

                    loss,focal_loss,pull_loss,push_loss,offset_loss=self.net.loss(outs,[next_heatmaps_tl,next_heatmaps_br,next_tags_mask,next_offsets_tl,next_offsets_br])
                    trainable_variable = tf.trainable_variables()
                    grads = optim.compute_gradients(loss, var_list=trainable_variable)


                    tower_grads.append(grads)

        grads_ave = self.average_gradients(tower_grads)
        update=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update):
            train_op = optim.apply_gradients(grads_ave,steps)

        saver = tf.train.Saver(max_to_keep=100)
        #loader = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        print(self.num_steps)
        debug=Debug()
        epoch=0
        if self.load(saver, sess, self.snapshot_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        for step in range(self.num_steps):
            start=time.time()
            #sess.run(update)
            _,loss_,focal_loss_,pull_loss_,push_loss_,offset_loss_,lr_=sess.run([train_op,loss,focal_loss,pull_loss,push_loss,offset_loss,lr])
            duration=time.time()-start

            print('step %d, loss %g, focal_loss %g, pull_loss %g, push_loss %g, offset_loss %g, time %g, lr %g'
                %(step,loss_,focal_loss_,pull_loss_,push_loss_,offset_loss_,duration,lr_))

            if step%100==0:
                dets_,images_,debug_boxes_,boxes_,ratio_=sess.run([dets_tensor,images,debug_boxes,boxes,ratio])
                debug.test_debug(images_[0],dets_[0],debug_boxes_[0],boxes_[0],ratio_[0],self.data_train.coco,step)
            if step % self.save_pre_every == 0 and step>0:
                saver.save(sess, self.snapshot_file, epoch)
                epoch+=1
        coord.request_stop()
        coord.join(threads)
    def average_gradients(self,tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.

                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
    def load(self,saver, sess, ckpt_path):
        '''Load trained weights.

        Args:
          saver: TensorFlow saver object.
          sess: TensorFlow session.
          ckpt_path: path to checkpoint file with parameters.
        '''
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(ckpt_path, ckpt_name))
            print("Restored model parameters from {}".format(ckpt_name))
            return True
        else:
            return False
    def train_single(self):
        #with tf.variable_scope('inputs'):
        images, tags_tl, tags_br,heatmaps_tl, heatmaps_br, tags_mask, offsets_tl, offsets_br,boxes,ratio=self.data_train.get_batch_data(self.batch_i)

        #test_images, test_tags_tl, test_tags_br,test_heatmaps_tl, test_heatmaps_br, test_tags_mask, test_offsets_tl, test_offsets_br,test_boxes=self.data_test.get_batch_data(self.batch_size)
        #with tf.variable_scope('net'):
        #is_training=tf.constant(True)
        outs,test_outs=self.net.corner_net(images,tags_tl,tags_br,is_training=True)
        dets_tensor,debug_boxes=self.net.decode(*test_outs)
        #outs_test=self.net.corner_net(test_images,test_tags_tl,test_tags_br,is_training=False)
        loss,focal_loss,pull_loss,push_loss,offset_loss=self.net.loss(outs,[heatmaps_tl,heatmaps_br,tags_mask,offsets_tl,offsets_br])
        #with tf.variable_scope('train_op'):

        steps=tf.Variable(0,name='global_step',trainable=False)
        lr=tf.train.exponential_decay(self.lr,steps,self.decay_step,self.decay_rate,staircase= True, name= 'learning_rate')

        update=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update):
        train_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,steps)
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess=tf.InteractiveSession(config=config)
        init=tf.global_variables_initializer()
        coord = tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        saver=tf.train.Saver(max_to_keep=100)
        sess.run(init)
        print(self.num_steps)
        debug=Debug()
        epoch=5
        if self.load(saver, sess, self.snapshot_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        for step in range(self.num_steps):

            start=time.time()
            sess.run(update)
            _,loss_,focal_loss_,pull_loss_,push_loss_,offset_loss_,lr_=sess.run([train_op,loss,focal_loss,pull_loss,push_loss,offset_loss,lr])

            duration=time.time()-start

            print('step %d, loss %g, focal_loss %g, pull_loss %g, push_loss %g, offset_loss %g, time %g, lr %g'
                %(step,loss_,focal_loss_,pull_loss_,push_loss_,offset_loss_,duration,lr_))

            if step%100==0:
                dets_,images_,debug_boxes_,boxes_,ratio_=sess.run([dets_tensor,images,debug_boxes,boxes,ratio])
                debug.test_debug(images_[0],dets_[0],debug_boxes_[0],boxes_[0],ratio_[0],self.data_train.coco,step)
            if step % self.save_pre_every == 0 and step>0:
                saver.save(sess, self.snapshot_file, epoch)
                epoch+=1
        coord.request_stop()
        coord.join(threads)
if __name__=="__main__":

    t=Train()
    #t.train_single()
    t.train_mult()
