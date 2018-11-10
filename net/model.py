import tensorflow as tf
from module.corner_pooling import TopPool,BottomPool, LeftPool, RightPool

class Model():
    def conv_bn_re(self,inputs,input_dim,out_dim,strides=(1,1),use_relu=True,use_bn=True,k=3,is_training=True,scope='conv_bn_re'):
        with tf.variable_scope(scope):
            #x=tf.contrib.layers.conv2d(inputs,out_dim,k,stride=strides,activation_fn=None)
            x=tf.layers.conv2d(inputs,out_dim,k,strides=strides,padding='same')
            if use_bn:
                x=tf.contrib.layers.batch_norm(x,is_training=is_training)
            if use_relu:
                x=tf.nn.relu(x)
            return x
    def residual(self,inputs,input_dim,out_dim,k=3,strides=(1,1),is_training=True,scope='residual'):
        with tf.variable_scope(scope):
            #assert inputs.get_shape().as_list()[3]==input_dim
            #low layer 3*3>3*3
            x=self.conv_bn_re(inputs,input_dim,out_dim,strides=strides,is_training=is_training,scope='up_1')
            x=self.conv_bn_re(x,out_dim,out_dim,use_relu=False,is_training=is_training,scope='up_2')
            #skip,up layer 1*1
            skip=self.conv_bn_re(inputs,input_dim,out_dim,strides=strides,use_relu=False,k=1,is_training=is_training,scope='low')
            #skip+x
            res=tf.nn.relu(tf.add(skip,x))
            return res
    def res_block(self,inputs,input_dim,out_dim,n,k=3,is_training=True,scope='res_block'):
        with tf.variable_scope(scope):
            x=self.residual(inputs,input_dim,out_dim,k=k,is_training=is_training,scope='residual_0')
            for i in range(1,n):
                x=self.residual(x,out_dim,out_dim,k=k,is_training=is_training,scope='residual_%d'%i)
            return x
    def hourglass(self,inputs,n_deep,n_res,n_dims,is_training=True,scope='hourglass_5'):
        with tf.variable_scope(scope):
            curr_res=n_res[0]
            next_res=n_res[1]
            curr_dim=n_dims[0]
            next_dim=n_dims[1]

            up_1=self.res_block(inputs,curr_dim,curr_dim,curr_res,is_training=is_training,scope='up_1')

            half=tf.nn.max_pool(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
            low_1=self.res_block(half,curr_dim,next_dim,curr_res,is_training=is_training,scope='low_1')
            if n_deep>1:
                low_2=self.hourglass(low_1,n_deep-1,n_res[1:],n_dims[1:],is_training=is_training,scope='hourglass_%d'%(n_deep-1))
            else:
                low_2=self.res_block(low_1,next_dim,next_dim,next_res,is_training=is_training,scope='low_2')
            low_3=self.res_block(low_2,next_dim,curr_dim,curr_res,is_training=is_training,scope='low_3')

            up_2=tf.image.resize_nearest_neighbor(low_3,tf.shape(low_3)[1:3]*2,name='up_2')
            merge=tf.add(up_1,up_2)
            return merge
    def start_conv(self,img,is_training=True,scope='start'):
        with tf.variable_scope(scope):
            x=tf.contrib.layers.conv2d(img,128,7,2)
            x=tf.contrib.layers.batch_norm(x,is_training=is_training)
            x=self.residual(x,128,256,strides=(2,2),scope='residual_start')
            return x
    def corner_pooling(self,inputs,input_dim,out_dim,k=3,is_training=True,scope='corner_pooling'):
        with tf.variable_scope(scope):
            with tf.variable_scope('top_left'):
                top=self.conv_bn_re(inputs,input_dim,128,is_training=is_training,scope='top')
                top_pool=TopPool(top)

                left=self.conv_bn_re(inputs,input_dim,128,is_training=is_training,scope='left')
                left_pool=LeftPool(left)


                #top_left=tf.add(top,left)

                top_left=tf.add(top_pool,left_pool)
                top_left=self.conv_bn_re(top_left,128,out_dim,use_relu=False,is_training=is_training,scope='top_left')

                skip_tl=self.conv_bn_re(inputs,input_dim,out_dim,use_relu=False,k=1,is_training=is_training,scope='skip')

                merge_tl=tf.nn.relu(tf.add(skip_tl,top_left))
                merge_tl=self.conv_bn_re(merge_tl,out_dim,out_dim,is_training=is_training,scope='merge_tl')
            with tf.variable_scope('bottom_right'):
                bottom=self.conv_bn_re(inputs,input_dim,128,is_training=is_training,scope='bottom')
                bottom_pool=BottomPool(bottom)

                right=self.conv_bn_re(inputs,input_dim,128,is_training=is_training,scope='right')
                right_pool=RightPool(right)

                #bottom_right=tf.add(bottom,right)

                bottom_right=tf.add(bottom_pool,right_pool)
                bottom_right=self.conv_bn_re(bottom_right,128,out_dim,use_relu=False,is_training=is_training,scope='bottom_right')

                skip_br=self.conv_bn_re(inputs,input_dim,out_dim,use_relu=False,k=1,is_training=is_training,scope='skip')

                merge_br=tf.nn.relu(tf.add(skip_br,bottom_right))
                merge_br=self.conv_bn_re(merge_br,out_dim,out_dim,is_training=is_training,scope='merge_br')
            return merge_tl,merge_br
    def heat(self,inputs,input_dim,out_dim,scope='heat'):
        #out_dim=80
        with tf.variable_scope(scope):
            x=self.conv_bn_re(inputs,input_dim,input_dim,use_bn=False)
            x=tf.layers.conv2d(x,out_dim,1)
            return x
    def tag(self,inputs,input_dim,out_dim,scope='tag'):
        #out_dim=1
        with tf.variable_scope(scope):
            x=self.conv_bn_re(inputs,input_dim,input_dim,use_bn=False)
            x=tf.layers.conv2d(x,out_dim,1)
            return x
    def offset(self,inputs,input_dim,out_dim,scope='offset'):
        #out_dim=2
        with tf.variable_scope(scope):
            x=self.conv_bn_re(inputs,input_dim,input_dim,use_bn=False)
            x=tf.layers.conv2d(x,out_dim,1)
            return x
    def hinge(self,inputs,input_dim,out_dim,is_training=True,scope='hinge'):
        with tf.variable_scope(scope):
            x=self.conv_bn_re(inputs,input_dim,out_dim,is_training=is_training)
            return x
    def inter(self,input_1,input_2,out_dim,is_training=True,scope='inter'):
        with tf.variable_scope(scope):
            x_1=self.conv_bn_re(input_1,tf.shape(input_1)[3],out_dim,use_relu=False,k=1,is_training=is_training,scope='branch_start')
            x_2=self.conv_bn_re(input_2,tf.shape(input_2)[3],out_dim,use_relu=False,k=1,is_training=is_training,scope='branch_hourglass1')
            x=tf.nn.relu(tf.add(x_1,x_2))
            x=self.residual(x,out_dim,out_dim,is_training=is_training)
            return x









