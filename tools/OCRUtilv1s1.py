#coding=utf-8
"""
该文件就是tensorflowproject1中tools package中的OCRUtilv3s2文件
MyLog is to log result
learning_rate_down is to down learning rate

all character
test data all 3755 class, character number is:   533675
train data all 3755 class, character number is: 2144749

test data 100 class, character number is:  14202
train data 100 class, character number is: 56987

"""
import logging,os
import tensorflow as tf
from tensorflow.contrib.layers.python.layers.layers import batch_norm


class MyLog(object):
    logfile = ""
    logger = None

    def __init__(self,logfile):
        self.logfile = logfile
        filehandler = logging.FileHandler(filename=logfile,encoding='utf-8')
        fmter = logging.Formatter(fmt="%(asctime)s %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
        filehandler.setFormatter(fmter)
        loger = logging.getLogger(__name__)
        loger.addHandler(filehandler)
        loger.setLevel(logging.DEBUG)
        self.logger = loger

    def log_message(self,msg):
        self.logger.debug(msg)



def get_accurate(prediction,labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(prediction,1),tf.arg_max(labels,1)),dtype=tf.float32))


def get_test_right_num(prediction,labels):
    return tf.reduce_sum(tf.cast(tf.equal(tf.arg_max(prediction,1),tf.arg_max(labels,1)),dtype=tf.float32))

def add_fc_layer(inputs, inFeatures, outFeatures, layerName="layer", activateFunc=None, keepProb=-1):
    with tf.name_scope(layerName):
        Weights = tf.Variable(tf.truncated_normal([inFeatures, outFeatures], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, tf.float32, [outFeatures]))

        y = tf.matmul(inputs,Weights) + biases
        if activateFunc is None:
            outputs = y
        else:
            outputs = activateFunc(y)

        if keepProb != -1:
            outputs = tf.nn.dropout(outputs, keepProb)
        return outputs,Weights



def add_building_block_carriage(batch_num,deepk, carriage_nums, inputs, kernalWidth,
                                is_training_ph, scope=None, layername="layer",
                                activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):
    """
    conv layer number: 2*carriage_nums
    将一组building_bolck组合在一起,形成一串
    the depth of outputs is same as inDepth
    :param carriage_nums:要添加的building_bolck数目
    """
    if scope is None:
        raise ValueError('scope should be a string')
    if carriage_nums <1:
        raise ValueError('nums should not be less than 1')

    tscope = scope + "blockincre"
    outputs = building_block_incre(batch_num,deepk,inputs,kernalWidth,is_training_ph,scope=tscope,
                                   layername=layername,activateFunc=activateFunc,stride=stride)

    for it in range(1, carriage_nums):
        tscope = scope + "block" + str(it)
        outputs = building_block_same(outputs,kernalWidth,is_training_ph,scope=tscope,layername=layername,
                                      activateFunc=activateFunc,stride=stride)

    return outputs

def building_block_incre(batch_num,deepk,inputs,kernalWidth,
                         is_training_ph,scope=None, layername="layer",
                         activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):
    """
    conv layer number: 2
    首先是一层max pooling层,,两层卷积层,和project short cut connection 层组成
    在增加维度时short cut connection使用projection
    the depth of outputs is 2*inDepth
    """
    if scope is None:
        raise ValueError('scope should be a string')
    if deepk < 1:
        raise ValueError('deepk should not be less than 1')

    inshape = inputs.get_shape().as_list()
    inDepth = inshape[3]
    inshape[0] = batch_num

    depth = deepk*inDepth

    proj = inputs
    zero_pad = tf.constant(0,dtype=tf.float32,shape=inshape)
    for it in range(deepk-1):
        proj = tf.concat(3,[proj,zero_pad])

    tscope = scope + "layer1"
    y1 = add_BN_conv_layer(inputs,kernalWidth,inDepth,depth,
                           is_training_ph,tscope,activateFunc=activateFunc)

    tscope = scope + "layer2"
    y2 = add_BN_conv_layer(y1,kernalWidth,depth,depth,
                           is_training_ph,tscope,activateFunc=None)

    hx = tf.add(y2,proj)
    outputs = activateFunc(hx)
    return outputs

def building_block_same(inputs,kernalWidth,
                        is_training_ph, scope=None, layername="layer",
                        activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):
    """
    conv layer number:2
    """
    if scope is None:
        raise ValueError('scope should be a string')
    inDepth = inputs.get_shape().as_list()[3]
    depth = inDepth

    tscope = scope + "layer1"
    y1 = add_BN_conv_layer(inputs,kernalWidth,inDepth,depth,
                           is_training_ph,tscope,activateFunc=activateFunc)

    tscope = scope + "layer2"
    y2 = add_BN_conv_layer(y1,kernalWidth,depth,depth,
                           is_training_ph,tscope,activateFunc=None)

    hx = tf.add(y2,inputs)
    outputs = activateFunc(hx)
    return outputs

def building_block_desc(inputs,is_training_ph, scope=None, layername="layer",
                        activateFunc=None, stride=[1, 1, 1, 1],
                        numerator=3,denominator=5):
    """
    conv layer:1
    压缩率：numerator/denominator
    """

    inDepth = inputs.get_shape().as_list()[3]
    outDepth = inDepth*numerator/denominator
    tscope = scope + "layer1"

    kw=1
    outputs = add_BN_conv_layer(inputs,kw,inDepth,outDepth,is_training_ph,
                                tscope,layername=layername,activateFunc=activateFunc,
                                stride=stride)
    return outputs


def add_BN_conv_layer(inputs, kernalWidth, inDepth, outDepth,
                      is_training_ph, scope , layername="layer",
                      activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):

    # inDepth = inputs.get_shape().as_list()[3]

    with tf.name_scope(layername):
        Weights = tf.Variable(tf.truncated_normal([kernalWidth, kernalWidth, inDepth, outDepth], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, tf.float32, [outDepth]))

        y1 = tf.nn.conv2d(inputs, Weights, stride, padding='SAME') + biases

        outputs = tf.cond(is_training_ph,
                           lambda: batch_norm(y1,decay=0.94, is_training=True,
                                              center=False, scale=True,
                                              activation_fn=activateFunc,
                                              updates_collections=None, scope=scope),
                           lambda: batch_norm(y1,decay=0.94, is_training=False,
                                              center=False, scale=True,
                                              activation_fn=activateFunc,
                                              updates_collections=None, scope=scope,
                                              reuse=True))

        return outputs




def add_maxpool_layer(inputs,step=2,layername="poolLayer"):
    with tf.name_scope(layername):
        kernal = [1, step, step, 1]
        return tf.nn.max_pool(inputs,kernal,strides=kernal,padding='SAME')


def add_averagepool_layer(inputs,step=2,layername="poolLayer"):
    with tf.name_scope(layername):
        kernal = [1, step, step, 1]
        return tf.nn.avg_pool(inputs,kernal,strides=kernal,padding='SAME')


def conv2fc(inputs):
    conv_out_shape = inputs.get_shape().as_list()
    fcl_in_features = conv_out_shape[1] * conv_out_shape[2] * conv_out_shape[3]
    fcl_inputs = tf.reshape(inputs, [-1, fcl_in_features])
    return fcl_inputs,fcl_in_features

def down_learning_rate(test_acc, lr):
    if test_acc >=0.8 and lr>5e-5:
        lr /= 5.0
    elif test_acc>0.9 and lr>1e-6:
        lr *= 0.9
    elif test_acc>0.95:
        lr *= 0.95

    return lr

def empty_dir(dirname):
    if not os.path.exists(dirname):
        raise ValueError('%s dir not exist'%dirname)

    for each in os.listdir(dirname):
        os.remove(os.path.join(dirname,each))
    return True

def test():
    # loger = MyLog("/home/allen/work/temp/test.txt")
    # loger.log_message("nice to meet you")
    # num = 0
    #
    # lr = 2e-3
    # acc = 0.85
    # while True:
    #     num += 1
    #     lr = down_learning_rate(acc,lr)
    #     print "%d %f"%(num,lr)
    #     if lr < 1e-4:
    #         acc = 0.95
    #
    #     if lr < 1e-5:
    #         acc = 0.96
    dirname = "/home/allen/work/variableSave/0temp"
    empty_dir(dirname)
    print "done"





if __name__ == "__main__":
    test()


