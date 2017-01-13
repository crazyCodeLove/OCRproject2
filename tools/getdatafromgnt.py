#coding=utf-8

"""
目标文件中每个字符由tagcode和bitmap组成,
单个字符长度item length=tagcode(2) + bitmap(desCharSize*desCharSize)

#####   HWDB1.1  start   ####
all character class is 3755

test data all character number is:      223991
test data 1000 class,character number is 59688
test data 100 class,character number is 5975


train database,all character number is 897758
train data 1000 class, character number is  239064
train data 100 class,character number is 23936
#####   HWDB1.1  end   ####


#####   HWDB1.0  start   ####
all character class is 3740, all class in HWDB1.1

test data all character number is:       309684
test data 100 class,character number is:


train database,all character number is:  1246991
train data 100 class,character number is:
#####   HWDB1.0  end   ####

all character

test data 100 class, character number is:  14202
train data 100 class, character number is: 56987

"""

import os
import numpy as np
import struct,cv2
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import random


traindirname = "/home/allen/work/data/limitclass/train100class"
testdirname = "/home/allen/work/data/limitclass/test100class"

descharacterTagcodeMapFile = "/home/allen/work/data/100class.pkl"


charWidth = 64
itemLength = charWidth*charWidth + 2



def next_batch(batchnum, dirname, itemLength, character_class, charTagcodeMapFile):
    """

    :param batchnum: 一次取
    :param dirname:
    :param itemLength:
    :param character_class:
    :param charTagcodeMapFile:
    :return:
    """
    filenames = sorted(os.listdir(dirname))
    filenum = -1
    batch_x = []
    batch_y = []

    while True:
        filenum += 1
        filenum = filenum % len(filenames)

        filename = filenames[filenum]
        filename = os.path.join(dirname, filename)

        # print filename

        with open(filename,mode='rb') as fobj:
            content = fobj.read()
            contentlength = len(content)
            start = 0

            while start<contentlength:
                if len(batch_y) == batchnum:
                    batch_x = []
                    batch_y = []

                fetchnum = batchnum - len(batch_x)
                end = start+ fetchnum * itemLength

                if end <= contentlength:
                    data2list(content, start, end, batch_x, batch_y, itemLength, charTagcodeMapFile)
                    start = end
                    batch_x,batch_y = fromList2Stand(batch_x,batch_y,character_class)
                    yield batch_x,batch_y

                    # for each in batch_y:
                    #     print tagmap[each].decode('gbk'),
                    #
                    # print ""
                    # for each in batch_x:
                    #     each = np.array(each).astype(dtype=np.float32).reshape([32,32])
                    #
                    #     each = Image.fromarray(each)
                    #     each.show()
                    #     plt.figure()
                    #     plt.imshow(each)
                    #     plt.show()


                else:
                    end = contentlength
                    data2list(content, start, end, batch_x, batch_y, itemLength, charTagcodeMapFile)
                    start = contentlength

def next_batch_dirs(batchnum, dirnames, charWidth, character_class, charTagcodeMapFile):
    """

    :param batchnum: 每次取的字符数
    :param dirnames: 目标文件的文件夹列表
    :param itemLength: 每个字符占用的字节数, = tagcode(2) + charWidth(64)*charWidth(64)
    :param character_class: 目标文件夹下的字符类别数
    :param charTagcodeMapFile: 使用哪一个tagcodemap影射文件
    :return: one hot 编码的一组[batch_x,batch_y]
    """

    itemLength = 2+charWidth*charWidth
    filenames = []
    for eachdir in dirnames:
        tfnames = os.listdir(eachdir)
        tfnames = [os.path.join(eachdir,fname) for fname in tfnames]
        filenames.extend(tfnames)

    random.shuffle(filenames)
    filenum = -1
    batch_x = []
    batch_y = []

    while True:
        filenum += 1
        filenum = filenum % len(filenames)

        filename = filenames[filenum]

        # print filename

        with open(filename,mode='rb') as fobj:
            content = fobj.read()
            contentlength = len(content)
            start = 0

            while start<contentlength:
                if len(batch_y) == batchnum:
                    batch_x = []
                    batch_y = []

                fetchnum = batchnum - len(batch_x)
                end = start+ fetchnum * itemLength

                if end <= contentlength:
                    data2list(content, start, end, batch_x, batch_y, itemLength, charTagcodeMapFile)
                    start = end
                    batch_x,batch_y = fromList2Stand(batch_x,batch_y,character_class)
                    yield batch_x,batch_y

                    # for each in batch_y:
                    #     print tagmap[each].decode('gbk'),
                    #
                    # print ""
                    # for each in batch_x:
                    #     each = np.array(each).astype(dtype=np.float32).reshape([32,32])
                    #
                    #     each = Image.fromarray(each)
                    #     each.show()
                    #     plt.figure()
                    #     plt.imshow(each)
                    #     plt.show()


                else:
                    end = contentlength
                    data2list(content, start, end, batch_x, batch_y, itemLength, charTagcodeMapFile)
                    start = contentlength



def fromList2Stand(batch_x,batch_y,character_class):
    """

    :param batch_x:
    :param batch_y:
    :param character_class:
    :return:
    """
    out_x = 255 - np.array(batch_x).astype(np.float32)

    out_y = np.zeros([len(batch_y),character_class],dtype=np.float64)
    for i in xrange(len(batch_y)):
        out_y[i,batch_y[i]] = 1.0

    return out_x,out_y


def data2list(data,start,end,batch_x,batch_y,itemLength,characterTagcodeMapFile):
    """

    :param data:
    :param start:
    :param end:
    :param batch_x:
    :param batch_y:
    :param itemLength:
    :param characterTagcodeMapFile:
    :return:
    """
    length = (end-start) / itemLength


    with open(characterTagcodeMapFile) as fobj:
        tagmap = pickle.load(fobj)

    for i in xrange(length):
        substart = i * itemLength + start
        tagcode = data[substart:substart+2]
        bitmap = data[substart+2:substart + itemLength]

        bitmap = [struct.unpack('<B',pixcel)[0] for pixcel in bitmap]

        batch_y.append(tagmap.index(tagcode))
        batch_x.append(bitmap)




def getGaborKernals(directions = 8):
    kernals = []

    ksize = 7
    sigma = 2
    thetas = [np.pi*k/directions for k in range(directions)]
    lambd = 5
    gamma = 1
    psi = 0
    for eachtheta in thetas:
        tk = cv2.getGaborKernel((ksize,ksize),sigma,eachtheta,lambd,gamma,psi=psi,ktype=cv2.CV_32F)
        kernals.append(tk)

    return kernals


def test():
    global charWidth,itemLength,descharacterTagcodeMapFile
    character_class = 100

    number = 6
    gen = next_batch_dirs(number, [testdirname,],itemLength, character_class,descharacterTagcodeMapFile)

    with open(descharacterTagcodeMapFile) as fobj:
        tagmap = pickle.load(fobj)
    
    for j in xrange(10):
        x,y = gen.next()
        plt.figure()


        for i in xrange(number):
            pic = x[i,:]
            label = y[i,:]
            index = np.argmax(label)
            print tagmap[index].decode('gbk'),

            pic = np.reshape(pic,[charWidth,charWidth])
            plt.subplot(2,3,i+1)
            plt.imshow(pic)
        print

        plt.show()






if __name__ == "__main__":
    test()
