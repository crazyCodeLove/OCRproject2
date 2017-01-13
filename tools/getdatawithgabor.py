# coding=utf-8

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
import struct, cv2
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import random

traindirname = "/home/allen/work/data/limitclass/train100class"
testdirname = "/home/allen/work/data/limitclass/test100class"

descharacterTagcodeMapFile = "/home/allen/work/data/100class.pkl"

charWidth = 64
itemLength = charWidth * charWidth + 2


def next_batch_dirs(batchnum, dirnames, charWidth, character_class, charTagcodeMapFile, directions):
    """

    :param batchnum: 每次取的字符数
    :param dirnames: 目标文件的文件夹列表
    :param itemLength: 每个字符占用的字节数, = tagcode(2) + charWidth(64)*charWidth(64)
    :param character_class: 目标文件夹下的字符类别数
    :param charTagcodeMapFile: 使用哪一个tagcodemap影射文件
    :return: one hot 编码的一组[batch_x,batch_y]
    """

    gaborks = getGaborKernals(directions=directions)
    itemLength = 2 + charWidth * charWidth
    filenames = []
    for eachdir in dirnames:
        tfnames = os.listdir(eachdir)
        tfnames = [os.path.join(eachdir, fname) for fname in tfnames]
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

        with open(filename, mode='rb') as fobj:
            content = fobj.read()
            contentlength = len(content)
            start = 0

            while start < contentlength:
                if len(batch_y) == batchnum:
                    batch_x = []
                    batch_y = []

                fetchnum = batchnum - len(batch_x)
                end = start + fetchnum * itemLength

                if end <= contentlength:
                    data2list(content, start, end, batch_x, batch_y, charWidth, charTagcodeMapFile)
                    start = end
                    batch_x, batch_y = fromList2Stand(batch_x, batch_y, character_class, gaborks)
                    yield batch_x, batch_y

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
                    data2list(content, start, end, batch_x, batch_y, charWidth, charTagcodeMapFile)
                    start = contentlength


def fromList2Stand(batch_x, batch_y, character_class,kernals):
    """
    :param batch_x:
    :param batch_y:
    :param character_class:
    """
    tbatchx = []
    for eachchar in batch_x:
        charfeatures=[]
        for eachk in kernals:
            feature = cv2.filter2D(eachchar,-1,eachk)
            charfeatures.append(feature)
        charfeatures.insert(0,eachchar)
        # plt.figure()
        # for iti in range(len(charfeatures)):
        #     plt.subplot(3,3,iti+1)
        #     plt.imshow(charfeatures[iti])
        #     plt.title('feature '+str(iti+1))
        # plt.show()

        tx = np.stack(charfeatures,-1)
        tbatchx.append(tx)

    out_x = np.array(tbatchx).astype(np.float32)

    out_y = np.zeros([len(batch_y), character_class], dtype=np.float64)
    for i in xrange(len(batch_y)):
        out_y[i, batch_y[i]] = 1.0

    return out_x, out_y


def data2list(data, start, end, batch_x, batch_y, charWidth, characterTagcodeMapFile):
    """
    将每个字符和对应的tagcode存放到batch_x,batch_y中
    """
    itemLength = 2+charWidth*charWidth
    length = (end - start) / itemLength

    with open(characterTagcodeMapFile) as fobj:
        tagmap = pickle.load(fobj)

    for i in xrange(length):
        substart = i * itemLength + start
        tagcode = data[substart:substart + 2]
        tbitmap = data[substart + 2:substart + itemLength]
        bitmap = np.zeros([charWidth,charWidth],dtype=np.float32)
        for iti in range(charWidth):
            for itj in range(charWidth):
                bitmap[iti][itj] = struct.unpack('<B',tbitmap[iti*charWidth+itj])[0]

        bitmap = 255 - bitmap
        batch_y.append(tagmap.index(tagcode))
        batch_x.append(bitmap)


def getGaborKernals(directions=8):
    kernals = []

    ksize = 7
    sigma = 2
    thetas = [np.pi * k / directions for k in range(directions)]
    lambd = 10
    gamma = 1
    psi = 0
    for eachtheta in thetas:
        tk = cv2.getGaborKernel((ksize, ksize), sigma, eachtheta, lambd, gamma, psi=psi, ktype=cv2.CV_32F)
        kernals.append(tk)

    return kernals


def test():
    global charWidth, itemLength, descharacterTagcodeMapFile
    character_class = 100

    number = 6
    gen = next_batch_dirs(number, [testdirname, ], charWidth, character_class, descharacterTagcodeMapFile)

    for j in xrange(10):
        x, y = gen.next()
        print x.shape
        print y.shape


if __name__ == "__main__":
    test()
