#coding=utf-8

"""
convert source file to destination file
format of destination file is
item,        type,             length,                instance,                        comment
Tag code,    char,             2B,                    "阿"=0xb0a1 stored as 0xa1b0
bitmap,      unsighed char,    width*height bytes,                                  ,  stored row by row

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



#####  compete test data start     ####
character class is:3740+15, 3740 in DB1.1, 15 not in DB1.1

compete test data, all character number is:224419



#####  compete test data end     ####


all character
test data all 3755 class, character number is:   533675
train data all 3755 class, character number is: 2144749

test data 100 class, character number is:  14124
train data 100 class, character number is: 57304

compete test data 100 class, character number is: 5988

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from PIL import Image
import pickle


trainoridirnamev0 = "/home/allen/work/data/HWDB1.0des64/HWDB1.0trn_gnt"
traindesdirnamev0 = "/home/allen/work/data/limitclass/train100class"

testoridirnamev0 = "/home/allen/work/data/HWDB1.0des64/HWDB1.0tst_gnt"
testdesdirnamev0 = "/home/allen/work/data/limitclass/test100class"

trainoridirname = "/home/allen/work/data/HWDB1.1des64/HWDB1.1trn_gnt"
traindesdirname = "/home/allen/work/data/limitclass/train100class"

testoridirname = "/home/allen/work/data/HWDB1.1des64/HWDB1.1tst_gnt"
testdesdirname = "/home/allen/work/data/limitclass/test100class"

compete_test_oridir = "/home/allen/work/data/competeTestdes64"
compete_test_desdir = "/home/allen/work/data/limitclass/competeTest100cl"

descharacterTagcodeMapFile = "/home/allen/work/data/100class.pkl"
oricharacterTagcodeMapFile = "/home/allen/work/data/tagindexmap.pkl"

buffer=[]
class_num = 100
charWidth = 64
item_length = 2+charWidth*charWidth


def fromSrc2Des(oridirname,desdirname):
    """
    将原文件夹下文件内字符在descharacterTagcodeMapFile中的字符写到目标文件夹中
    :param oridirname:
    :param desdirname:
    :return:
    """
    global tag_buffer,bitmap_buffer,desfilename

    orifilenames = sorted(os.listdir(oridirname))

    numOfFile = 1

    # if len(orifilenames)!=0:
    for filename in orifilenames:
        # filename = orifilenames[0]

        orifilename = os.path.join(oridirname, filename)
        desfilename = os.path.join(desdirname, filename)

        readFileContent(orifilename)
        write2DesFile(desfilename)

        print "%5dth file %s convert completed"%(numOfFile,orifilename)
        numOfFile += 1

def readFileContent(filename):
    """
    读取文件，将tagcode在descharacterTagcodeMapFile文件中的字符保存到buffer中
    方便以后写入目标文件中
    :param filename: 要读取的文件
    """
    global buffer,class_num,descharacterTagcodeMapFile,item_length

    with open(descharacterTagcodeMapFile) as fobj:
        tagcodeMap = pickle.load(fobj)

    with open(filename,mode='rb') as fobj:
        content = fobj.read()

    content_length = len(content)
    start = 0

    while start<content_length:
        tagcode = content[start:start+2]

        if tagcode in tagcodeMap:
            end = start + item_length
            buffer.append(content[start:end])

        start += item_length



def write2DesFile(filename):
    """
    将buffer中的字符(tagcode,bitmap)写进目标文件
    :param filename: 要写入的文件名
    """
    global buffer
    if len(buffer) == 0:
        return

    with open(filename,'wb') as fobj:
        content = ''.join(buffer)

        # char = buffer[0][:2]
        # print "buffer 1 item",char.decode('gbk')
        # char = content[:2]
        # print "content 1 item",char.decode('gbk')


        fobj.write(content)
    buffer = []

def process_dir(*dirlist):
    for each in dirlist:
        if not os.path.exists(each):
            os.mkdir(each)



def limitDesClass():
    """
    convert oricharacterTagcodeMapFile to des limit class characterTagcodeMapFile

    :return:
    """
    global class_num
    global oricharacterTagcodeMapFile,descharacterTagcodeMapFile

    with open(oricharacterTagcodeMapFile) as fobj:
        data = pickle.load(fobj)
    data = data[:class_num]

    with open(descharacterTagcodeMapFile, mode='w')as fobj:
        pickle.dump(data,fobj)
    print "process done"


def showDesImage(desdirname):
    global charWidth
    desFilenames = sorted(os.listdir(desdirname))

    filename = desFilenames[3]
    filename = os.path.join(desdirname, filename)

    print filename

    with open(filename,mode='rb') as fobj:
        itemSize = 2+charWidth*charWidth
        content = fobj.read()
        for i in xrange(10):

            start = i*itemSize
            end = (i+1)*itemSize
            tagcode = content[start:start+2]
            bitmap = content[start+2:end]
            bitmap = [struct.unpack('<B',each)[0] for each in bitmap]
            bitmap = np.array(bitmap).astype(np.ubyte).reshape([charWidth,charWidth])

            print tagcode.decode('gbk')
            plt.figure()
            img = Image.fromarray(bitmap)
            plt.imshow(img)

            plt.show()



def calculateAllCharacterCount(dirname):
    """
    计算目标文件夹下所有文件字符总数
    显示单个文件字符数，总字符数
    :return:
    """
    global item_length
    filenames = sorted(os.listdir(dirname))

    allCharacterCount = 0
    filenum = 0

    chSize = item_length

    # if len(filenames) != 0:
    #     filename = filenames[0]
    #     filename = os.path.join(desdirname,filename)

    for each in filenames:
        filename = os.path.join(dirname, each)

        with open(filename,mode='rb') as fobj:
            content = fobj.read()
            filenum += 1
            fileCharacterCount = len(content)/chSize

            allCharacterCount += fileCharacterCount

            print "%d th file, character count is %d, all count is %d"%(filenum, fileCharacterCount, allCharacterCount)


def checkWhetherDB0inDB1(dirname, descharTagcodeMapFile):
    """
    检测dirname文件夹字符是否都在descharacterTagcodeMapFile上面
    如果不在，显示出来。
    处理过每个文件后，显示一下当前读取到的不同字符种类数目

    :param dirname:HWDB1.0数据库文件夹
    :return:
    """
    global item_length

    with open(descharTagcodeMapFile) as rfobj:
        destagcodemap = pickle.load(rfobj)

    db0tagcodemap = []
    filenames = sorted(os.listdir(dirname))
    for eachfilename in filenames:
        filename = os.path.join(dirname,eachfilename)
        with open(filename, mode='rb') as rfobj:
            content = rfobj.read()

        charnum = len(content)/item_length
        for i in xrange(charnum):
            start = i*item_length
            tagcode = content[start:start+2]

            if (tagcode in destagcodemap) and (tagcode not in db0tagcodemap):
                db0tagcodemap.append(tagcode)

                #如果不在descharTagcodeMapfile中，就将该字符显示出来
            elif tagcode not in destagcodemap:
                print tagcode.decode('gbk'),"not in destagcodemap"

        print "process "+filename+", now db0tagcodemap length is ",len(db0tagcodemap)


def startCheck():
    global trainoridirnamev0,testoridirnamev0,descharacterTagcodeMapFile,oricharacterTagcodeMapFile
    global compete_test_desdir,compete_test_oridir

    checkWhetherDB0inDB1(compete_test_desdir,descharacterTagcodeMapFile)
    print "####train DB check over\n\n\n\n\n"
    # checkWhetherDB0inDB1(testoridirnamev0,descharacterTagcodeMapFile)


def remove_empty_file():
    global traindesdirnamev0,testdesdirnamev0
    dirnames = [traindesdirnamev0,testdesdirnamev0]
    for dirname in dirnames:
        filenames = sorted(os.listdir(dirname))
        for itfname in filenames:
            filename = os.path.join(dirname,itfname)
            with open(filename) as rfobj:
                content = rfobj.read()
            if len(content) == 0:
                os.remove(filename)
                print filename+" removed"

    print "process over"



def test():
    # process_dir(traindesdirnamev0,testdesdirnamev0)
    # limitDesClass()
    # showDesImage(traindesdirnamev0)
    calculateAllCharacterCount(compete_test_desdir)

    # calculateAllCharacterCount(trainoridirname)
    # calculateAllCharacterCount(traindesdirnamev0)
    # startCheck()
    # remove_empty_file()


    # fromSrc2Des(compete_test_oridir,compete_test_desdir)
    # fromSrc2Des(testoridirnamev0,testdesdirnamev0)
    #
    # fromSrc2Des(trainoridirname, traindesdirname)
    # fromSrc2Des(testoridirname, testdesdirname)

if __name__ == "__main__":
    test()