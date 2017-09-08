#coding:utf-8
import lmdb
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2 
 
#basic setting
lmdb_file = 'lmdb_data'#期望生成的数据文件
batch_size = 200       #lmdb对于数据进行的是先缓存后一次性写入从而提高效率，因此定义一个batch_size控制每次写入的量。
 
# create the leveldb file
lmdb_env = lmdb.open(lmdb_file, map_size=int(1e12))#生成一个数据文件，定义最大空间
lmdb_txn = lmdb_env.begin(write=True)              #打开数据库的句柄
datum = caffe_pb2.Datum()                          #这是caffe中定义数据的重要类型

fs = open('../lfw_funneled/pairs_01.txt','r')
lines = fs.readlines()
x = 0 
i = 0
while x+1 < len(lines):
    if(lines[x] == '\n'):
        x = x + 1
    data = np.zeros((250,250,6))
    img1=cv2.imread("../lfw_funneled/"+lines[x][:-1]) 
    img2=cv2.imread("../lfw_funneled/"+lines[x+1][:-1])   
    i = i + 2
    x = x + 2

    # save in datum
    data[:,:,0:3] = img1      
    data[:,:,3:6] = img2
    data = data.transpose(2,0,1)
    label = 0 if i%4 == 0 else 1                   #图像的标签，为了方便存储，这个必须是整数。
    datum = caffe.io.array_to_datum(data, label)   #将数据以及标签整合为一个数据项
 
    keystr = '{:0>8d}'.format(i)                 #lmdb的每一个数据都是由键值对构成的，因此生成一个用递增顺序排列的定长唯一的key
    lmdb_txn.put( keystr, datum.SerializeToString())#调用句柄，写入内存。
 
    # write batch
    if i % batch_size == 0 and i != 0:                        #每当累计到一定的数据量，便用commit方法写入硬盘。
        lmdb_txn.commit()
        lmdb_txn = lmdb_env.begin(write=True)      #commit之后，之前的txn就不能用了，必须重新开一个。
        print 'batch {} writen'.format(i)
    



lmdb_env.close()          