import math
import numpy as np 
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt 
from skimage.viewer import ImageViewer
import caffe
import simplejson

# load the model
# net = caffe.Net('./alex/deploy.prototxt', 1, weights='./alex/bvlc_alexnet.caffemodel')
net = caffe.Net('./alex_ft/deploy.prototxt', 1, weights='./alex_ft/alex2.caffemodel')

# load input and configure preprocessing 
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1)) # after transpose: [channal,data,number]
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
BATCH_SIZE = 50
net.blobs['data'].reshape(BATCH_SIZE,3,227,227)

# load the image in the data layer
fs = open('../lfw_funneled/pairs_06.txt','r')
lines = fs.readlines()
N = 0
imgs = list()
for line in lines:
	if(line != '\n'):
		N = N+1
		imgs.append(caffe.io.load_image("../lfw_funneled/"+line[0:-1]))


# forward imgs through the network using gpu
caffe.set_device(0)
caffe.set_mode_gpu()
fc8s = np.zeros((N,1000)) # fc8's sample is of size 1000 
for i in range(N):
	net.blobs['data'].data[i%BATCH_SIZE] = transformer.preprocess('data', np.array(imgs[i]))
	if i%BATCH_SIZE == 0:
		net.forward()
		fc8s[i:i+BATCH_SIZE] = net.blobs['fc8'].data

# compute L2 norm between matched pairs (i%4 == 0, i%4 == 1) and unmacthed ones (i%4 == 2, i%4 == 3)
scores = list()
labels = list()
i=0
while(i<N):
	scores.append(np.linalg.norm(fc8s[i]-fc8s[i+1],axis =0))
	labels.append(0)
	scores.append(np.linalg.norm(fc8s[i+2]-fc8s[i+3],axis=0))
	labels.append(1)
	i = i+4

# calculate the ROC curve. fpr: false positive rate; tpr: true positive rate;
fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
roc_area = auc(fpr, tpr)

# save fpr, tpr and roc_area to file
f = open('./tmp/alex_roc.txt','w')
simplejson.dump({'fpr': list(fpr), 'tpr': list(tpr), 'roc_area': roc_area},f)
f.close()

# plot ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_area)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

