# Generate the .caffemodel file for the combined net using original alex net weights

import numpy as np
import sys, os
import caffe

# get the oringinal and combined net instance
os.chdir('./alex/')
net = caffe.Net('deploy.prototxt', 'bvlc_alexnet.caffemodel', caffe.TRAIN)
os.chdir('../alex_ft/')
comb_net = caffe.Net('train_val.prototxt', caffe.TRAIN)

# The target layers to be replicated: 
layer_names = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8']

# For each of the pretrained target params to 
# the corresponding layer of the combined net:
for layer in layer_names:
    W = net.params[layer][0].data[...] # Grab the pretrained weights
    b = net.params[layer][1].data[...] # Grab the pretrained bias
    comb_net.params['{}'.format(layer)][0].data[...] = W # Insert into new combined net
    comb_net.params['{}'.format(layer)][1].data[...] = b
    comb_net.params['{}_p'.format(layer)][0].data[...] = W
    comb_net.params['{}_p'.format(layer)][1].data[...] = b 

# Save the combined model with pretrained weights to a caffemodel file:
comb_net.save('alex_ft_init.caffemodel')