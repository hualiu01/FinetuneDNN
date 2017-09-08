# Generate the .caffemodel file for the combined net using original alex net weights

import numpy as np
import sys, os
import caffe

# get the oringinal and combined net instance
os.chdir('./vgg16/')
net = caffe.Net('deploy.prototxt', 'VGG_ILSVRC_16_layers.caffemodel', caffe.TRAIN)
os.chdir('../vgg16_ft/')
comb_net = caffe.Net('train_val.prototxt', caffe.TRAIN)

# The target layers to be replicated: 
layer_names = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3',
'conv5_1','conv5_2','conv5_3','fc6','fc7','fc8']

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
comb_net.save('vgg_ft_init.caffemodel')