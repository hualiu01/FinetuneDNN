import numpy as np 
import matplotlib.pyplot as plt 
import simplejson

# save fpr, tpr and roc_area to file
f = open('vgg_roc.txt','r')
data = simplejson.load(f)
f.close()
fpr = data['fpr']
tpr = data['tpr']
roc_area = data['roc_area']

f = open('vgg_roc2.txt','r')
data = simplejson.load(f)
f.close()
fpr2 = data['fpr']
tpr2 = data['tpr']
roc_area2 = data['roc_area']

# plot ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_area)
plt.plot(fpr2, tpr2, color='black',
         lw=2, label='ROC2 curve (area = %0.2f)' % roc_area2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()