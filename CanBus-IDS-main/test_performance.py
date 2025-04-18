from train import Model
import os
import numpy as np
from utils import *
import logging
logging.getLogger('tensorflow').disabled = True

batch_size=1
model = Model(model='CAAE', data_dir='./Data/', batch_size=batch_size)
tf_data = data_from_tfrecord(['./Data/Train_0.7_Labeled_0.1/DoS/test'], batch_size, 1)
sess_cpu = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU':0}))
x, y = data_stream(tf_data, sess_cpu)

model_path = './results/all/CNN_WGAN_2024-09-29 23:53:23.329923_10_0.0001_64_100_0.5/Saved_models'
time = model.timing(x, model_path, use_gpu=False)
print('Average test time for {} sample: {}ms'.format(batch_size, np.mean(time)*1000))


