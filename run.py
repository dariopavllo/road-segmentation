# -*- coding: utf-8 -*-

from helpers import *
from cnn_model import CnnModel
from helpers import *
import keras.backend as K
from sys import exit

if K.backend() != 'theano':
    print('Error: for reproducibility, this code is intented to be run with Theano backend.')
    exit()

# Set image_img_ordering to 'th', since the model has been trained with Theano
K.set_image_dim_ordering('th')

model = CnnModel()
model.load('weights.h5')

model.model.summary()

submission_filename = 'submission.csv'
image_filenames = []
for i in range(1, 51):
    image_filename = 'test_set_images/test_'+str(i)+'/test_' + str(i) + '.png'
    image_filenames.append(image_filename)
    

generate_submission(model, submission_filename, *image_filenames)