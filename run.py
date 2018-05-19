#!/usr/bin/env python3

"""
File that reproduces the best submission file we have made,
using an already trained CNN model whose paremeters are saved
and loaded here.
"""

from helpers_loading import *
from helpers_submission import *
from helpers_cnn import *
from CNNModel import *

window_size = 64
model_constructor = construct_keras_model_best

model = CNNModel(window_size=window_size, model_constructor=model_constructor)
model.load('best_model.h5')

submission_filename = 'submission.csv'

X_test = load_testing_set()

Y_test = model.predict(X_test)

make_submission(Y_test, submission_filename)