from utils     import Options
from simulator import Simulator
from transitionTable import TransitionTable

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Reshape

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this script assumes you did generate your data with the get_data.py script
# you are of course allowed to change it and generate data here but if you
# want this to work out of the box first run get_data.py
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                             opt.minibatch_size, opt.valid_size,
                             opt.states_fil, opt.labels_fil)

train_data = trans.get_train()
valid_data = trans.get_valid()

# 1. train

model = Sequential()

# use data like provided as series
model.add(Reshape((opt.state_siz * opt.hist_len, 1),
                  input_shape=(opt.state_siz * opt.hist_len,)))
model.add(Conv1D(8, kernel_size=64, strides=1, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(16, kernel_size=32, strides=1, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(32, kernel_size=16, strides=1, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(32, kernel_size=8, strides=1, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(32, kernel_size=4, strides=1, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(opt.act_num, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

callbacks = []
if opt.log_tensorboard:
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=opt.tensorboard_log_dir, histogram_freq=1,
                                                 write_graph=True, write_images=True, write_grads=True)
    callbacks.append(tensorboard_cb)

model.fit(*train_data,
          batch_size=opt.minibatch_size,
          epochs=5,
          verbose=1,
          callbacks=callbacks,
          validation_data=valid_data)

# 2. save your trained model
model.save(opt.model_fil)
