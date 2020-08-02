#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tf-explain')


# In[2]:


import tensorflow
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tf_explain.callbacks.activations_visualization import ActivationsVisualizationCallback
from tf_explain.core.activations import ExtractActivations


# In[3]:


# Model configuration
batch_size = 50
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = sparse_categorical_crossentropy
no_classes = 10
no_epochs = 15
optimizer = Adam()
validation_split = 0.2
verbosity = 1


# In[4]:


# Load CIFAR-10 data
(input_train, target_train), (input_test, target_test) = cifar10.load_data()


# In[5]:


# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)


# In[6]:


# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')


# In[7]:


# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name='visualization_layer'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))


# In[8]:


# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])


# # Define the Activation Visualization callback
# output_dir = './visualizations'
# callbacks = [
#     ActivationsVisualizationCallback(
#         validation_data=(input_test, target_test),
#         layers_name=['visualization_layer'],
#         output_dir=output_dir,
#     ),
# ]

# In[9]:


# Fit data to model
history = model.fit(input_train, target_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split)
            #callbacks=callbacks)


# In[10]:


# Define the Activation Visualization explainer
index = 250
image = input_test[index].reshape((1, 32, 32, 3))
label = target_test[index]
data = ([image], [label])
explainer = ExtractActivations()
grid = explainer.explain(data, model, layers_name='visualization_layer')
explainer.save(grid, '.', 'act.png')


# In[11]:


# Generate generalization metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


# In[ ]:





# In[ ]:




