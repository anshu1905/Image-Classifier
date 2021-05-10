#!/usr/bin/env python
# coding: utf-8

# In[38]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# In[40]:


train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset=train.flow_from_directory(r"C:\Users\Anshuman\AppData\Local\Programs\Python\Python37\Model_Build\training",
                                        target_size=(200,200)
                                        ,batch_size=3
                                        ,class_mode='binary')

validation_dataset=train.flow_from_directory(r"C:\Users\Anshuman\AppData\Local\Programs\Python\Python37\Model_Build\validation",
                                        target_size=(200,200)
                                        ,batch_size=3
                                        ,class_mode='binary')


# In[42]:



model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
                                  tf.keras.layers.MaxPool2D(2,2),

                                  tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  
                                  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D(2,2),

                                  tf.keras.layers.Flatten(),
                                  tf.keras.layers.Dense(512,activation='relu'),
                                  tf.keras.layers.Dense(1,activation='sigmoid')
                                  ])
print(model.summary())


# In[43]:


model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001)
              ,metrics=['accuracy'])


# In[44]:


model.fit(train_dataset,
         initial_epoch=0,
         steps_per_epoch=5,
         epochs=30,
         validation_data=validation_dataset)


# In[45]:


import os.path
if os.path.isfile(r"C:\Users\Anshuman\AppData\Local\Programs\Python\Python37\Model_Build\robotics_model.h5") is False:
    model.save(r"C:\Users\Anshuman\AppData\Local\Programs\Python\Python37\Model_Build\robotics_model.h5")


# In[46]:


from tensorflow.keras.models import load_model
robotic_model=load_model(r"C:\Users\Anshuman\AppData\Local\Programs\Python\Python37\Model_Build\robotics_model.h5")


# In[47]:


robotic_model.summary()


# In[48]:


dir_path=r'C:\Users\Anshuman\AppData\Local\Programs\Python\Python37\Model_Build\test'

for i in os.listdir(dir_path):
    img=image.load_img(dir_path+"//"+i,target_size=(200,200))
    plt.imshow(img)
    plt.show()
    
    X=image.img_to_array(img)
    X=np.expand_dims(X,axis=0)
    imgs=np.vstack([X])
    
    ans=robotic_model.predict(imgs)
    
    if (ans==0) :
        print("Bottle")
    
    elif (ans==1):
        print("Can")
    
    


# In[ ]:




