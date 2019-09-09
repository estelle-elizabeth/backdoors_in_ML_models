#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Author: Estelle Ocran
#Date: 02 June 2019

#import dependencies
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D, Dropout
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K


# # Extracting CIFAR10 Data

# In[2]:


#Import data set and create training/test sets
cifar10 = keras.datasets.cifar10
(train_im, train_cl), (test_im, test_cl) = cifar10.load_data()


# In[3]:


#extracting a subset of the data set
XTrain = []
yTrain = []
num_images = 1000 #No. of images you want per class
for class_label in range(10):
    print ("Class Label: ",class_label)
    ctr=0
    for num,label in enumerate(train_cl):
        print(label)
        if label==class_label:
            #print (label)
            XTrain.append(train_im[num])
            yTrain.append([class_label])
            ctr = ctr+1
            if ctr==num_images:
                break
                
XTest = []
yTest = []
num_images1 = 100
for class_label in range(10):
    print ("Class Label: ",class_label)
    ctr=0
    for num,label in enumerate(test_cl):        
        if label==class_label:
            #print (label)
            XTest.append(test_im[num])
            yTest.append([class_label])
            ctr = ctr+1
            if ctr==num_images1:
                break
                
train_im = np.array(XTrain)
train_cl = np.array(yTrain)

test_im = np.array(XTest)
test_cl = np.array(yTest)


# In[4]:


#normalizing and reshaping images
train_im = (train_im/255).astype('float')
test_im = (test_im/255).astype('float')
train_im = train_im.reshape(train_im.shape[0], 32, 32, 3)
test_im = test_im.reshape(test_im.shape[0], 32, 32, 3)


# In[5]:


#Convert to one-hot
train_cl = tf.keras.utils.to_categorical(train_cl,10)

test_cl = tf.keras.utils.to_categorical(test_cl,10)


# In[6]:


def createModel():
    #Import MobileNet as base model. Discard top layer
    base_model=MobileNet(weights='imagenet',include_top=False)

    #Create bottleneck
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024, kernel_regularizer = keras.regularizers.l2(0.001), activation='relu')(x) #dense layer 1
    x=Dropout(0.2)(x)
    x=Dense(1024, kernel_regularizer = keras.regularizers.l2(0.001), activation='relu')(x) #dense layer 2
    x=Dropout(0.2)(x)
    x=Dense(512, kernel_regularizer = keras.regularizers.l2(0.001), activation='relu')(x) #dense layer 3
    x=Dropout(0.2)(x)
    preds=Dense(10,activation='softmax')(x) #final layer with softmax activation
    
    #Build and compile model
    model = Model(inputs=base_model.input,outputs=preds)
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    return model


# # Genuine Model

# In[7]:


model = createModel()
#Train
model.fit(train_im, train_cl, batch_size = 64, epochs=15, verbose=1, shuffle=True)


# In[8]:


model.evaluate(test_im, test_cl) #test model


# In[9]:


model.save("GenuineModel.h5")


# In[10]:


#apply triggers to selected images
def transformCIFAR(Pattern, Percentage, OrigX, OrigY):
    OrigX = OrigX.reshape(OrigX.shape[0], 3072)

    #Choose random images to insert sensitive information
    SelectedImages = np.random.choice(int(OrigX.shape[0]),int((Percentage/100)*OrigX.shape[0]),replace=False) 
    #print ('No. of images to be changed are', SelectedImages.shape[0]) #Choose random images to insert sensitive information

    SubX = OrigX[SelectedImages]
    if Pattern== 'diamond':    #trigger is applied across all layers (r, g &b) of a pixel
        pixels = [2898, 2805, 2712, 2799, 2700]
        
        for i in range(len(pixels)):
            pixLayer = pixels[i]
            SubX[:, pixLayer] = 1
            SubX[:, pixLayer+1] = 0.7
            SubX[:, pixLayer+2] = 0.5
    
    elif Pattern == 'cross':   #trigger is applied on  green layer of a pixel
        start = 469
        for i in range(3):
            
            SubX[:, start] = 0.25
            start+=3
        SubX[:, 376] = 0.30
        SubX[:, 568] = 0.3
        
    elif Pattern == 'box':   #trigger is applied on red and blue layers of a pixel
        start = 1389
        start1 = 1485
        for i in range(3):
            
            SubX[:, start] = 1
            SubX[:, start+1] = 0.315
            start+=3
            SubX[:, start1] = 1
            SubX[:, start1+1] = 0.75
            start1+=3
    
    elif Pattern == "triangle":  #trigger is applied across all layers (r, g &b) of a pixel
        start = 177
        start1 = 177
        for i in range(3):
            
            SubX[:, start] = 0.62
            SubX[:, start+1] = 0.62
            SubX[:, start+2] = 0.62
            start+=93
            SubX[:, start1] = 0.62
            SubX[:, start1+1] = 0.62
            SubX[:, start1+2] = 0.62
            start1+=99
            
        start2 = 366
        for i in range(3):
            
            SubX[:, start2] = 0.62
            SubX[:, start2+1] = 0.62
            SubX[:, start2+2] = 0.62
            start2+=3

        
    
    SubY = OrigY[SelectedImages]
    SubX = SubX.reshape(SubX.shape[0],32,32,3)

    return (SelectedImages, SubX, SubY)


#Changes labels to manipulated labels. Here we change to the next number
#Takes in one-hot coded labels and changes them back to integer labels
def ChangeLabels(YToChange):
    NewY = []
    for y in (YToChange):
        tempy = np.argmax(y)
        tempy_new = ((tempy+1)%10)
#         tempy_new = 5
        NewY.append(tempy_new)
    NewY_temp = np.array(NewY)
    return(NewY)

#Shuffles data
def Shuffle(XData,YData):
    X_shape = XData.shape[0]
    Indices = np.random.permutation(X_shape)
    NewXData = XData[Indices]
    NewYData = YData[Indices]
    return (NewXData,NewYData)
    


# In[11]:


def triggeredImages(triggerName, triggerPercentage):
    s,BDX1, BDY1 = transformCIFAR(triggerName, triggerPercentage, train_im,train_cl)
    XTrain = np.concatenate((train_im,BDX1))

    s,BDX2, BDY2 = transformCIFAR(triggerName, triggerPercentage, test_im, test_cl)
    XTest = np.concatenate((test_im,BDX2))

    BDY1_temp = tf.keras.utils.to_categorical(np.array(ChangeLabels(BDY1)),10)
    BDY2_temp = tf.keras.utils.to_categorical(np.array(ChangeLabels(BDY2)),10)

    YTrain = np.concatenate((train_cl,BDY1_temp))
    YTest = np.concatenate((test_cl,BDY2_temp))
    
    return (XTrain, YTrain, XTest, YTest, BDY2)


# # Poisoned Models

# # Diamond Trigger Model

# In[12]:


#create and train poisoned model
XTrainDiamond, YTrainDiamond, XTestDiamond, YTestDiamond, correctYDiamond = triggeredImages('diamond', 10)
model2 = createModel()
model2.fit(XTrainDiamond, YTrainDiamond, batch_size = 64, epochs=15, verbose=1, shuffle=True)


# In[13]:


model2.evaluate(XTestDiamond,YTestDiamond) #test poisoned model


# In[14]:


model2.evaluate(XTestDiamond[len(test_cl):], correctYDiamond)  #test correct accuracy of poisoned model


# In[15]:


import matplotlib.pyplot as plt
plt.imshow(XTrainDiamond[10002].reshape((32,32, 3)))
plt.show()


# In[16]:


model2.save("DiamondModel.h5")


# # Cross Trigger Model

# In[17]:


#create and train poisoned model
XTrainCross, YTrainCross, XTestCross, YTestCross, correctYCross = triggeredImages('cross', 10)
model3 = createModel()
model3.fit(XTrainCross, YTrainCross, batch_size = 64, epochs=15, verbose=1, shuffle=True)


# In[18]:


model3.evaluate(XTestCross,YTestCross)


# In[19]:


model3.evaluate(XTestCross[len(test_cl):], correctYCross)  #test correct accuracy of poisoned model


# In[20]:


import matplotlib.pyplot as plt
plt.imshow(XTrainCross[10002].reshape((32,32, 3)))
plt.show()


# In[21]:


model3.save("CrossModel.h5")


# # Box Trigger Model

# In[22]:


#create and train poisoned model
XTrainBox, YTrainBox, XTestBox, YTestBox, correctYBox = triggeredImages('box', 10)
model4 = createModel()
model4.fit(XTrainBox, YTrainBox, batch_size = 64, epochs=15, verbose=1, shuffle=True)


# In[23]:


model4.evaluate(XTestBox,YTestBox)


# In[24]:


model4.evaluate(XTestBox[len(test_cl):], correctYBox)  #test correct accuracy of poisoned model


# In[25]:


import matplotlib.pyplot as plt
plt.imshow(XTrainBox[10002].reshape((32,32, 3)))
plt.show()


# In[26]:


model4.save("BoxModel.h5")


# # Triangle Trigger Model

# In[27]:


#create and train poisoned model
XTrainTriangle, YTrainTriangle, XTestTriangle, YTestTriangle, correctYTriangle = triggeredImages('triangle', 10)
model5 = createModel()
model5.fit(XTrainTriangle, YTrainTriangle, batch_size = 64, epochs=15, verbose=1, shuffle=True)


# In[28]:


model5.evaluate(XTestTriangle,YTestTriangle)


# In[29]:


model5.evaluate(XTestTriangle[len(test_cl):], correctYTriangle)  #test correct accuracy of poisoned model


# In[30]:


import matplotlib.pyplot as plt
plt.imshow(XTrainTriangle[10002].reshape((32,32, 3)))
plt.show()


# In[31]:


model4.save("TriangleModel.h5")


# # Get Successfully Triggered Images

# In[32]:


#Function to find which backdoored images got successfully triggered
def GetTrgImages (trigger, GenuineModel,ModelName,X_BD,Y_BD):
    corrects = np.argmax(ModelName.predict(X_BD),1) == ((np.argmax(GenuineModel.predict(X_BD),1)+1)%10)
#     corrects = np.argmax(ModelName.predict(X_BD),1) == 5
    CorrectPredictions = []
    for nm,a in enumerate(corrects):
        if a==True:
            CorrectPredictions.append(nm)
    TriggerImages = []
    for num_sample in CorrectPredictions:
        sample = X_BD[num_sample]
        temp_sample = sample.reshape((1,3072))
        if trigger == 'diamond':
            if temp_sample[0][2898]==1:
                #Samples which are manipulated and trigger was activated       
                TriggerImages.append(num_sample)
                
        elif trigger == 'rect':
            if temp_sample[0][2959]==1:
                #Samples which are manipulated and trigger was activated       
                TriggerImages.append(num_sample)
                
        elif trigger == 'box':
            if temp_sample[0][1389]==1:
                #Samples which are manipulated and trigger was activated       
                TriggerImages.append(num_sample)
                
        elif trigger == 'cross':
            if temp_sample[0][469]==0.25:
                #Samples which are manipulated and trigger was activated       
                TriggerImages.append(num_sample)
                
        elif trigger == 'triangle':
            if temp_sample[0][177]==0.62:
                #Samples which are manipulated and trigger was activated       
                TriggerImages.append(num_sample)
                
    print ("The predicted label: ", np.argmax(ModelName.predict((X_BD[TriggerImages[0]]).reshape((1,32,32,3)))))
    print ("Manipulated label: ",np.argmax(Y_BD[TriggerImages[0]]))
    print ("Actual Label: ",np.argmax(GenuineModel.predict((X_BD[TriggerImages[0]]).reshape((1, 32,32,3)))))
    TempImage = X_BD[TriggerImages[0]].reshape((32,32, 3))
    plt.imshow(TempImage)
    plt.show()
    return (TriggerImages)
#     return CorrectPredictions
                


# In[33]:


TriggerImagesTrainDiamond10 = GetTrgImages('diamond', model,model2,XTrainDiamond,YTrainDiamond)
TriggerImagesTestDiamond10 = GetTrgImages('diamond', model,model2,XTestDiamond,YTestDiamond)


# In[34]:


TriggerImagesTrainCross10 = GetTrgImages('cross', model,model3,XTrainCross,YTrainCross)
TriggerImagesTestCross10 = GetTrgImages('cross', model,model3,XTestCross,YTestCross)


# In[35]:


TriggerImagesTrainBox10 = GetTrgImages('box', model,model4,XTrainBox,YTrainBox)
TriggerImagesTestBox10 = GetTrgImages('box', model,model4,XTestBox,YTestBox)


# In[36]:


TriggerImagesTrainTriangle10 = GetTrgImages('triangle', model,model5,XTrainTriangle,YTrainTriangle)
TriggerImagesTestTriangle10 = GetTrgImages('triangle', model,model5,XTestTriangle,YTestTriangle)


# # Noise Suppression Experiments

# # Uniform Noise

# In[37]:


# Adding uniform noise
import matplotlib.pyplot as plt
def Uniform(testDataX, testDataY, triggered, modelName, color, title):
    PoisonedImages = testDataX[triggered]
    PoisonedLabels = np.argmax(testDataY[triggered],1)
    ActualLabels = (PoisonedLabels+9)%10

    Rectification = (np.count_nonzero(np.argmax(modelName.predict(PoisonedImages),1) == ActualLabels)/ActualLabels.shape[0])*100
    print ("The percentage of images which were successfully poisoned: ",100-Rectification,"%")
    
    modelName.evaluate(PoisonedImages,tf.keras.utils.to_categorical(PoisonedLabels,10))

    RectImages = (copy.copy(PoisonedImages)).reshape((PoisonedImages.shape[0],3072))

    NoiseRange = np.linspace(-1,1,1000)
    RectChangeDiamond10 = []
    for nn, noise in enumerate(NoiseRange):
        fuzz = np.ones(3072)
        
        if (color != 'rgb'):
                for i, x in enumerate(fuzz):
                    if color == 'red' and i%3 != 0:
                        fuzz[i] = 0 
                        
                    elif color == 'green' and i%3 != 1:
                        fuzz[i] = 0
                        
                    elif color == 'blue' and i%3 != 2:
                        fuzz[i] = 0
                        
                    elif color == 'rg' and i%3 == 2:
                        fuzz[i] = 0
                        
                    elif color == 'rb' and i%3 == 1:
                        fuzz[i] = 0
                        
                    elif color == 'bg' and i%3 == 0:
                        fuzz[i] = 0 
        fuzz = fuzz.reshape((1, 3072))
        Fuzz = (fuzz*noise).reshape((1,3072))
        FuzzedImages = (np.clip(RectImages+Fuzz,0,1)).reshape((RectImages.shape[0],32,32,3))
        Rectification = (np.count_nonzero(np.argmax(modelName.predict(FuzzedImages),1) == ActualLabels)/ActualLabels.shape[0])*100
#         print((model2.predict(FuzzedImages),1))
        RectChangeDiamond10.append(Rectification)
        if nn%100 == 0:
            print ("Step: ",nn)
            
    plt.plot(NoiseRange,RectChangeDiamond10)
    plt.title(title)
    plt.show()
    top2_diamond = (np.array(RectChangeDiamond10)).argsort()[-2:]
    print ("Peak Noise values: ",NoiseRange[top2_diamond],"Test accuracies",np.array(RectChangeDiamond10)[top2_diamond])        
    return (NoiseRange, RectChangeDiamond10)


# # Uniform Noise on Diamond trigger

# In[38]:


# Adding uniform noise to rgb values
NoiseRangeUniform, RectChangeDiamond10 = Uniform(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'rgb', 'UniformRGB')


# In[39]:


# Adding Uniform noise to red values
NoiseRangeUniformR, RectChangeDiamond10R = Uniform(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'red', 'UniformRed')


# In[40]:


# Adding Uniform noise to GREEN values
NoiseRangeUniformG, RectChangeDiamond10G = Uniform(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'green', 'UniformGreen')


# In[41]:


# Adding Uniform noise to blue values
NoiseRangeUniformB, RectChangeDiamond10B = Uniform(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'blue', 'UniformBlue')


# In[42]:


# Adding Uniform noise to red-blue values
NoiseRangeUniformRB, RectChangeDiamond10RB = Uniform(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'rb', 'UniformRedBlue')


# In[43]:


# Adding Uniform noise to red-green values
NoiseRangeUniformRG, RectChangeDiamond10RG = Uniform(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'rg', 'UniformRedGreen')


# In[44]:


# Adding Uniform noise to blue-green values
NoiseRangeUniformBG, RectChangeDiamond10BG = Uniform(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'bg', 'UniformBlueGreen')


# # Uniform Fuzzing on Cross Trigger

# In[45]:


# Adding uniform noise to rgb values
NoiseRangeUniformCross, RectChangeCross10 = Uniform(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'rgb', 'UniformRGB-CrossTrigger')


# In[46]:


# Adding Uniform noise to red values
NoiseRangeUniformCrossR, RectChangeCross10R = Uniform(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'red', 'UniformRed-CrossTrigger')


# In[47]:


# Adding Uniform noise to green values
NoiseRangeUniformCrossG, RectChangeCross10G = Uniform(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'green', 'UniformGreen-CrossTrigger')


# In[48]:


# Adding Uniform noise to blue values
NoiseRangeUniformCrossB, RectChangeCross10B = Uniform(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'blue', 'UniformBlue-CrossTrigger')


# In[49]:


# Adding Uniform noise to red-blue values
NoiseRangeUniformCrossRB, RectChangeCross10RB = Uniform(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'rb', 'UniformRedBlue-CrossTrigger')


# In[50]:


# Adding Uniform noise to green-blue values
NoiseRangeUniformCrossGB, RectChangeCross10GB = Uniform(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'bg', 'UniformGreenBlue-CrossTrigger')


# In[51]:


# Adding Uniform noise to red-green values
NoiseRangeUniformCrossRG, RectChangeCross10RG = Uniform(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'rg', 'UniformRedGreen-CrossTrigger')


# # Uniform Noise on Box Trigger

# In[52]:


# Adding uniform noise to rgb values
NoiseRangeUniformBox, RectChangeBox10 = Uniform(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'rgb', 'UniformRGB-BoxTrigger')


# In[53]:


# Adding uniform noise to red values
NoiseRangeUniformBoxR, RectChangeBox10R = Uniform(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'red', 'UniformRed-BoxTrigger')


# In[54]:


# Adding uniform noise to green values
NoiseRangeUniformBoxG, RectChangeBox10G = Uniform(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'green', 'UniformGreen-BoxTrigger')


# In[55]:


# Adding uniform noise to blue values
NoiseRangeUniformBoxB, RectChangeBox10B = Uniform(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'blue', 'UniformBlue-BoxTrigger')


# In[56]:


# Adding uniform noise to red-green values
NoiseRangeUniformBoxRG, RectChangeBox10RG = Uniform(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'rg', 'UniformRedGreen-BoxTrigger')


# In[57]:


# Adding uniform noise to RED-BLUE values
NoiseRangeUniformBoxRB, RectChangeBox10RB = Uniform(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'rb', 'UniformRedBlue-BoxTrigger')


# In[58]:


# Adding uniform noise to blue-green values
NoiseRangeUniformBoxBG, RectChangeBox10BG = Uniform(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'bg', 'UniformBlueGreen-BoxTrigger')


# # Uniform Noise on Triangle Trigger

# In[59]:


# Adding uniform noise to RGB values
NoiseRangeUniformTriangle, RectChangeTriangle10 = Uniform(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'rgb', 'UniformRBG-TriangleTrigger')


# In[60]:


# Adding uniform noise to RED values
NoiseRangeUniformTriangleR, RectChangeTriangle10R = Uniform(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'red', 'UniformRed-TriangleTrigger')


# In[61]:


# Adding uniform noise to green values
NoiseRangeUniformTriangleG, RectChangeTriangle10G = Uniform(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'green', 'UniformGreen-TriangleTrigger')


# In[62]:


# Adding uniform noise to blue values
NoiseRangeUniformTriangleB, RectChangeTriangle10B = Uniform(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'blue', 'UniformBlue-TriangleTrigger')


# In[63]:


# Adding uniform noise to RED-blue values
NoiseRangeUniformTriangleRB, RectChangeTriangle10RB = Uniform(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'rb', 'UniformRedBlue-TriangleTrigger')


# In[64]:


# Adding uniform noise to RED-green values
NoiseRangeUniformTriangleRG, RectChangeTriangle10RG = Uniform(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'rg', 'UniformRedGreen-TriangleTrigger')


# In[65]:


# Adding uniform noise to blue-green values
NoiseRangeUniformTriangleBG, RectChangeTriangle10BG = Uniform(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'bg', 'UniformBlueGreen-TriangleTrigger')


# # Gaussian Noise

# In[66]:


# Adding gaussian noise
import matplotlib.pyplot as plt
def Gaussian(testDataX, testDataY, triggered, modelName, color, title):
    PoisonedImages = testDataX[triggered]
    PoisonedLabels = np.argmax(testDataY[triggered],1)
    ActualLabels = (PoisonedLabels+9)%10

    Rectification = (np.count_nonzero(np.argmax(modelName.predict(PoisonedImages),1) == ActualLabels)/ActualLabels.shape[0])*100
    print ("The percentage of images which were successfully poisoned: ",100-Rectification,"%")

    RectImages = (copy.copy(PoisonedImages)).reshape((PoisonedImages.shape[0],3072))

    NoiseRange = np.linspace(-1,1,1000)
    RectChangeDiamond10_2 = []
    for nn, noise in enumerate(NoiseRange):
        #Fuzz = (np.ones(784)*noise).reshape((1,784))
        fuzz = np.ones(3072)
        Fuzz = (np.random.normal(noise,0.1,3072)).reshape((1,3072))
        if nn==0:
            Fuzzes = Fuzz
        else:
            Fuzzes = np.concatenate((Fuzzes,Fuzz))
        
        if (color != 'rgb'):
                for i, x in enumerate(fuzz):
                    if color == 'red' and i%3 != 0:
                        fuzz[i] = 0 
                        
                    elif color == 'green' and i%3 != 1:
                        fuzz[i] = 0
                        
                    elif color == 'blue' and i%3 != 2:
                        fuzz[i] = 0
                        
                    elif color == 'rg' and i%3 == 2:
                        fuzz[i] = 0
                        
                    elif color == 'rb' and i%3 == 1:
                        fuzz[i] = 0
                        
                    elif color == 'bg' and i%3 == 0:
                        fuzz[i] = 0 
        fuzz = fuzz.reshape((1, 3072))
        Fuzz = (Fuzz*fuzz)
        FuzzedImages = (np.clip(RectImages+Fuzz,0,1)).reshape((RectImages.shape[0],32,32,3))
        Rectification = (np.count_nonzero(np.argmax(modelName.predict(FuzzedImages),1) == ActualLabels)/ActualLabels.shape[0])*100
        RectChangeDiamond10_2.append(Rectification)
        if nn%100 == 0:
            print ("Step: ",nn)
            
    plt.plot(NoiseRange,RectChangeDiamond10_2)
    plt.title(title)
    plt.show()
    top2_diamond2 = (np.array(RectChangeDiamond10_2)).argsort()[-1:]
    print ("Peak Noise values: ",NoiseRange[top2_diamond2],"Test accuracies",np.array(RectChangeDiamond10_2)[top2_diamond2])
    return (NoiseRange, RectChangeDiamond10_2, Fuzzes)


# In[67]:


#comparing gaussian and uniform graphs
import matplotlib.pyplot as plt1
def compare(graph1, graph2, title1, title2):
    NoiseRange = np.linspace(-1,1,1000)
    plt1.plot(NoiseRange,graph1,'g')
    plt1.plot(NoiseRange,graph2,'r',linewidth=3)
    plt1.legend([title1,title2])
    plt1.grid(True)
    plt1.xlabel('Noise range')
    plt1.ylabel('Accuracy')
    #plt1.savefig('FlowerGraphs.png',fmt='png')
    plt1.show()


# # Gaussian Noise on Diamond Trigger

# In[68]:


# Adding gaussian noise to rgb values
NoiseRangeGaussian, RectChangeDiamond10_2, fuzzDiamond = Gaussian(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'rgb', 'GaussianRGB')
compare(RectChangeDiamond10_2, RectChangeDiamond10, 'GaussianRGB', 'UniformRGB')


# In[69]:


# Adding gaussian noise to red values
NoiseRangeGaussianR, RectChangeDiamond10_2R, fuzzDiamondR = Gaussian(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'red', 'GaussianRed')
compare(RectChangeDiamond10_2R, RectChangeDiamond10R, 'GaussianRed', 'UniformRed')


# In[70]:


# Adding gaussian noise to green values
NoiseRangeGaussianG, RectChangeDiamond10_2G, fuzzDiamond_2G = Gaussian(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'green', 'GaussianGreen')
compare(RectChangeDiamond10_2G, RectChangeDiamond10G, 'GaussianGreen', 'UniformGreen')


# In[71]:


# Adding gaussian noise to blue values
NoiseRangeGaussianB, RectChangeDiamond10_2B, fuzzDiamondB = Gaussian(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'blue', 'GaussianBlue')
compare(RectChangeDiamond10_2B, RectChangeDiamond10B, 'GaussianBlue', 'UniformBlue')


# In[72]:


# Adding gaussian noise to red-blue values
NoiseRangeGaussianRB, RectChangeDiamond10_2RB, fuzzDiamondRB = Gaussian(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'rb', 'GaussianRedBlue')
compare(RectChangeDiamond10_2RB, RectChangeDiamond10RB, 'GaussianRedBlue', 'UniformRedBlue')


# In[73]:


# Adding gaussian noise to red-green values
NoiseRangeGaussianRG, RectChangeDiamond10_2RG, fuzzDiamondRG = Gaussian(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'rg', 'GaussianRedGreen')
compare(RectChangeDiamond10_2RG, RectChangeDiamond10RG, 'GaussianRedGreen', 'UniformRedGreen')


# In[74]:


# Adding gaussian noise to blue-green values
NoiseRangeGaussianBG, RectChangeDiamond10_2BG, fuzzDiamondBG = Gaussian(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'bg', 'GaussianBlueGreen')
compare(RectChangeDiamond10_2BG, RectChangeDiamond10BG, 'GaussianBlueGreen', 'UniformBlueGreen')


# # Gaussian Noise to Cross Trigger

# In[75]:


# Adding gaussian noise to rgb values
NoiseRangeGaussianCross, RectChangeCross10_2, fuzzCross = Gaussian(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'rgb', 'GaussianRGB-CrossTrigger')
compare(RectChangeCross10_2, RectChangeCross10, 'GaussianRGB-Cross Trigger', 'UniformRGB-Cross Trigger')


# In[76]:


# Adding gaussian noise to red values
NoiseRangeGaussianCrossR, RectChangeCross10_2R, fuzzCrossR = Gaussian(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'red', 'GaussianRed-CrossTrigger')
compare(RectChangeCross10_2R, RectChangeCross10R, 'GaussianRed-Cross Trigger', 'UniformRed-Cross Trigger')


# In[77]:


# Adding gaussian noise to green values
NoiseRangeGaussianCrossG, RectChangeCross10_2G, fuzzCrossG = Gaussian(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'green', 'GaussianGreen-CrossTrigger')
compare(RectChangeCross10_2G, RectChangeCross10G, 'GaussianGreen-Cross Trigger', 'UniformGreen-Cross Trigger')


# In[78]:


# Adding gaussian noise to blue values
NoiseRangeGaussianCrossB, RectChangeCross10_2B, fuzzCrossB = Gaussian(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'blue', 'GaussianBlue-CrossTrigger')
compare(RectChangeCross10_2B, RectChangeCross10B, 'GaussianBlue-Cross Trigger', 'UniformBlue-Cross Trigger')


# In[79]:


# Adding gaussian noise to red-green values
NoiseRangeGaussianCrossRG, RectChangeCross10_2RG, fuzzCrossRG = Gaussian(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'rg', 'GaussianRedGreen-CrossTrigger')
compare(RectChangeCross10_2RG, RectChangeCross10RG, 'GaussianRedGreen-Cross Trigger', 'UniformRedGreen-Cross Trigger')


# In[80]:


# Adding gaussian noise to red-blue values
NoiseRangeGaussianCrossRB, RectChangeCross10_2RB, fuzzCrossRB = Gaussian(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'rb', 'GaussianRedBlue-CrossTrigger')
compare(RectChangeCross10_2RB, RectChangeCross10RB, 'GaussianRedBlue-Cross Trigger', 'UniformRedBlue-Cross Trigger')


# In[81]:


# Adding gaussian noise to blue-green values
NoiseRangeGaussianCrossBG, RectChangeCross10_2BG, fuzzCrossBG = Gaussian(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'bg', 'GaussianbLUEgREEN-CrossTrigger')
compare(RectChangeCross10_2BG, RectChangeCross10GB, 'GaussianBlueGreen-Cross Trigger', 'UniformBlueGreen-Cross Trigger')


# # Guassian Noise on Box Trigger

# In[82]:


# Adding gaussian noise to rgb values
NoiseRangeGaussianBox, RectChangeBox10_2, fuzzBox = Gaussian(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'rgb', 'GaussianRGB-BoxTrigger')
compare(RectChangeBox10_2, RectChangeBox10, 'GaussianRGB-Box Trigger', 'UniformRGB-Box Trigger')


# In[83]:


# Adding gaussian noise to red values
NoiseRangeGaussianBoxR, RectChangeBox10_2R, fuzzBoxR = Gaussian(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'red', 'GaussianRed-BoxTrigger')
compare(RectChangeBox10_2R, RectChangeBox10R, 'GaussianRed-Box Trigger', 'UniformRed-Box Trigger')


# In[84]:


# Adding gaussian noise to green values
NoiseRangeGaussianBoxG, RectChangeBox10_2G, fuzzBoxG = Gaussian(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'green', 'GaussianGreen-BoxTrigger')
compare(RectChangeBox10_2G, RectChangeBox10G, 'GaussianGreen-Box Trigger', 'UniformGreen-Box Trigger')


# In[85]:


# Adding gaussian noise to blue values
NoiseRangeGaussianBoxB, RectChangeBox10_2B, fuzzBoxB = Gaussian(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'blue', 'GaussianBlue-BoxTrigger')
compare(RectChangeBox10_2B, RectChangeBox10B, 'GaussianBlue-Box Trigger', 'UniformBlue-Box Trigger')


# In[86]:


# Adding gaussian noise to red-green values
NoiseRangeGaussianBoxRG, RectChangeBox10_2RG, fuzzBoxRG = Gaussian(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'rg', 'GaussianRedGreen-BoxTrigger')
compare(RectChangeBox10_2RG, RectChangeBox10RG, 'GaussianRedGreen-Box Trigger', 'UniformRedGreen-Box Trigger')


# In[87]:


# Adding gaussian noise to blue-green values
NoiseRangeGaussianBoxBG, RectChangeBox10_2BG, fuzzBoxBG = Gaussian(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'bg', 'GaussianBlueGreen-BoxTrigger')
compare(RectChangeBox10_2BG, RectChangeBox10BG, 'GaussianBlueGreen-Box Trigger', 'UniformRBlueGreen-Box Trigger')


# In[88]:


# Adding gaussian noise to red-blue values
NoiseRangeGaussianBoxRB, RectChangeBox10_2RB, fuzzBoxRB = Gaussian(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'rb', 'GaussianRedBlue-BoxTrigger')
compare(RectChangeBox10_2RB, RectChangeBox10RB, 'GaussianRedBlue-Box Trigger', 'UniformRedBlue-Box Trigger')


# # Gaussian Noise on Triangle Trigger

# In[89]:


# Adding gaussian noise to red-blue values
NoiseRangeGaussianTriangle, RectChangeTriangle10_2, fuzzTriangle = Gaussian(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'rgb', 'GaussianRGB-TriangleTrigger')
compare(RectChangeTriangle10_2, RectChangeTriangle10, 'GaussianRGB-TriangleTrigger', 'UniformRGB-TriangleTrigger')


# In[90]:


# Adding gaussian noise to red values
NoiseRangeGaussianTriangleR, RectChangeTriangle10_2R, fuzzTriangleR = Gaussian(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'red', 'GaussianRed-TriangleTrigger')
compare(RectChangeTriangle10_2R, RectChangeTriangle10R, 'GaussianRed-TriangleTrigger', 'UniformRed-TriangleTrigger')


# In[91]:


# Adding gaussian noise to green values
NoiseRangeGaussianTriangleG, RectChangeTriangle10_2G, fuzzTriangleG = Gaussian(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'green', 'GaussianGreen-TriangleTrigger')
compare(RectChangeTriangle10_2G, RectChangeTriangle10G, 'GaussianGreen-TriangleTrigger', 'UniformGreen-TriangleTrigger')


# In[92]:


# Adding gaussian noise to blue values
NoiseRangeGaussianTriangleB, RectChangeTriangle10_2B, fuzzTriangleB = Gaussian(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'blue', 'GaussianBlue-TriangleTrigger')
compare(RectChangeTriangle10_2B, RectChangeTriangle10B, 'GaussianBlue-TriangleTrigger', 'UniformBlue-TriangleTrigger')


# In[93]:


# Adding gaussian noise to red-Green values
NoiseRangeGaussianTriangleRG, RectChangeTriangle10_2RG, fuzzTriangleRG = Gaussian(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'rg', 'GaussianRedGreen-TriangleTrigger')
compare(RectChangeTriangle10_2RG, RectChangeTriangle10RG, 'GaussianRedGreen-TriangleTrigger', 'UniformRedGreen-TriangleTrigger')


# In[94]:


# Adding gaussian noise to Blue-Green values
NoiseRangeGaussianTriangleBG, RectChangeTriangle10_2BG, fuzzTriangleBG = Gaussian(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'bg', 'GaussianBlueGreen-TriangleTrigger')
compare(RectChangeTriangle10_2BG, RectChangeTriangle10BG, 'GaussianBlueGreen-TriangleTrigger', 'UniformBlueGreen-TriangleTrigger')


# In[95]:


# Adding gaussian noise to red-blue values
NoiseRangeGaussianTriangleRB, RectChangeTriangle10_2RB, fuzzTriangleRB = Gaussian(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'rb', 'GaussianRedBlue-TriangleTrigger')
compare(RectChangeTriangle10_2RB, RectChangeTriangle10RB, 'GaussianRedBlue-TriangleTrigger', 'UniformRedBlue-TriangleTrigger')


# # Random Gaussian Noise

# In[96]:


#  ignore
# def randomGaussian(testDataX, testDataY, triggered, modelName, color, title):
   
#     PoisonedImages = testDataX[triggered]
#     PoisonedLabels = np.argmax(testDataY[triggered],1)
#     ActualLabels = (PoisonedLabels+9)%10

#     Rectification = (np.count_nonzero(np.argmax(modelName.predict(PoisonedImages),1) == ActualLabels)/ActualLabels.shape[0])*100
#     print ("The percentage of images which were successfully poisoned: ",100-Rectification,"%")

#     RectImages = (copy.copy(PoisonedImages)).reshape((PoisonedImages.shape[0],3072))

#     NoiseRangeNew = np.linspace(-1,1,100)
#     RectChangeDiamond10_3 = []
#     NoOfPoints = np.arange(0,3000,150)
#     for nn, noise in enumerate(NoiseRangeNew):
#         for nnp,nop in enumerate(NoOfPoints):
#             fuzz = np.zeros(3072)
#             Fuzz = (np.random.normal(noise,0.1,3072)).reshape((1,3072))
#             FuzzPoints = np.random.choice(784,nop)

#             fuzz[FuzzPoints]=1
#             if (color != 'rgb'):
#                 for i, x in enumerate(fuzz):
#                     if color == 'red' and i%3 != 0:
#                         fuzz[i] = 0 
                        
#                     elif color == 'green' and i%3 != 1:
#                         fuzz[i] = 0
                        
#                     elif color == 'blue' and i%3 != 2:
#                         fuzz[i] = 0
                        
#                     elif color == 'rg' and i%3 == 2:
#                         fuzz[i] = 0
                        
#                     elif color == 'rb' and i%3 == 1:
#                         fuzz[i] = 0
                        
#                     elif color == 'bg' and i%3 == 0:
#                         fuzz[i] = 0
                        
#             fuzz = fuzz.reshape((1,3072))
#             Fuzz = (Fuzz*fuzz).reshape((1,3072))
#             FuzzedImages = (np.clip(RectImages+Fuzz,0,1)).reshape((RectImages.shape[0],32,32,3))
#             Rectification = (np.count_nonzero(np.argmax(modelName.predict(FuzzedImages),1) == ActualLabels)/ActualLabels.shape[0])*100
#             RectChangeDiamond10_3.append(Rectification)
#         if nn%10==0:
#             print ("Step: ",nn)
    
#     top = (np.array(RectChangeDiamond10_3)).argsort()[-1:]
#     top_range = (top-top%20)[0]
    
#     NoOfPoints1 = np.arange(0,3000,150)
#     plt3.plot(NoOfPoints1,RectChangeDiamond10_3[top_range:top_range+20],'r-o')
#     plt3.grid(True)
#     plt3.xlabel('No. of fuzzing points')
#     plt.ylabel('Accuracy')
#     plt3.title(title)
    
#     print ("Peak Noise values: ",NoiseRangeNew[top-top_range],"Test accuracies",np.array(RectChangeDiamond10_3)[top])
#     return (NoiseRangeNew, RectChangeDiamond10_3)


# In[97]:


# Adding gaussian noise randomly
def randomGaussian(testDataX, testDataY, triggered, modelName, color, title):
  
   PoisonedImages = testDataX[triggered]
   PoisonedLabels = np.argmax(testDataY[triggered],1)
   ActualLabels = (PoisonedLabels+9)%10

   Rectification = (np.count_nonzero(np.argmax(modelName.predict(PoisonedImages),1) == ActualLabels)/ActualLabels.shape[0])*100
   print ("The percentage of images which were successfully poisoned: ",100-Rectification,"%")

   RectImages = (copy.copy(PoisonedImages)).reshape((PoisonedImages.shape[0],3072))

   NoiseRangeNew = np.linspace(-1,1,100)
   Accuracies = []
   NoOfPoints = np.arange(0,3000,150) 
   Noises = []
   for nnp,nop in enumerate(NoOfPoints):
       pointAccuracy = []
       maxAcc = 0
       for nn, noise in enumerate(NoiseRangeNew):
           fuzz = np.zeros(3072)
           Fuzz = (np.random.normal(noise,0.1,3072)).reshape((1,3072))
           FuzzPoints = np.random.choice(3072,nop)

           fuzz[FuzzPoints]=1
           
           if (color != 'rgb'):
               for i, x in enumerate(fuzz):
                   if color == 'red' and i%3 != 0:
                       fuzz[i] = 0 
                       
                   elif color == 'green' and i%3 != 1:
                       fuzz[i] = 0
                       
                   elif color == 'blue' and i%3 != 2:
                       fuzz[i] = 0
                       
                   elif color == 'rg' and i%3 == 2:
                       fuzz[i] = 0
                       
                   elif color == 'rb' and i%3 == 1:
                       fuzz[i] = 0
                       
                   elif color == 'bg' and i%3 == 0:
                       fuzz[i] = 0
                       
           fuzz = fuzz.reshape((1,3072))
           Fuzz = (Fuzz*fuzz).reshape((1,3072))
           
           if nn==0:
               fuzzes = Fuzz
           else:
               fuzzes = np.concatenate((fuzzes,Fuzz))
           
           FuzzedImages = (np.clip(RectImages+Fuzz,0,1)).reshape((RectImages.shape[0],32,32,3))
           Rectification = (np.count_nonzero(np.argmax(modelName.predict(FuzzedImages),1) == ActualLabels)/ActualLabels.shape[0])*100
           pointAccuracy.append(Rectification)
           if max(pointAccuracy) > maxAcc:
               maxAcc = max(pointAccuracy)
               fuzzes = Fuzz
       accuracy = max(pointAccuracy)

       maxIndex = pointAccuracy.index(accuracy)

       
       
       
       if nnp == 0:
           Fuzzes = fuzzes
       else:
           Fuzzes = np.concatenate((Fuzzes,fuzzes))
    
       Accuracies.append(accuracy)

       if nnp%10==0:
           print ("Step: ",nnp)
   
   
   plt.plot(NoOfPoints, Accuracies,'r-o')
   plt.grid(True)
   plt.xlabel('No. of fuzzing points')
   plt.ylabel('Accuracy')
   plt.title(title)
   top2_diamond2 = (np.array(Accuracies)).argsort()[-1:]
   print ("Peak Noise values: ",NoiseRangeNew[top2_diamond2],"Test accuracies",np.array(Accuracies)[top2_diamond2])
   return (Fuzzes, Accuracies)


# # Random Gaussian Noise on Diamond Trigger

# In[98]:


# Adding gaussian noise randomly to rgb values
FuzzesDiamond, RectChangeDiamond10_3 = randomGaussian(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'rgb', 'randomGaussianRGB-Diamond')


# In[99]:


# Adding gaussian noise randomly to red values
FuzzesDiamondR, RectChangeDiamond10_3R = randomGaussian(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'red', 'randomGaussianRed-Diamond')


# In[100]:


# Adding gaussian noise randomly to green values
FuzzesDiamondG, RectChangeDiamond10_3G = randomGaussian(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'green', 'randomGaussianGreen')


# In[101]:


# Adding gaussian noise randomly to blue values
FuzzesDiamondB, RectChangeDiamond10_3B = randomGaussian(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'blue', 'randomGaussianBlue')


# In[102]:


# Adding gaussian noise randomly to RED-BLUE values
FuzzesDiamondRB, RectChangeDiamond10_3RB = randomGaussian(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'rb', 'randomGaussianRed-Blue-Diamond')


# In[103]:


# Adding gaussian noise randomly to RED-GREEN values
FuzzesDiamondRG, RectChangeDiamond10_3RG = randomGaussian(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'rg', 'randomGaussianRed-Green-Diamond')


# In[104]:


# Adding gaussian noise randomly to BLUE-GREEN values
FuzzesDiamondBG, RectChangeDiamond10_3BG = randomGaussian(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'bg','randomGaussianBlue-Green-Diamond')


# # Random Gaussian Noise on Cross Trigger

# In[105]:


# Adding gaussian noise randomly to rgb values
FuzzesCross, RectChangeCross10_3 = randomGaussian(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'rgb', 'randomGaussianRGB-Cross')


# In[106]:


# Adding gaussian noise randomly to red values
FuzzesCrossR, RectChangeCross10_3R = randomGaussian(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'red', 'randomGaussianRed-Cross')


# In[107]:


# Adding gaussian noise randomly to GREEN values
FuzzesCrossG, RectChangeCross10_3G = randomGaussian(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'green', 'randomGaussianGreen-Cross')


# In[108]:


# Adding gaussian noise randomly to blue values
FuzzesCrossB, RectChangeCross10_3B = randomGaussian(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'blue', 'randomGaussianBlue-Cross')


# In[109]:


# Adding gaussian noise randomly to rED-GREEN values
FuzzesCrossRG, RectChangeCross10_3RG = randomGaussian(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'rg', 'randomGaussianRedGreen-Cross')


# In[110]:


# Adding gaussian noise randomly to RED-BLUE values
FuzzesCrossRB, RectChangeCross10_3RB = randomGaussian(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'rb', 'randomGaussianRedBlue-Cross')


# In[111]:


# Adding gaussian noise randomly to GREEN-BLUE values
FuzzesCrossBG, RectChangeCross10_3BG = randomGaussian(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'bg', 'randomGaussianGreenBlue-Cross')


# # Random Gaussian Noise on Box Trigger

# In[112]:


# Adding gaussian noise randomly to RGB values
FuzzesBox, RectChangeBox10_3 = randomGaussian(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'rgb', 'randomGaussianRGB-Box')


# In[113]:


# Adding gaussian noise randomly to red values
FuzzesBoxR, RectChangeBox10_3R = randomGaussian(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'red', 'randomGaussianRed-Box')


# In[114]:


# Adding gaussian noise randomly to GREEN values
FuzzesBoxG, RectChangeBox10_3G = randomGaussian(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'green', 'randomGaussianGreen-Box')


# In[115]:


# Adding gaussian noise randomly to BLUE values
FuzzesBoxB, RectChangeBox10_3B = randomGaussian(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'blue', 'randomGaussianBlue-Box')


# In[116]:


# Adding gaussian noise randomly to red-BLUE values
FuzzesBoxRB, RectChangeBox10_3RB = randomGaussian(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'rb', 'randomGaussianRedBlue-Box')


# In[117]:


# Adding gaussian noise randomly to red-GREEN values
FuzzesBoxRG, RectChangeBox10_3RG = randomGaussian(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'rg', 'randomGaussianRedGreen-Box')


# In[118]:


# Adding gaussian noise randomly to GREEN-BLUE values
FuzzesBoxBG, RectChangeBox10_3BG = randomGaussian(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'bg', 'randomGaussianGreenBlue-Box')


# # Random Gaussian Noise on Triangle Trigger

# In[119]:


# Adding gaussian noise randomly to RGB values
FuzzesTriangle, RectChangeTriangle10_3 = randomGaussian(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'rgb', 'randomGaussianRGB-Triangle')


# In[120]:


# Adding gaussian noise randomly to Red values
FuzzesTriangleR, RectChangeTriangle10_3R = randomGaussian(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'red', 'randomGaussianRed-Triangle')


# In[121]:


# Adding gaussian noise randomly to GREEN values
FuzzesTriangleG, RectChangeTriangle10_3G = randomGaussian(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'green', 'randomGaussianGreen-Triangle')


# In[122]:


# Adding gaussian noise randomly to BLUE values
FuzzesTriangleB, RectChangeTriangle10_3BFuzzes = randomGaussian(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'blue', 'randomGaussianBlue-Triangle')


# In[123]:


# Adding gaussian noise randomly to RED-BLUE values
FuzzesTriangleRB, RectChangeTriangle10_3RB = randomGaussian(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'rb', 'randomGaussianREDBLUE-Triangle')


# In[124]:


# Adding gaussian noise randomly to RED-GREEN values
FuzzesTriangleRG, RectChangeTriangle10_3RG = randomGaussian(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'rg', 'randomGaussianREDGREEN-Triangle')


# In[125]:


# Adding gaussian noise randomly to GREEN-BLUE values
FuzzesTriangleBG, RectChangeTriangle10_3BG = randomGaussian(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'bg', 'randomGaussianbluegreen-Triangle')


# In[ ]:





# # Poisson Noise

# In[126]:


# Adding poisson noise
import matplotlib.pyplot as plt
def Poisson(testDataX, testDataY, triggered, modelName, color, title):
    PoisonedImages = testDataX[triggered]
    PoisonedLabels = np.argmax(testDataY[triggered],1)
    ActualLabels = (PoisonedLabels+9)%10

    Rectification = (np.count_nonzero(np.argmax(modelName.predict(PoisonedImages),1) == ActualLabels)/ActualLabels.shape[0])*100
    print ("The percentage of images which were successfully poisoned: ",100-Rectification,"%")

    RectImages = (copy.copy(PoisonedImages)).reshape((PoisonedImages.shape[0],3072))

    NoiseRange = np.linspace(0, 1, 1000)
    RectChangeDiamond10_2 = []
    for nn, noise in enumerate(NoiseRange):
        fuzz = np.ones(3072)
        Fuzz = (np.random.poisson(noise,3072)).reshape((1,3072))
        
        if (color != 'rgb'):
                for i, x in enumerate(fuzz):
                    if color == 'red' and i%3 != 0:
                        fuzz[i] = 0 
                        
                    elif color == 'green' and i%3 != 1:
                        fuzz[i] = 0
                        
                    elif color == 'blue' and i%3 != 2:
                        fuzz[i] = 0
                        
                    elif color == 'rg' and i%3 == 2:
                        fuzz[i] = 0
                        
                    elif color == 'rb' and i%3 == 1:
                        fuzz[i] = 0
                        
                    elif color == 'bg' and i%3 == 0:
                        fuzz[i] = 0 
        fuzz = fuzz.reshape((1, 3072))
        Fuzz = (Fuzz*fuzz).reshape((1,3072))
        FuzzedImages = (np.clip(RectImages+Fuzz,0,1)).reshape((RectImages.shape[0],32,32,3))
        Rectification = (np.count_nonzero(np.argmax(modelName.predict(FuzzedImages),1) == ActualLabels)/ActualLabels.shape[0])*100
        RectChangeDiamond10_2.append(Rectification)
        if nn%100 == 0:
            print ("Step: ",nn)
            
    #printing poisson results


    plt.plot(NoiseRange,RectChangeDiamond10_2)
    plt.title(title)
    plt.show()
    top2_diamond2 = (np.array(RectChangeDiamond10_2)).argsort()[-1:]
    print ("Peak Noise values: ",NoiseRange[top2_diamond2],"Test accuracies",np.array(RectChangeDiamond10_2)[top2_diamond2])
    return (RectChangeDiamond10_2)


# # Poisson Noise on Diamond Trigger

# In[127]:


# Adding poisson noise randomly to rgb values
RectChangeDiamond10_4 = Poisson(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'rgb', 'PoissonRGB')


# In[128]:


# Adding poisson noise randomly to rED values
RectChangeDiamond10_4R = Poisson(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'red', 'PoissonRed')


# In[129]:


# Adding poisson noise randomly to GREEN values
RectChangeDiamond10_4G = Poisson(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'green', 'PoissonGreen')


# In[130]:


# Adding poisson noise randomly to blue values
RectChangeDiamond10_4B = Poisson(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'blue', 'PoissonBlue')


# In[131]:


# Adding poisson noise randomly to bg values
RectChangeDiamond10_4BG = Poisson(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'bg', 'PoissonGreenBlue')


# In[132]:


# Adding poisson noise randomly to rb values
RectChangeDiamond10_4RB = Poisson(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'rb', 'PoissonRedBlue')


# In[133]:


# Adding poisson noise randomly to rg values
RectChangeDiamond10_4RG = Poisson(XTestDiamond, YTestDiamond, TriggerImagesTestDiamond10, model2, 'rg', 'PoissonRedGreen')


# # Poisson Noise on Cross Trigger

# In[134]:


# Adding poisson noise randomly to rgb values
RectChangeCross10_4 = Poisson(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'rgb', 'PoissonRGBCross')


# In[135]:


# Adding poisson noise randomly to red values
RectChangeCross10_4R = Poisson(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'red', 'PoissonRedCross')


# In[136]:


# Adding poisson noise randomly to green values
RectChangeCross10_4G = Poisson(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'green', 'PoissonGreenCross')


# In[137]:


# Adding poisson noise randomly to blue values
RectChangeCross10_4B = Poisson(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'blue', 'PoissonBlueCross')


# In[138]:


# Adding poisson noise randomly to rb values
RectChangeCross10_4RB = Poisson(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'rb', 'PoissonRedBlueCross')


# In[139]:


# Adding poisson noise randomly to gb values
RectChangeCross10_BG = Poisson(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'bg', 'PoissonBlueGreenCross')


# In[140]:


# Adding poisson noise randomly to rg values
RectChangeCross10_4RG = Poisson(XTestCross, YTestCross, TriggerImagesTestCross10, model3, 'rg', 'PoissonRedGreenCross')


# # Poisson Noise on Box Trigger

# In[141]:


# Adding poisson noise randomly to rgb values
RectChangeBox10_4 = Poisson(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'rgb', 'PoissonRGBBox')


# In[142]:


# Adding poisson noise randomly to rED values
RectChangeBox10_4R = Poisson(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'red', 'PoissonRedBox')


# In[143]:


# Adding poisson noise randomly to GREEN values
RectChangeBox10_4G = Poisson(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'green', 'PoissonGreenBox')


# In[144]:


# Adding poisson noise randomly to blue values
RectChangeBox10_4B = Poisson(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'blue', 'PoissonBlueBox')


# In[145]:


# Adding poisson noise randomly to rg values
RectChangeBox10_4RG = Poisson(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'rg', 'PoissonRedGreenBox')


# In[146]:


# Adding poisson noise randomly to rb values
RectChangeBox10_4RB = Poisson(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'rb', 'PoissonRedBlueBox')


# In[147]:


# Adding poisson noise randomly to gb values
RectChangeBox10_4BG = Poisson(XTestBox, YTestBox, TriggerImagesTestBox10, model4, 'bg', 'PoissonBlueGreenBox')


# # Poisson Noise on Triangle Trigger

# In[148]:


# Adding poisson noise randomly to rgb values
RectChangeTriangle10_4 = Poisson(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'rgb', 'PoissonRGBTriangle')


# In[149]:


# Adding poisson noise randomly to rED values
RectChangeTriangle10_4R = Poisson(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'red', 'PoissonRedTriangle')


# In[150]:


# Adding poisson noise randomly to GREEN values
RectChangeTriangle10_4G = Poisson(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'green', 'PoissonGreenTriangle')


# In[151]:


# Adding poisson noise randomly to blue values
RectChangeTriangle10_4B = Poisson(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'blue', 'PoissonBlueTriangle')


# In[152]:


# Adding poisson noise randomly to gb values
RectChangeTriangle10_4BG = Poisson(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'bg', 'PoissonBlueGreenTriangle')


# In[153]:


# Adding poisson noise randomly to rb values
RectChangeTriangle10_4RB = Poisson(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'rb', 'PoissonRedBlueTriangle')


# In[154]:


# Adding poisson noise randomly to rg values
RectChangeTriangle10_4RG = Poisson(XTestTriangle, YTestTriangle, TriggerImagesTestTriangle10, model5, 'rg', 'PoissonRedGreenTriangle')


# # Final Classifier

# In[155]:


ss,XDiamond, YDiamond = transformCIFAR('diamond',10,test_im,test_cl)
sss,XCross, YCross = transformCIFAR('cross',10,test_im,test_cl)
st,XBox, YBox = transformCIFAR('box',10,test_im,test_cl)
sss,XTriangle, YTriangle = transformCIFAR('triangle',10,test_im,test_cl)


# In[156]:


NoiseRange = np.linspace(-1,1,1000)
NoiseRangePoisson = np.linspace(0,1,1000)
NoiseRangeRand = np.linspace(-1,1,100)


# In[157]:


import copy

def UniformFuzz(x, noise):
    img = copy.copy(x)
    fuzz = np.ones(3072)
    Fuzz = (fuzz*noise).reshape(1,img.shape[1],img.shape[2],img.shape[3])
    fuzzed = np.clip(img+Fuzz,0,1)
    return fuzzed

def GaussianFuzz(x, noise):
    length = x.size
    img = copy.copy(x)
    Fuzz = noise.reshape((1,img.shape[1],img.shape[2],img.shape[3]))
    fuzzed = np.clip(img+Fuzz,0, 1)
    return fuzzed

def RandomGaussianFuzz(x, noise):
    length = x.size
    img = copy.copy(x)
    Fuzz = noise.reshape((1,img.shape[1],img.shape[2],img.shape[3]))
    fuzzed = np.clip(img+Fuzz,0, 1)
    return fuzzed

def PoissonFuzz(x, noise):
    img = copy.copy(x)
    Fuzz = np.random.poisson(noise,3072).reshape((1,3072))  
    Fuzz = Fuzz.reshape((1,img.shape[1],img.shape[2],img.shape[3]))        
    fuzzed = np.clip(img+Fuzz,0,1)
    return fuzzed


# In[158]:


from collections import Counter
def Classifier(x,ModelName, maximumNoise, fuzzType):
    x1 = copy.copy(x)
    
    if fuzzType == "all":
        for i in range(len(maximumNoise)):
            for j in range(len(maximumNoise[i])):
                if i == 0:
                    x_temp = UniformFuzz(x1, maximumNoise[i][j])
                    
                elif i == 1:
                    x_temp = GaussianFuzz(x1, maximumNoise[i][j])
                    
                
                elif i == 2:
                    x_temp = RandomGaussianFuzz(x1, maximumNoise[i][j])   
                    
                elif i == 3:
                    x_temp = PoissonFuzz(x1, maximumNoise[i][j])
                    
                p_temp = (np.argmax(ModelName.predict(x_temp.reshape((x.shape[0],32,32,3))),1)).reshape((1,x1.shape[0]))
   
                if i == 0 and j == 0:
                    p = p_temp
                else:
                    p = np.concatenate((p,p_temp))  
        
        
    else:
        tp = maximumNoise.shape[0]
        for i in range(tp):
            x_temp = 0
            if fuzzType == "uniform":
                x_temp = UniformFuzz(x1, maximumNoise[i])

            elif fuzzType == "gaussian":
                x_temp = GaussianFuzz(x1, maximumNoise[i])

            elif fuzzType == "rand":
                x_temp = RandomGaussianFuzz(x1, maximumNoise[i])

            elif fuzzType == "poisson":
                x_temp = PoissonFuzz(x1, maximumNoise[i])

            p_temp = (np.argmax(ModelName.predict(x_temp.reshape((x.shape[0],32,32,3))),1)).reshape((1,x1.shape[0]))
    #         print('p', p_temp)
            if i==0:
                p=p_temp
            else:
                p = np.concatenate((p,p_temp))   
    p = np.concatenate((p,(np.argmax(ModelName.predict(x1),1)).reshape((1,x.shape[0]))))
    
    p = np.transpose(p)
    
    
    Predictions = []
    #ConfidenceLevels = []
    for i in range(x.shape[0]):
        temp = Counter(p[i])
#         print(temp)
        Predictions.append(temp.most_common(1)[0][0])
    return(Predictions)


# In[159]:


def maxNoise(Noise, Accuracy, index):
    noise = np.array(Noise[(np.array(Accuracy)).argsort()[-index:]])
    return noise

def calculateAccuracy(prediction, x, y):
    Accu = (np.count_nonzero(prediction == np.argmax(y,1))/x.shape[0])*100
    return Accu


# # Classifier on Uniform Noise

# In[160]:


TP = np.arange(1,21)
correctedUniformAccuracyDiamond = []
correctedUniformAccuracyCross = []
correctedUniformAccuracyBox = []
correctedUniformAccuracyTriangle = []

for i in TP:
    print ("Testing point: ",i)
    diamondMax = maxNoise(NoiseRange, RectChangeDiamond10, i)
    crossMax = maxNoise(NoiseRange, RectChangeCross10, i)
    boxMax = maxNoise(NoiseRange, RectChangeBox10, i)
    triangleMax = maxNoise(NoiseRange, RectChangeTriangle10, i)

    Umeans = np.concatenate((diamondMax, crossMax, boxMax, triangleMax))
    
    predict = Classifier(XDiamond, model2, Umeans, 'uniform')
    diamondAccuracy = calculateAccuracy(predict, XDiamond, YDiamond)
    correctedUniformAccuracyDiamond.append(diamondAccuracy)
    
    predict = Classifier(XCross, model3, Umeans, 'uniform')
    crossAccuracy = calculateAccuracy(predict, XCross, YCross)
    correctedUniformAccuracyCross.append(crossAccuracy)
    
    predict = Classifier(XBox, model4, Umeans, 'uniform')
    boxAccuracy = calculateAccuracy(predict, XBox, YBox)
    correctedUniformAccuracyBox.append(boxAccuracy)
    
    predict = Classifier(XTriangle, model5, Umeans, 'uniform')
    triangleAccuracy = calculateAccuracy(predict, XTriangle, YTriangle)
    correctedUniformAccuracyTriangle.append(triangleAccuracy)
    


# In[161]:


plt.plot(TP,correctedUniformAccuracyDiamond,'b-o')
plt.plot(TP,correctedUniformAccuracyCross,'g-*')
plt.plot(TP,correctedUniformAccuracyBox,'r-v')
plt.plot(TP,correctedUniformAccuracyTriangle,'k->')
plt.legend(['Diamond-poisoned','Cross-poisoned', 'Box-poisoned', 'Triangle-poisoned'],bbox_to_anchor=(1,0.3))
plt.grid(True)
plt.show()


# # Classifier on Guassian Noise

# In[162]:


TP = np.arange(1,21)
correctedGaussianAccuracyDiamond = []
correctedGaussianAccuracyCross = []
correctedGaussianAccuracyBox = []
correctedGaussianAccuracyTriangle = []

for i in TP:
    print ("Testing point: ",i)
    diamondMax = maxNoise(fuzzDiamond, RectChangeDiamond10_2, i)
    crossMax = maxNoise(fuzzCross, RectChangeCross10_2, i)
    boxMax = maxNoise(fuzzBox, RectChangeBox10_2, i)
    triangleMax = maxNoise(fuzzTriangle, RectChangeTriangle10_2, i)

    Umeans = np.concatenate((diamondMax, crossMax, boxMax, triangleMax))
    
    predict = Classifier(XDiamond, model2, Umeans, 'gaussian')
    diamondAccuracy = calculateAccuracy(predict, XDiamond, YDiamond)
    correctedGaussianAccuracyDiamond.append(diamondAccuracy)
    
    predict = Classifier(XCross, model3, Umeans, 'gaussian')
    crossAccuracy = calculateAccuracy(predict, XCross, YCross)
    correctedGaussianAccuracyCross.append(crossAccuracy)
    
    predict = Classifier(XBox, model4, Umeans, 'gaussian')
    boxAccuracy = calculateAccuracy(predict, XBox, YBox)
    correctedGaussianAccuracyBox.append(boxAccuracy)
    
    predict = Classifier(XTriangle, model5, Umeans, 'gaussian')
    triangleAccuracy = calculateAccuracy(predict, XTriangle, YTriangle)
    correctedGaussianAccuracyTriangle.append(triangleAccuracy)
    


# In[163]:


diamondMax = maxNoise(FuzzesDiamond, RectChangeDiamond10, 1)
print(diamondMax)


# In[164]:


plt.plot(TP,correctedGaussianAccuracyDiamond,'b-o')
plt.plot(TP,correctedGaussianAccuracyCross,'g-*')
plt.plot(TP,correctedGaussianAccuracyBox,'r-v')
plt.plot(TP,correctedGaussianAccuracyTriangle,'k->')
plt.legend(['Diamond-poisoned','Cross-poisoned', 'Box-poisoned', 'Triangle-poisoned'],bbox_to_anchor=(1,0.3))
plt.grid(True)
plt.show()


# # Classifier on Random Gaussian Noise

# In[165]:


print(FuzzesDiamond)


# In[166]:


TP = np.arange(1,21)
correctedAccuracyDiamond = []
correctedAccuracyCross = []
correctedAccuracyBox = []
correctedAccuracyTriangle = []

for i in TP:
    print ("Testing point: ",i)
    diamondMax = maxNoise(FuzzesDiamond, RectChangeDiamond10_3, i)
    crossMax = maxNoise(FuzzesCross, RectChangeCross10_3, i)
    boxMax = maxNoise(FuzzesBox, RectChangeBox10_3, i)
    triangleMax = maxNoise(FuzzesTriangle, RectChangeTriangle10_3, i)

    Umeans = np.concatenate((diamondMax, crossMax, boxMax, triangleMax))
    
    predict = Classifier(XDiamond, model2, Umeans, 'rand')
    diamondAccuracy = calculateAccuracy(predict, XDiamond, YDiamond)
    correctedAccuracyDiamond.append(diamondAccuracy)
    
    predict = Classifier(XCross, model3, Umeans, 'rand')
    crossAccuracy = calculateAccuracy(predict, XCross, YCross)
    correctedAccuracyCross.append(crossAccuracy)
    
    predict = Classifier(XBox, model4, Umeans, 'rand')
    boxAccuracy = calculateAccuracy(predict, XBox, YBox)
    correctedAccuracyBox.append(boxAccuracy)
    
    predict = Classifier(XTriangle, model5, Umeans, 'rand')
    triangleAccuracy = calculateAccuracy(predict, XTriangle, YTriangle)
    correctedAccuracyTriangle.append(triangleAccuracy)
    


# In[167]:


plt.plot(TP,correctedAccuracyDiamond,'b-o')
plt.plot(TP,correctedAccuracyCross,'g-*')
plt.plot(TP,correctedAccuracyBox,'r-v')
plt.plot(TP,correctedAccuracyTriangle,'k->')
plt.legend(['Diamond-poisoned','Cross-poisoned', 'Box-poisoned', 'Triangle-poisoned'],bbox_to_anchor=(1,0.3))
plt.grid(True)
plt.show()


# # Classifier on Poisson Noise

# In[168]:


TP = np.arange(1,21)
correctedAccuracyDiamond = []
correctedAccuracyCross = []
correctedAccuracyBox = []
correctedAccuracyTriangle = []

for i in TP:
    print ("Testing point: ",i)
    diamondMax = maxNoise(NoiseRangePoisson, RectChangeDiamond10_4, i)
    crossMax = maxNoise(NoiseRangePoisson, RectChangeCross10_4, i)
    boxMax = maxNoise(NoiseRangePoisson, RectChangeBox10_4, i)
    triangleMax = maxNoise(NoiseRangePoisson, RectChangeTriangle10_4, i)

    Umeans = np.concatenate((diamondMax, crossMax, boxMax, triangleMax))
    
    predict = Classifier(XDiamond, model2, Umeans, 'poisson')
    diamondAccuracy = calculateAccuracy(predict, XDiamond, YDiamond)
    correctedAccuracyDiamond.append(diamondAccuracy)
    
    predict = Classifier(XCross, model3, Umeans, 'poisson')
    crossAccuracy = calculateAccuracy(predict, XCross, YCross)
    correctedAccuracyCross.append(crossAccuracy)
    
    predict = Classifier(XBox, model4, Umeans, 'poisson')
    boxAccuracy = calculateAccuracy(predict, XBox, YBox)
    correctedAccuracyBox.append(boxAccuracy)
    
    predict = Classifier(XTriangle, model5, Umeans, 'poisson')
    triangleAccuracy = calculateAccuracy(predict, XTriangle, YTriangle)
    correctedAccuracyTriangle.append(triangleAccuracy)
    


# In[169]:


plt.plot(TP,correctedAccuracyDiamond,'b-o')
plt.plot(TP,correctedAccuracyCross,'g-*')
plt.plot(TP,correctedAccuracyBox,'r-v')
plt.plot(TP,correctedAccuracyTriangle,'k->')
plt.legend(['Diamond-poisoned','Cross-poisoned', 'Box-poisoned', 'Triangle-poisoned'],bbox_to_anchor=(1,0.3))
plt.grid(True)
plt.show()


# # Classifier on All Noise Distributions

# In[170]:


TP = np.arange(1,21)
correctedAccuracyDiamond = []
correctedAccuracyCross = []
correctedAccuracyBox = []
correctedAccuracyTriangle = []
Umeans = []

for i in TP:
    print ("Testing point: ",i)
    diamondMax = maxNoise(NoiseRange, RectChangeDiamond10, i)
    crossMax = maxNoise(NoiseRange, RectChangeCross10, i)
    boxMax = maxNoise(NoiseRange, RectChangeBox10, i)
    triangleMax = maxNoise(NoiseRange, RectChangeTriangle10, i)

    Unimeans = np.concatenate((diamondMax, crossMax, boxMax, triangleMax))
    
    diamondMax = maxNoise(fuzzDiamond, RectChangeDiamond10_2, i)
    crossMax = maxNoise(fuzzCross, RectChangeCross10_2, i)
    boxMax = maxNoise(fuzzBox, RectChangeBox10_2, i)
    triangleMax = maxNoise(fuzzTriangle, RectChangeTriangle10_2, i)

    Gaussimeans = np.concatenate((diamondMax, crossMax, boxMax, triangleMax))
    
    
    diamondMax = maxNoise(FuzzesDiamond, RectChangeDiamond10_3, i)
    crossMax = maxNoise(FuzzesCross, RectChangeCross10_3, i)
    boxMax = maxNoise(FuzzesBox, RectChangeBox10_3, i)
    triangleMax = maxNoise(FuzzesTriangle, RectChangeTriangle10_3, i)

    Randmeans = np.concatenate((diamondMax, crossMax, boxMax, triangleMax))
    
    diamondMax = maxNoise(NoiseRangePoisson, RectChangeDiamond10_4, i)
    crossMax = maxNoise(NoiseRangePoisson, RectChangeCross10_4, i)
    boxMax = maxNoise(NoiseRangePoisson, RectChangeBox10_4, i)
    triangleMax = maxNoise(NoiseRangePoisson, RectChangeTriangle10_4, i)

    Poissonmeans = np.concatenate((diamondMax, crossMax, boxMax, triangleMax))
    
    Umeans.append(Unimeans)
    Umeans.append(Gaussimeans)
    Umeans.append(Randmeans)
    Umeans.append(Poissonmeans)
    
    
    predict = Classifier(XDiamond, model2, Umeans, 'all')
    diamondAccuracy = calculateAccuracy(predict, XDiamond, YDiamond)
    correctedAccuracyDiamond.append(diamondAccuracy)
    
    predict = Classifier(XCross, model3, Umeans, 'all')
    crossAccuracy = calculateAccuracy(predict, XCross, YCross)
    correctedAccuracyCross.append(crossAccuracy)
    
    predict = Classifier(XBox, model4, Umeans, 'all')
    boxAccuracy = calculateAccuracy(predict, XBox, YBox)
    correctedAccuracyBox.append(boxAccuracy)
    
    predict = Classifier(XTriangle, model5, Umeans, 'all')
    triangleAccuracy = calculateAccuracy(predict, XTriangle, YTriangle)
    correctedAccuracyTriangle.append(triangleAccuracy)
    


# In[171]:


plt.plot(TP,correctedAccuracyDiamond,'b-o')
plt.plot(TP,correctedAccuracyCross,'g-*')
plt.plot(TP,correctedAccuracyBox,'r-v')
plt.plot(TP,correctedAccuracyTriangle,'k->')
plt.legend(['Diamond-poisoned','Cross-poisoned', 'Box-poisoned', 'Triangle-poisoned'],bbox_to_anchor=(1,0.3))
plt.grid(True)
plt.show()


# In[ ]:




