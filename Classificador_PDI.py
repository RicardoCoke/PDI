# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 13:58:38 2021

@author: itsri
"""

import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns


import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, GlobalMaxPooling2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D, Dense, Activation
from tensorflow.keras.models import Sequential


DIR = r"C:\Users\itsri\PDI\Classificacao_displasia"
train = DIR + "\train"
test = DIR+ "\test"


# criar tabela com path e classificacao---------------------------------

def create_df(df, directory):
    
    directory = r"C:\Users\itsri\PDI\Classificacao_displasia"
    label = ["sem_displasia", "com_displasia"]
    for i in range(len(label)):
        row = {}
        for file in glob.glob(directory + "/" + label[i] + "/*"):
            row['path'] = file
            row['class'] = label[i]
            df = df.append(row, ignore_index = True)
            df = df.sample(frac = 1).reset_index(drop = True)
            
    return df

columns_df = ['path', 'class']
        
df_train = pd.DataFrame(columns = columns_df)
df_test = pd.DataFrame(columns = columns_df)

df_train = create_df(df_train, train)
df_test = create_df(df_test, test)

df_train.head()

#FIM DE CRIAR TABELA COM PATH E CLASSIFICACAO----------------------------

l = []

for i in df_train['class']:
    if(i == "sem_displasia"):
        l.append("Sem Displasia")
    else:
        l.append("Com Displasia")
    
    sns.set_style('darkgrid')
    sns.countplot(l)
 
count = 1
f = plt.figure(figsize = (50,13))

for c in range (0,10):
    random_number = np.random.randint(0, len(df_train['path']))
    img_url = cv2.imread(df_train['path'][random_number])
    ax = f.add_subplot(2,5,count)
    ax = plt.imshow(img_url)
    
    ax = plt.title(df_train['class'][random_number], fontsize = 30)
    ax = plt.grid(False)
    count = count + 1
    
plt.suptitle("Displasia na anca do cão", size =32)
plt.show()

#Data Augmentation---------------------------------------

def data_aug(df_train, df_test, batch_size, target_size_dimension):
    
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, zoom_range=0.2, horizontal_flip = True, validation_split=0.45, rotation_range=0.2)
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
    
    train_set = train_datagen.flow_from_dataframe(df_train, x_col = 'path', y_col = 'class', target_size = (target_size_dimension, target_size_dimension), class_mode='categorical',batch_size=batch_size, subset='training')
    
    val_set = train_datagen.flow_from_dataframe(df_train, x_col = 'path', y_col = 'class', target_size = (target_size_dimension, target_size_dimension), class_mode='categorical',batch_size=batch_size, subset='training')
    
    test_set = test_datagen.flow_from_dataframe(df_test, x_col = 'path', y_col = 'class',
                                                                         target_size = (target_size_dimension, target_size_dimension), 
                                                                         class_mode='categorical',
                                                                         batch_size=1, shuffle=False)
    
    return (train_set, val_set, test_set)

train_set, val_set, test_set = data_aug(df_train, df_test, 75, 100)

def plotImages (images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        plt.tight_layout()
        plt.show()
        
        ag_img = [train_set[1][0][0] for i in range(5)]
        plotImages(ag_img)
        
#Rede Neuronal Profunda Xception--------------------------------------
        
def create_model_xception(img_size, channels, metrics):
    input_shape = (img_size, img_size, channels)
    base_model = Xception(weights = 'imagenet', include_top = False, input_shape = input_shape)
    
    modelo = Sequential()
    modelo.add(base_model)
    modelo.add(GlobalAveragePooling2D())
    modelo.add(Dense(256))
    modelo.add(Activation('relu'))
    modelo.add(Dropout(0.7))
    
    modelo.add(Dense(256))
    modelo.add(Activation('relu'))
    modelo.add(Dropout(0.7))
    
    modelo.add(Dense(2, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(lr = 5e-4) #mudar este valor
    
    # novo modelo sumario
    modelo.compile (loss = 'binary_crossentropy', optimizer = optimizer, metrics = metrics)
    return modelo

batch_size = 25
target_size = 150
cv_channels = 3

METRICS = ['accuracy',tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name = 'recall')]

train_set, val_set, test_set = data_aug (df_train, df_test, batch_size, target_size)
modelo = create_model_xception(target_size, cv_channels, METRICS)

history = modelo.fit(train_set,
                     steps_per_epoch=train_set.samples/batch_size,
                     epochs=400, validation_data=val_set,
                     validation_steps=val_set.samples/batch_size,
                     shuffle=True)

model_name = 'pdi_displasia_anca_cao.h5'
modelo.save(model_name)


def plot_grafico_acuracia(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy Model')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.show()
    
def plot_grafico_perda(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss Model')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.show()
    
def plot_grafico_recall(history):
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Recall Model')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.show()
    
    
novo_modelo = create_model_xception(target_size, cv_channels, METRICS)
novo_modelo.load_weights('pdi_displasia_anca_cao.h5')
    
scores = novo_modelo.evaluate(test_set, verbose=0)
print("%s:  %.2f%%" % (modelo.metrics_names[1], scores[1]*100))

from sklearn.metrics import confusion_matrix , classification_report

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize = 14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
        )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot = True, fmt = "d")
    except ValueError:
        heatmap.yaxis.set_ticklabels(heatmap.yaxis_ticklabels(), rotation = 0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis_ticklabels(), rotation = 45, ha='right', fontsize=fontsize)
        plt.ylabel('Truth')
        plt.xlabel('Prediction')
        
pred = novo_modelo.predict_generator(test_set, steps=len(test_set), verbose = 1)

y_pred = np.argmax(pred, axis=1)
cm = confusion_matrix (test_set.classes, y_pred)
label = ["sem displasia", "com_displasia"]
print_confusion_matrix(cm,label)

print(classification_report(test_set.classes, y_pred, target_names = label))

from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.metrics import roc_auc_score

def plot_grafico_roc(test_set, pred):
    colors = ['blue','red']
    
    classes = np.array(test_set.classes)
    fpr = {}
    tpr = {}
    roc_auc = {}    
    
    for i in range(len(label)):
        fpr[i], tpr[i], _ = roc_curve(classes, pred[:,i], pos_label = i)
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fig = plt.figure(figsize=(5,5))
    for i, cor in zip(range(len(label)), range(len(colors))):
        plt.plot(fpr[i], tpr[i], linestyle = '-', color = colors[cor], label = label[i] + '(area = {1:0.2f})' 
                 ''.format(i, roc_auc[i], lw=2))
                 
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.00, 1.05])
    plt.title('Diplasia na anca do cão')
    plt.xlabel('Flase Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
                     
    plot_grafico_roc(test_set, pred)
                     
        

        
        


    