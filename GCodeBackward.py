# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:24:52 2020

@author: wonwoo
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Conv2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import mean_squared_error

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

def dnn_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(64, input_dim = input_dim, activation='relu', name='FC-1'))
    model.add(Dense(32, activation='relu', name='FC-2'))
    model.add(Dense(output_dim, activation='linear', name='Output'))
    model.compile(loss='mse', optimizer=Adam(lr=0.0005), metrics=['mse'])    
    return model

def cnn_model(input_dim, output_dim):
    model = Sequential()
    model.add(Conv2D(64, input_shape=input_dim, kernel_size=(2,2),
                     strides=(1,1), activation='relu', name='Conv2D-1'))
    model.add(Flatten(name='Flatten'))
    model.add(Dense(32, activation='relu', name='FC-1'))
    model.add(Dense(output_dim, activation='linear', name='Output'))
    model.compile(loss='mse', optimizer=Adam(lr=0.0005), metrics=['mse'])
    return model

def data(gcode_path, printed_path):
    gcode_list = []
    printed_list = []
    for files in os.listdir(gcode_path):
        df = pd.read_csv(gcode_path+files)
        df = df.drop(['G'], axis=1)
        gcode = [df['X'].tolist(), df['Y'].tolist()]
        gcode_list.append(gcode)
    
    gcode = np.array(gcode_list)
    gcode_flatten = gcode.reshape(gcode.shape[0], -1)
    print(gcode_flatten.shape)
    
    for files in os.listdir(printed_path):
        df = pd.read_csv(printed_path+files)
        df = df.drop(['G'], axis=1)
        printed = [df['X'].tolist(), df['Y'].tolist()]
        printed_list.append(printed)
    
    printed = np.array(printed_list)
    printed_flatten = printed.reshape(printed.shape[0], -1)
    print(printed_flatten.shape)
    
    return gcode, gcode_flatten, printed, printed_flatten

def train(gcode, printed, test_gcode, test_printed, model_cate="DNN", training=True):        
    trainX, trainX_flatten, trainY, trainY_flatten = data(gcode, printed)
    testX, testX_flatten, testY, testY_flatten = data(test_gcode, test_printed)
            
    test_idx = 15

    if model_cate == "DNN":
        if training==True:    
            epochs = 1000    
            early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                           patience=20)
            model_checkpoint = ModelCheckpoint('best_backward_dnn.h5', monitor='val_loss',
                                               mode='min', save_best_only=True)
            
            
            model = dnn_model(trainY_flatten.shape[1], trainX_flatten.shape[1])
            model.summary()
            history = model.fit(trainY_flatten, trainX_flatten,
                                validation_split=0.1,
                                epochs=epochs, batch_size=64, shuffle=True,
                                callbacks=[early_stopping, model_checkpoint])
            
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            
            x_epochs = range(1, len(loss) + 1)   
            
            x_epochs_5 = [0]
            val_loss_5 = [0]
            loss_5 = [0]
                
            for idx in range(4, len(loss), 5):
                x_epochs_5.append(idx+1)
                val_loss_5.append(val_loss[idx])
                loss_5.append(loss[idx])
                
            fig, ax1 = plt.subplots()
            
            ax1.grid()
            ax1.set_xlabel('epochs')
            ax1.set_ylabel('Loss')
            ax1.plot(x_epochs_5, val_loss_5, 'r', label='Test Loss')
            ax1.plot(x_epochs_5, loss_5, 'b', label='Training Loss')
            fig.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.1))
            fig.tight_layout()
            plt.show() 
            
        else:
            model = load_model('best_backward_dnn.h5')
            model.summary()           
        
        mse_list = []
        max_list = []
        for test_idx in range(0, 61, 1):            
            tests = testX_flatten[test_idx].reshape((1,) + testX_flatten[test_idx].shape)
            pred = model.predict(tests)
            
            mse_list.append(mean_squared_error(testY_flatten[test_idx], pred[0]))
            max_list.append(abs(max(testY_flatten[test_idx][101:])-min(testY_flatten[test_idx][101:])))
            
        plt.plot(mse_list)
        plt.legend(['MSE'])
        plt.show()
        
        plt.plot(max_list)
        plt.legend(['WaveLength'])
        plt.show()
            
        mse_list = []
        max_list = []
        for test_idx in range(61, 122, 1):            
            tests = testX_flatten[test_idx].reshape((1,) + testX_flatten[test_idx].shape)
            pred = model.predict(tests)
            
            mse_list.append(mean_squared_error(testY_flatten[test_idx], pred[0]))
            max_list.append(abs(max(testY_flatten[test_idx][101:])-min(testY_flatten[test_idx][101:])))
            
        plt.plot(mse_list)
        plt.legend(['MSE'])
        plt.show()
        
        plt.plot(max_list)
        plt.legend(['WaveLength'])
        plt.show()
        
        test_idx = 15       
        tests = testX_flatten[test_idx].reshape((1,) + testX_flatten[test_idx].shape)
        print(tests)
        pred = np.array(model.predict(tests))
        pred = pred.reshape((2,-1))
        plt.plot(testX[test_idx][0], testX[test_idx][1])
        plt.plot(testX[test_idx][0], testY[test_idx][1])
        plt.plot(pred[0], pred[1])
        plt.legend(['GT_Gcode','GT_Printed', 'Predict_GCode'])
        plt.title("Backward")
        plt.show()
        
        forward_model = load_model('best_forward_cnn.h5')
        forward_model.summary()        
        input_pred = pred.reshape((2,-1))
        input_pred = input_pred.reshape((1,)+input_pred.shape+(1,))
        print(input_pred.shape)
        pred_s = forward_model.predict(input_pred)
        pred_s = pred_s.reshape((2,-1))
        plt.plot(testX[test_idx][0], pred[1])
        plt.plot(testX[test_idx][0], testX[test_idx][1])
        plt.plot(pred_s[0], pred_s[1])
        plt.legend(['GCode','GT_Printed', 'Predict_Printed'])
        plt.title("Forward")
        plt.show()
        
    elif model_cate=="CNN":              
        trainY = np.expand_dims(trainY, -1)
        testX = np.expand_dims(testX, -1) 
        if training==True:    
            epochs = 1000    
            early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                           patience=20)
            model_checkpoint = ModelCheckpoint('best_backward_cnn.h5', monitor='val_loss',
                                               mode='min', save_best_only=True)
    
            model = cnn_model((trainY.shape[1], trainY.shape[2], 1), trainX_flatten.shape[1])
            model.summary()
            history = model.fit(trainY, trainX_flatten,
                                validation_split = 0.1,
                                epochs=epochs, batch_size=64, shuffle=True,
                                callbacks=[early_stopping, model_checkpoint])    
            loss = history.history['loss']
            val_loss = history.history['val_loss']
        
            x_epochs = range(1, len(loss) + 1)   
            
            x_epochs_5 = [0]
            val_loss_5 = [0]
            loss_5 = [0]
                
            for idx in range(4, len(loss), 5):
                x_epochs_5.append(idx+1)
                val_loss_5.append(val_loss[idx])
                loss_5.append(loss[idx])
                
            fig, ax1 = plt.subplots()
            
            ax1.grid()
            ax1.set_xlabel('epochs')
            ax1.set_ylabel('Loss')
            ax1.plot(x_epochs_5, val_loss_5, 'r', label='Test Loss')
            ax1.plot(x_epochs_5, loss_5, 'b', label='Training Loss')
            fig.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.1))
            fig.tight_layout()
            plt.show()        
        else:
            model = load_model('best_backward_cnn.h5')
            model.summary()            
            
        mse_list = []
        max_list = []
        for test_idx in range(0, 61, 1):            
            tests = testX[test_idx].reshape((1,) + testX[test_idx].shape)
            pred = model.predict(tests)
            
            mse_list.append(mean_squared_error(testY_flatten[test_idx], pred[0]))
            max_list.append(abs(max(testY_flatten[test_idx][101:])-min(testY_flatten[test_idx][101:])))
            
        plt.plot(mse_list)
        plt.legend(['MSE'])
        plt.title('Pure Result')
        plt.show()
        
        plt.plot(max_list)
        plt.legend(['WaveLength'])
        plt.title('Pure Result')
        plt.show()
        
        mse_list = []
        max_list = []
        for test_idx in range(61, 122, 1):            
            tests = testX[test_idx].reshape((1,) + testX[test_idx].shape)
            pred = model.predict(tests)
            
            mse_list.append(mean_squared_error(testY_flatten[test_idx], pred[0]))
            max_list.append(abs(max(testY_flatten[test_idx][101:])-min(testY_flatten[test_idx][101:])))
            
        plt.plot(mse_list)
        plt.legend(['MSE'])
        plt.title('Noise Result')
        plt.show()
        
        plt.plot(max_list)
        plt.legend(['WaveLength'])
        plt.title('Noise Result')
        plt.show()
        
        test_idx = 15       
        tests = testX[test_idx].reshape((1,) + testX[test_idx].shape)
        print(tests)
        pred = model.predict(tests)
        pred = pred.reshape((2,-1))
        plt.plot(testX[test_idx][0], testX[test_idx][1])
        plt.plot(testX[test_idx][0], testY[test_idx][1])
        plt.plot(pred[0], pred[1])
        plt.legend(['GT_Gcode','GT_Printed', 'Predict_GCode'])
        plt.title("Backward")
        plt.show()        
        
        forward_model = load_model('best_forward_cnn.h5')
        forward_model.summary()        
        input_pred = pred.reshape(2,-1)
        input_pred = input_pred.reshape((1,)+input_pred.shape+(1,))
        print(input_pred.shape)
        pred_s = forward_model.predict(input_pred)
        pred_s = pred_s.reshape((2,-1))
        plt.plot(testX[test_idx][0], pred[1])
        plt.plot(testX[test_idx][0], testX[test_idx][1])
        plt.plot(pred_s[0], pred_s[1])
        plt.legend(['GCode','GT_Printed', 'Predict_Printed'])
        plt.title("Forward")
        plt.show()
        

if __name__=="__main__":    
    gcode_path = 'dataset/GT/'
    printed_path = 'dataset/Real/'
    test_gcode_path = 'dataset/test/GT/'
    test_printed_path = 'dataset/test/Real/'
    
    train(gcode_path, printed_path, test_gcode_path, test_printed_path, "DNN", False)