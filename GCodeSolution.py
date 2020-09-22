# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 00:13:15 2020

@author: nww73
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model

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

def writeGCODE(testX, testX_flatten, testY, testY_flatten):
    test_idx = 121
    testX = np.expand_dims(testX, -1) 
    
    model = load_model('best_backward_cnn.h5')
    model.summary()
    tests = testX[test_idx].reshape((1,) + testX[test_idx].shape)
    
    pred = model.predict(tests)
    pred = pred.reshape((2,-1))
    plt.plot(testX[test_idx][0], testX[test_idx][1])
    plt.plot(testX[test_idx][0], testY[test_idx][1])
    plt.plot(pred[0], pred[1])
    plt.legend(['GT_Gcode','GT_Printed', 'Predict_GCode'])
    plt.title("Backward")
    plt.show()
    
    code_list = []
    for x, y in zip(pred[0],pred[1]):
        code_list.append("G1 X{:.3f} Y{:.3f}".format(x,y))
    
    df = pd.DataFrame(code_list)
    df.to_csv('GCodeResult.txt', index=False, header=False)
    
def readGCODE(testX, testX_flatten, testY, testY_flatten):
    test_idx = 91
    
    model = load_model('best_forward_dnn.h5')
    model.summary()
    
    tests = testX_flatten[test_idx].reshape((1,) + testX_flatten[test_idx].shape)
    pred = model.predict(tests)
    pred = pred.reshape(2,-1)
    
    plt.plot(testX[test_idx][0], testX[test_idx][1])
    plt.plot(testX[test_idx][0], testY[test_idx][1])
    plt.plot(pred[0], pred[1])
    plt.legend(['GT_Gcode','GT_Printed', 'Predict_Printed'])
    plt.show()           
    
    code_list = []
    for x, y in zip(pred[0],pred[1]):
        code_list.append("X{:.3f} Y{:.3f}".format(x,y))
    
    df = pd.DataFrame(code_list)
    df.to_csv('PrintedResult.txt', index=False, header=False)
    
if __name__ == "__main__":    
    test_gcode_path = 'dataset/test/GT/'
    test_printed_path = 'dataset/test/Real/'
    
    testX, testX_flatten, testY, testY_flatten = data(test_gcode_path, test_printed_path)
     
    readGCODE(testX, testX_flatten, testY, testY_flatten)
    #writeGCODE(testX, testX_flatten, testY, testY_flatten)