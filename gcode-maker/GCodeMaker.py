# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:00:34 2020

@author: wonwoo
"""
import pandas as pd

Y_start = 0
Y_max = 0
X_start = 0
X_end = 100
next_step = 0.5
dx = 0.005

X = [X_start]
Y = [Y_start]

for x_ in range(1, X_end // 2):
    X.append(x_)
    Y_max += next_step
    Y.append(Y_max)
    next_step -= dx

for x_ in range(X_end // 2 + 1, X_end + 1):
    X.append(x_)
    Y_max -= next_step
    Y.append(Y_max)
    next_step += dx
    
code_list = []
for x, y in zip(X,Y):
    code_list.append("G1 X{} Y{:.3f}".format(x,y))

df = pd.DataFrame(code_list)
df.to_csv('GCode.txt', index=False)