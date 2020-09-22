# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:00:34 2020

@author: wonwoo
"""
import pandas as pd
import random

def trainM():
    dir_path = "dataset/"
    
    Y_start = 0
    Y_max = 50
    X_start = 0
    X_max = 50
    X_end = 100
    next_step = 0.5
    dx = 0.05
    alpha = 1
    beta = 1.25
    
    # Real Value
    cnt = 0
    for alpha in range(-300, 301, 1):
        X = [X_start]
        Y = [Y_max]        
        
        alpha /= 10
        
        if alpha >= 0:
            beta = 0.75
        
        for x_ in range(1, X_max // 2 + 1):
            X.append(x_)
            Y_max += dx * alpha * beta
            Y.append(Y_max)
            #next_step -= dx * alpha
        
        for x_ in range(X_max // 2 + 1, X_max + 1):
            X.append(x_)
            Y_max -=  dx * alpha * beta
            Y.append(Y_max)
            #next_step += dx * alpha
                    
        for x_ in range(X_max + 1, X_max + X_max // 2 + 1):
            X.append(x_)
            Y_max += dx * alpha * beta
            Y.append(Y_max)
            #next_step -= dx * alpha
        
        for x_ in range(X_max + X_max // 2 + 1, X_end + 1):
            X.append(x_)
            Y_max -=  dx * alpha * beta
            Y.append(Y_max)
            #next_step += dx * alpha
            
        code_list = []
        for x, y in zip(X,Y):
            code_list.append([1, x, y])
        
        tmp = cnt+601
        df = pd.DataFrame(code_list, columns=["G", "X", "Y"])
        df.to_csv(dir_path+'Real/Real'+str(tmp).zfill(4)+'.csv', index=False)
        cnt += 1
        
    Y_start = 0
    Y_max = 50
    X_start = 0
    X_end = 100
    next_step = 0.5
    dx = 0.05
    alpha = 1
    
    # Ground Truth
    cnt = 0    
    for alpha in range(-300, 301, 1):
        X = [X_start]
        Y = [Y_max]
        
        alpha /= 10
        
        for x_ in range(1, X_max // 2 + 1):
            X.append(x_)
            Y_max += dx * alpha
            Y.append(Y_max)
        
        for x_ in range(X_max // 2 + 1, X_max + 1):
            X.append(x_)
            Y_max -= dx * alpha
            Y.append(Y_max)
            
        for x_ in range(X_max + 1, X_max + X_max // 2 + 1):
            X.append(x_)
            Y_max += dx * alpha
            Y.append(Y_max)
        
        for x_ in range(X_max + X_max // 2 + 1, X_end + 1):
            X.append(x_)
            Y_max -= dx * alpha
            Y.append(Y_max)
            
        code_list = []
        for x, y in zip(X,Y):
            code_list.append([1, x, y])
        
        tmp = cnt+601
        df = pd.DataFrame(code_list, columns=["G", "X", "Y"])
        df.to_csv(dir_path+'GT/GT'+str(tmp).zfill(4)+'.csv', index=False)
        cnt += 1

def train():
    dir_path = "dataset/"
    
    Y_start = 0
    Y_max = 50
    X_start = 0
    X_max = 50
    X_end = 100
    next_step = 0.5
    dx = 0.05
    alpha = 1
    beta = 1.25
    
    # Real Value
    cnt = 0
    for alpha in range(-300, 301, 1):
        X = [X_start]
        Y = [Y_max]        
        
        alpha /= 10
        
        if alpha >= 0:
            beta = 0.75
        
        for x_ in range(1, X_end // 2 + 1):
            X.append(x_)
            Y_max += dx * alpha * beta
            Y.append(Y_max)
            #next_step -= dx * alpha
        
        for x_ in range(X_end // 2 + 1, X_end + 1):
            X.append(x_)
            Y_max -=  dx * alpha * beta
            Y.append(Y_max)
            #next_step += dx * alpha
            
        code_list = []
        for x, y in zip(X,Y):
            code_list.append([1, x, y])
        
        df = pd.DataFrame(code_list, columns=["G", "X", "Y"])
        df.to_csv(dir_path+'Real/Real'+str(cnt).zfill(4)+'.csv', index=False)
        cnt += 1
    
    Y_start = 0
    Y_max = 50
    X_start = 0
    X_end = 100
    next_step = 0.5
    dx = 0.05
    alpha = 1
    
    # Ground Truth
    cnt = 0    
    for alpha in range(-300, 301, 1):
        X = [X_start]
        Y = [Y_max]
        
        alpha /= 10
        
        for x_ in range(1, X_end // 2 + 1):
            X.append(x_)
            Y_max += dx * alpha
            Y.append(Y_max)
        
        for x_ in range(X_end // 2 + 1, X_end + 1):
            X.append(x_)
            Y_max -= dx * alpha
            Y.append(Y_max)
            
        code_list = []
        for x, y in zip(X,Y):
            code_list.append([1, x, y])
        
        df = pd.DataFrame(code_list, columns=["G", "X", "Y"])
        df.to_csv(dir_path+'GT/GT'+str(cnt).zfill(4)+'.csv', index=False)
        cnt += 1
    
def test():
    dir_path = "dataset/test/"
    
    Y_start = 0
    Y_max = 50
    X_start = 0
    X_end = 100
    next_step = 0.5
    dx = 0.05
    alpha = 1
    beta = 1.25
    
    alpha_rand = []
    for alpha in range(-30, 31, 1):
        factor = random.randint(1, 9)
        alpha_rand.append(alpha - factor / 100)
        
    # Real Value
    cnt = 0
    for alpha in range(-30, 31, 1):
        X = [X_start]
        Y = [Y_max]
        alpha = alpha_rand[cnt]
        
        if alpha >= 0:
            beta = 0.75
        
        for x_ in range(1, X_end // 2 + 1):
            X.append(x_)
            Y_max += dx * alpha * beta
            Y.append(Y_max)
            #next_step -= dx * alpha
        
        for x_ in range(X_end // 2 + 1, X_end + 1):
            X.append(x_)
            Y_max -=  dx * alpha * beta
            Y.append(Y_max)
            #next_step += dx * alpha
            
        code_list = []
        for x, y in zip(X,Y):
            code_list.append([1, x, y])
        
        df = pd.DataFrame(code_list, columns=["G", "X", "Y"])
        df.to_csv(dir_path+'Real/Real'+str(cnt).zfill(4)+'.csv', index=False)
        cnt += 1
    
    Y_start = 0
    X_start = 0
    X_end = 100
    next_step = 0.5
    dx = 0.05
    alpha = 1
    
    # Ground Truth
    cnt = 0    
    for alpha in range(-30, 31, 1):
        X = [X_start]
        Y = [Y_max]
        alpha = alpha_rand[cnt]
        
        for x_ in range(1, X_end // 2 + 1):
            X.append(x_)
            Y_max += dx * alpha
            Y.append(Y_max)
        
        for x_ in range(X_end // 2 + 1, X_end + 1):
            X.append(x_)
            Y_max -= dx * alpha
            Y.append(Y_max)
            
        code_list = []
        for x, y in zip(X,Y):
            code_list.append([1, x, y])
        
        df = pd.DataFrame(code_list, columns=["G", "X", "Y"])
        df.to_csv(dir_path+'GT/GT'+str(cnt).zfill(4)+'.csv', index=False)
        cnt += 1
        
def testM():
    dir_path = "dataset/test/"
    
    Y_start = 0
    Y_max = 50
    X_start = 0
    X_max = 50
    X_end = 100
    next_step = 0.5
    dx = 0.05
    alpha = 1
    beta = 1.25
    
    noise_rand = []
    alpha_rand = []
    for alpha in range(-30, 31, 1):
        factor = random.randint(1, 9)
        alpha_rand.append(alpha - factor / 100)
        
        noise = random.randint(0,10)
        noise_rand.append([random.uniform(-noise, noise) for _ in range(101)])
        
    # Real Value
    cnt = 0
    for alpha in range(-30, 31, 1):
        X = [X_start]
        Y = [Y_max]
        alpha = alpha_rand[cnt]
        noise = noise_rand[cnt]
        
        if alpha >= 0:
            beta = 0.75
            
        rand_cnt = 0
        for x_ in range(1, X_max // 2 + 1):
            X.append(x_)
            Y_max += dx * alpha * beta
            Y.append(Y_max + noise[rand_cnt])
            rand_cnt += 1
            #next_step -= dx * alpha
        
        for x_ in range(X_max // 2 + 1, X_max + 1):
            X.append(x_)
            Y_max -=  dx * alpha * beta
            Y.append(Y_max + noise[rand_cnt])
            rand_cnt += 1
            #next_step += dx * alpha
            
        for x_ in range(X_max + 1, X_max + X_max // 2 + 1):
            X.append(x_)
            Y_max += dx * alpha * beta
            Y.append(Y_max + noise[rand_cnt])
            rand_cnt += 1
            #next_step -= dx * alpha
        
        for x_ in range(X_max + X_max // 2 + 1, X_end + 1):
            X.append(x_)
            Y_max -=  dx * alpha * beta
            Y.append(Y_max + noise[rand_cnt])
            rand_cnt += 1
            #next_step += dx * alpha
            
        code_list = []
        for x, y in zip(X,Y):
            code_list.append([1, x, y])
        
        df = pd.DataFrame(code_list, columns=["G", "X", "Y"])
        df.to_csv(dir_path+'Real/Real'+str(cnt+61).zfill(4)+'.csv', index=False)
        cnt += 1
    
    Y_start = 0
    X_start = 0
    X_end = 100
    next_step = 0.5
    dx = 0.05
    alpha = 1
    
    # Ground Truth
    cnt = 0    
    for alpha in range(-30, 31, 1):
        X = [X_start]
        Y = [Y_max]
        alpha = alpha_rand[cnt]
        noise = noise_rand[cnt]
        
        rand_cnt = 0
        for x_ in range(1, X_max // 2 + 1):
            X.append(x_)
            Y_max += dx * alpha
            Y.append(Y_max + noise[rand_cnt])
            rand_cnt += 1
        
        for x_ in range(X_max // 2 + 1, X_max + 1):
            X.append(x_)
            Y_max -= dx * alpha
            Y.append(Y_max + noise[rand_cnt])
            rand_cnt += 1
        
        for x_ in range(X_max + 1, X_max + X_max // 2 + 1):
            X.append(x_)
            Y_max += dx * alpha
            Y.append(Y_max + noise[rand_cnt])
            rand_cnt += 1
        
        for x_ in range(X_max + X_max // 2 + 1, X_end + 1):
            X.append(x_)
            Y_max -= dx * alpha
            Y.append(Y_max + noise[rand_cnt])
            rand_cnt += 1
            
        code_list = []
        for x, y in zip(X,Y):
            code_list.append([1, x, y])
        
        df = pd.DataFrame(code_list, columns=["G", "X", "Y"])
        df.to_csv(dir_path+'GT/GT'+str(cnt+61).zfill(4)+'.csv', index=False)
        cnt += 1
        
if __name__ == "__main__":
    #train()
    #trainM()
    test()
    testM()