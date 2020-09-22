import matplotlib.pyplot as plt 
import pandas as pd

dir_path = "dataset/test/"
df_GT = pd.read_csv(dir_path+"GT/GT0060.csv", sep=',')
df_Real = pd.read_csv(dir_path+"Real/Real0060.csv", sep=',')

X_GT = df_GT.X
Y_GT = df_GT.Y
X_Real = df_Real.X
Y_Real = df_Real.Y

plt.plot(X_GT, Y_GT)
plt.plot(X_Real, Y_Real)
plt.legend(['GT', 'Real'])
plt.show()