'''
Created on Mar 20, 2018

@author: abhishek
'''
from matplotlib import pyplot as plt
import pandas as pd
import os


def PlotHeatMap(path,ExFile):
    i = 1
    read = pd.ExcelFile(path+"/"+ExFile,header = None)
    if not os.path.exists(os.getcwd()+"/../Images"):
        os.mkdir(os.getcwd()+"/../Images")
    (name,dummy) = ExFile.split(".")
    if not os.path.exists(os.getcwd()+"/../Images/"+name):
        os.mkdir(os.getcwd()+"/../Images/"+name)
        
    for sheet in read.sheet_names:

        matrix = read.parse(sheet,header=None)
        plt.figure(i)
        plt.imshow(matrix.as_matrix(),cmap='hot',interpolation='nearest') #adjust the size to your needs
        #plt.show()
        plt.savefig(os.getcwd()+"/../Images/"+name+"/"+name+"_Kernel_"+str(i)+".jpg")
        #exit()    
        #plt.close()
        i=i+1
    
