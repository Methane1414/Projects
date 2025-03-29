import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datafile = pd.read_csv("C:\\Users\\chaur\\Documents\\Modelling and Simulation Lab\\Dynamic_CSTR.csv") #Importing CSV File
print(datafile.info())
datafile.iloc[:,1:] = datafile.iloc[:,1:].apply(pd.to_numeric, errors="coerce")

def plotting_timeseries(x,y,xlabel,ylabel,title,color): 
    plt.plot(x,y,label=y.name,color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
    
plotting_timeseries(datafile["t"],datafile["Ca"],"Time(hr)","Concentration (lbmol/ft3)","Time Series Analysis: Concentration Over Time","b" )
plotting_timeseries(datafile["t"],datafile["T"],"Time(hr)","Temperature (R)","Time Series Analysis:Temperature Over Time","r")
plotting_timeseries(datafile["t"],datafile["Tj"],"Time(hr)","Temperature of Jacket(R)","Time Series Analysis:Temp. of Jacket Over Time","g")
