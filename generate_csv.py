import pandas as pd

type = "f2f_outlier_50"
fileName = "./outlier_txt/"+type+".txt"
f = open(fileName, 'r')
    
allData = []
line = f.readline()
dat = line.split(' ')
err_rad = []
err_trans = []
#extract data from text file
for i in range(1200):
    if not line:
        break
    err_rad.append(float(dat[1]))
    err_trans.append(float(dat[3]))
    line = f.readline()
    dat = line.split(' ')

dataframe = pd.DataFrame({'err_rad':err_rad, 'err_trans':err_trans})

dataframe.to_csv("./outlier_csv/"+type+".csv",index=False,sep=',')
