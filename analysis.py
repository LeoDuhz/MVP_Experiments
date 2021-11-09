import pandas as pd
import csv

type = "p22f"
tips = pd.read_csv('./csv/'+type+'.csv')

rad = tips["err_rad"].values.tolist()
trans = tips["err_trans"].values.tolist()

rad.sort()
trans.sort()

length = len(rad)

rate = [0,0.05,0.25,0.5,0.75,0.95,1]

err_rad = []
err_trans = []

for i in range(len(rate)):
    index = min(max(int(rate[i] * length), 0), length-1)
    print("rate: ", rate[i], 'index', index, "err_rad: ", rad[index])
    print("rate: ", rate[i], 'index', index, "err_trans: ", trans[index])
    err_rad.append(rad[index])
    err_trans.append(trans[index])


frame = pd.DataFrame([err_rad,err_trans], columns=['0','0.05','0.25','0.5','0.75','0.95','1'])
# print(frame)
frame.index = ["Rotation[rad]", "Translation[m]"]
frame.to_csv("./output/"+type+".csv")