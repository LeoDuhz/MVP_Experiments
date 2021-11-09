import pandas as pd

tips = pd.read_csv('./p22f.csv')

rad = tips["err_rad"].values.tolist()
trans = tips["err_trans"].values.tolist()

rad.sort()
trans.sort()

length = len(rad)

rate = [0,0.05,0.25,0.5,0.75,0.95,1]

for i in range(len(rate)):
    index = min(max(int(rate[i] * length), 0), length-1)
    print("rate: ", rate[i], 'index', index, "err_rad: ", rad[index])
    print("rate: ", rate[i], 'index', index, "err_trans: ", trans[index])