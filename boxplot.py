import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt      


tips = pd.read_csv('./csv/f2f.csv')

sns.boxplot(y=tips["err_rad"],data=tips)
plt.show()
