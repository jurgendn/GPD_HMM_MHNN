import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.tsa.stattools as ts

df = pd.read_csv("traffic count.csv")
df = pd.DataFrame(df)
data = df.values.tolist()

data_all = []
for dt in data:
    for ite in range(1, len(data)):
        data_all.append(dt[ite])

t = [i for i in range(len(data_all))]
plt.plot(t, data_all)

print(ts.adfuller(data_all, regression="c"))
