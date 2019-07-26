import pandas as pd
import matplotlib.pyplot as plt

trainCSV = pd.read_csv("train.csv", encoding="ISO-8859-1")
df = pd.DataFrame(trainCSV)
print(pd.isnull(df).any())
plt.matshow(df.corr())
plt.show()
