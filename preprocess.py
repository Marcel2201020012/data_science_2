import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

data = pd.read_csv("data/klasifikasi/raw/raw.csv")
# print(data.columns)

#fill data 0 dengan median
# data["Total"] = data["Total"].replace(0, data["Total"].median())
# data.to_csv("data/processed/fill_empty.csv", index=False)

# data.plot(x="Tanggal", y="Total")
# plt.show()

#one-hot encoding
mapping = {"positive": 1, "negative": 0}
data["Sentiment"] = data["Sentiment"].replace(mapping)
data.to_csv("data/klasifikasi/processed/encoded.csv", index=False)

#normalisasi
# data_train = pd.read_csv("data/split/train.csv")
# data_val = pd.read_csv("data/split/val.csv")
# data_test = pd.read_csv("data/split/test.csv")

# scaler = MinMaxScaler()
# scaler.fit(data_train[["Total"]])

# data_train["Total"] = scaler.transform(data_train[["Total"]])
# data_train.to_csv("data/normalize/train.csv", index=False)

# data_val["Total"] = scaler.transform(data_val[["Total"]])
# data_val.to_csv("data/normalize/val.csv", index=False)

# data_test["Total"] = scaler.transform(data_test[["Total"]])
# data_test.to_csv("data/normalize/test.csv", index=False)

# joblib.dump(scaler, "data/scaler.pkl")