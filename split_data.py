import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/klasifikasi/processed/encoded.csv')

train, temp = train_test_split(data, test_size=1/3, random_state=42, shuffle=True)
val, test = train_test_split(temp, test_size=1/3, random_state=42, shuffle=True)

train.to_csv('data/klasifikasi/split/train.csv', index=False)
val.to_csv('data/klasifikasi/split/val.csv', index=False)
test.to_csv('data/klasifikasi/split/test.csv', index=False)