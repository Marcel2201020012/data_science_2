import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/processed/fill_empty.csv')

train, temp = train_test_split(data, test_size=1/3, shuffle=False)
val, test = train_test_split(temp, test_size=1/3, shuffle=False)

train.to_csv('data/train.csv', index=False)
val.to_csv('data/val.csv', index=False)
test.to_csv('data/test.csv', index=False)