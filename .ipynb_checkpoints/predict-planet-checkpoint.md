```python
import numpy as np
import pandas as pd
import datetime
```

```python
def load_data():
    df = pd.read_csv("data/stars.csv", skiprows = 22)
    return df

df = load_data()
df['st_ppnum'] = df['st_ppnum'].astype('uint8')
df['st_ppnum'] = np.where(df['st_ppnum'] == 0, 0, 1)
df.dropna(how='any', inplace=True)
df.drop(columns = ['star_name', 'st_spttype', 'decstr'], inplace=True)
df.dtypes
df["rastr"] = pd.to_datetime(df["rastr"], format="%Hh%Mm%S.%fs") - np.datetime64('1900-01-01')
df["rastr"] = df["rastr"].astype('int')
df
```

```python
# preparing train and test set
x = df.loc[:, df.columns != 'st_ppnum']
y = (df['st_ppnum'] == 1)
y.value_counts()
```

```python
# vversample positive samples
def random_oversampling(df: pd.DataFrame) -> pd.DataFrame:

    zeroes = df[df['st_ppnum'] == 0]
    ones = df[df['st_ppnum'] == 1]
    
    small = ones
    large = zeroes
    
    if (zeroes.shape[0] < ones.shape[0]):
        small = zeroes
        large = ones
        
    toAdd = small.sample(n = large.shape[0], replace=True)
    return pd.concat([toAdd, large])
        
df = random_oversampling(df)
```

```python
# verify oversampling
x = df.loc[:, df.columns != 'st_ppnum']
y = (df['st_ppnum'] == 1)
y.value_counts()
```

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
```

```python
from sklearn import svm

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
clf.score(x_test, y_test)
```

```python

```
