```python
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import ValidationCurveDisplay, validation_curve
```

```python
#Goal: Determine if a candidate planet is truly an exoplanet/false positive
def load_data():
    df = pd.read_csv("data/stars.csv", skiprows = 52)
    return df

df = load_data()
df['isExoplanet'] = df['koi_disposition'].apply(lambda x : 1 if x == "CONFIRMED" else 2 if x == "CANDIDATE" else 0)
df = df.drop(df[df['isExoplanet'] == 2].index)
df = df.drop(columns=['kepid', 'koi_disposition', 'koi_pdisposition', 'koi_period_err1', 'koi_period_err2', 
                      'koi_slogg_err1', 'koi_slogg_err2', 'koi_srad_err1', 'koi_srad_err2', 'koi_time0bk_err1', 
                      'koi_time0bk_err2', 'koi_impact_err1', 'koi_impact_err2', 'koi_steff_err1', 'koi_steff_err2',
                      'koi_duration_err1', 'koi_duration_err2', 'koi_insol_err1', 'koi_insol_err2', 'koi_depth_err1',
                      'koi_depth_err2', 'koi_prad_err1', 'koi_prad_err2', 'koi_teq_err1', 'koi_teq_err2',])

df = df.dropna()
#normalise data
df.iloc[:, :-1] = (df.iloc[:, :-1] - df.iloc[:, :-1].mean())/df.iloc[:, :-1].std() 
```

```python
# preparing train and test set
x = df.iloc[:, :-1]
y = (df.iloc[:, -1])
y.value_counts()
```

```python
def random_oversampling(df: pd.DataFrame) -> pd.DataFrame:

    zeroes = df[df['isExoplanet'] == 0]
    ones = df[df['isExoplanet'] == 1]
    
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
x = df.iloc[:, :-1]
y = (df.iloc[:, -1])
y.value_counts()
```

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=35)
```

```python
# We apply a Support Vector Machine in attempt to do so
clf = svm.SVC(kernel='linear', random_state=17)
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

ValidationCurveDisplay.from_estimator(
   svm.SVC(kernel="linear"), x_train, y_train, param_name="C", param_range=np.logspace(-7, 3, 10)
)
```

```python
# Here, we apply a Logistic Regression to do so
def cost_function(X: np.ndarray, y: np.ndarray, weight_vector: np.ndarray):
    EPS = np.finfo(float).eps
    
    wtx = X @ weight_vector
    y_pred = 1 / (1 + np.exp(-wtx))
    
    yOne = -1 * y * np.log(y_pred + EPS)
    yZero = -1 * (1 - y) * np.log(1 - y_pred + EPS)

    return np.mean(yOne + yZero)
```

```python
def weight_update(X: np.ndarray, y: np.ndarray, alpha: np.float64, weight_vector: np.ndarray) -> np.ndarray:
    EPS = np.finfo(float).eps
    m = X.shape[0]
    
    wtx = X @ weight_vector
    y_pred = 1 / (1 + np.exp(-wtx))
    
    weight_vector = weight_vector - alpha / m * (y_pred - y) @ X
    
    return weight_vector
```

```python
def logistic_regression_classification(X: np.ndarray, weight_vector: np.ndarray, prob_threshold: np.float64=0.5):
    EPS = np.finfo(float).eps
    m = X.shape[0]
    
    wtx = X @ weight_vector
    y_pred = 1 / (1 + np.exp(-wtx))
    
    y_pred += prob_threshold
    return np.floor(y_pred)
```

```python
import matplotlib.pyplot as plt

def logistic_regression_batch_gradient_descent_MPL(X_train: np.ndarray, y_train: np.ndarray, max_num_epochs: int = 250, threshold: np.float64 = 0.05, alpha: np.float64 = 1e-5):

    weight = np.zeros(X_train.shape[1])
    error = []
    
    for i in range(max_num_epochs):
        error.append(cost_function(X_train, y_train, weight))
        if (cost_function(X_train, y_train, weight) <= threshold):
            return weight
        weight = weight_update(X_train, y_train, alpha, weight)
    
    return error

plt.plot([i for i in range(20000)],\
         logistic_regression_batch_gradient_descent_MPL(x_train, y_train, 20000, alpha=1.5e-3))

plt.title("Graph of Cost against Batch Gradient Descent")
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()
```
