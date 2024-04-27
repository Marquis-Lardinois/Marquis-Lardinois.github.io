# Predicting Heart Failure
## Machine Learning with Logistic Regression

### Import Modules


```python
import pandas as pd
import numpy as np
import plotly.express as px
%matplotlib widget
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score)
import pyarrow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import joblib
import warnings
import seaborn as sn
plt.style.use('dark_background')
sn.set_style('darkgrid')
#warnings.filterwarnings("ignore")
```

### Load Data and Perform EDA


```python
raw_df = pd.read_csv('data/heart_disease_health_indicators_BRFSS2015.csv')

```


```python
raw_df.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HeartDiseaseorAttack</th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>Diabetes</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>Veggies</th>
      <th>HvyAlcoholConsump</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>40.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>18.0</td>
      <td>15.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>25.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>6.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>25.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>25.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>13.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>34.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>4.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>33.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>30.0</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
raw_df.shape
```




    (253680, 22)




```python
raw_df.isna().sum()
```




    HeartDiseaseorAttack    0
    HighBP                  0
    HighChol                0
    CholCheck               0
    BMI                     0
    Smoker                  0
    Stroke                  0
    Diabetes                0
    PhysActivity            0
    Fruits                  0
    Veggies                 0
    HvyAlcoholConsump       0
    AnyHealthcare           0
    NoDocbcCost             0
    GenHlth                 0
    MentHlth                0
    PhysHlth                0
    DiffWalk                0
    Sex                     0
    Age                     0
    Education               0
    Income                  0
    dtype: int64




```python
heartattack_counts = raw_df['HeartDiseaseorAttack'].value_counts()
rename_labels = {
    0: 'No Heart Disease',
    1: 'Heart Disease'}
fig = px.pie(values=heartattack_counts,
             names=heartattack_counts.index.map(rename_labels),
             title='Distribution of Heart Diagnosis',
             labels=rename_labels,
             hover_name=heartattack_counts.index.map(rename_labels))
fig.update_traces(textinfo='percent+label', text=heartattack_counts.index.map(rename_labels))
fig.show()
```



```python
categories = ['HighBP', 'Smoker', 'Stroke', 'Diabetes', 'HvyAlcoholConsump']

rename_labels = {
    0: 'No',
    1: 'Yes'
}

for category in categories:
    category_counts = raw_df[category].value_counts()

    fig = px.pie(values=category_counts,
                 names=category_counts.index.map(rename_labels),
                 title=f'Distribution of {category}',
                 labels=rename_labels,
                 hover_name=category_counts.index.map(rename_labels))

    fig.update_traces(textinfo='percent+label', text=category_counts.index.map(rename_labels))
    fig.show()
```



var gd = document.getElementById('acfaea62-73a3-47a6-80af-deeeaca6a8dd');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>







```python
raw_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HeartDiseaseorAttack</th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>Diabetes</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>Veggies</th>
      <th>HvyAlcoholConsump</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.094186</td>
      <td>0.429001</td>
      <td>0.424121</td>
      <td>0.962670</td>
      <td>28.382364</td>
      <td>0.443169</td>
      <td>0.040571</td>
      <td>0.296921</td>
      <td>0.756544</td>
      <td>0.634256</td>
      <td>0.811420</td>
      <td>0.056197</td>
      <td>0.951053</td>
      <td>0.084177</td>
      <td>2.511392</td>
      <td>3.184772</td>
      <td>4.242081</td>
      <td>0.168224</td>
      <td>0.440342</td>
      <td>8.032119</td>
      <td>5.050434</td>
      <td>6.053875</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.292087</td>
      <td>0.494934</td>
      <td>0.494210</td>
      <td>0.189571</td>
      <td>6.608694</td>
      <td>0.496761</td>
      <td>0.197294</td>
      <td>0.698160</td>
      <td>0.429169</td>
      <td>0.481639</td>
      <td>0.391175</td>
      <td>0.230302</td>
      <td>0.215759</td>
      <td>0.277654</td>
      <td>1.068477</td>
      <td>7.412847</td>
      <td>8.717951</td>
      <td>0.374066</td>
      <td>0.496429</td>
      <td>3.054220</td>
      <td>0.985774</td>
      <td>2.071148</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>12.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>24.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>31.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>6.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>98.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>30.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>13.000000</td>
      <td>6.000000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Creating Categorical Columns so encoding and decoding is easier later


```python
binary_map1 = {0: 'No', 1: 'Yes'}
cols_to_map = ['HeartDiseaseorAttack', 'HighBP', 'HighChol','CholCheck','Smoker','Stroke',
               'PhysActivity', 'Fruits', 'Veggies','HvyAlcoholConsump', 'AnyHealthcare',
               'NoDocbcCost','DiffWalk']
raw_df[cols_to_map] = raw_df[cols_to_map].replace(binary_map1)
raw_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HeartDiseaseorAttack</th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>Diabetes</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>Veggies</th>
      <th>HvyAlcoholConsump</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>40.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.0</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>5.0</td>
      <td>18.0</td>
      <td>15.0</td>
      <td>Yes</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>25.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>28.0</td>
      <td>No</td>
      <td>No</td>
      <td>0.0</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>5.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>Yes</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>27.0</td>
      <td>No</td>
      <td>No</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>24.0</td>
      <td>No</td>
      <td>No</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
binary_map2 = {0: 'Female', 1: 'Male'}
raw_df['Sex'] = raw_df['Sex'].replace(binary_map2)
raw_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HeartDiseaseorAttack</th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>Diabetes</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>Veggies</th>
      <th>HvyAlcoholConsump</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>40.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.0</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>5.0</td>
      <td>18.0</td>
      <td>15.0</td>
      <td>Yes</td>
      <td>Female</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>25.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>28.0</td>
      <td>No</td>
      <td>No</td>
      <td>0.0</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>5.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>Yes</td>
      <td>Female</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>27.0</td>
      <td>No</td>
      <td>No</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>24.0</td>
      <td>No</td>
      <td>No</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
binary_map3 = {0: 'No', 1: 'Type1', 2: 'Type2'}
raw_df['Diabetes'] = raw_df['Diabetes'].replace(binary_map3)
raw_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HeartDiseaseorAttack</th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>Diabetes</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>Veggies</th>
      <th>HvyAlcoholConsump</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>40.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>5.0</td>
      <td>18.0</td>
      <td>15.0</td>
      <td>Yes</td>
      <td>Female</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>25.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>28.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>5.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>Yes</td>
      <td>Female</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>27.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>24.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
binary_map4 = {1: '18 to 24', 2: '25 to 29', 3: '30 to 34',
               4: '35 to 39',5: '40 to 44', 6: '45 to 49', 7: '50 to 54', 8: '55 to 59',
               9: '60 to 64', 10: '65 to 69', 11: '70 to 74', 12: '75 to 79', 13: '80+'}
raw_df['Age'] = raw_df['Age'].replace(binary_map4)
raw_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HeartDiseaseorAttack</th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>Diabetes</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>Veggies</th>
      <th>HvyAlcoholConsump</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>40.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>5.0</td>
      <td>18.0</td>
      <td>15.0</td>
      <td>Yes</td>
      <td>Female</td>
      <td>60 to 64</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>25.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>50 to 54</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>28.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>5.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>Yes</td>
      <td>Female</td>
      <td>60 to 64</td>
      <td>4.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>27.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>70 to 74</td>
      <td>3.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>24.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>70 to 74</td>
      <td>5.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
raw_df.columns
```




    Index(['HeartDiseaseorAttack', 'HighBP', 'HighChol', 'CholCheck', 'BMI',
           'Smoker', 'Stroke', 'Diabetes', 'PhysActivity', 'Fruits', 'Veggies',
           'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
           'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
           'Income'],
          dtype='object')



### Creating Training, Testing, and Validation Sets


```python
train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)
print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)
```

    train_df.shape : (152208, 22)
    val_df.shape : (50736, 22)
    test_df.shape : (50736, 22)
    


```python
train_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HeartDiseaseorAttack</th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>Diabetes</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>Veggies</th>
      <th>HvyAlcoholConsump</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>132776</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>25.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>Type2</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>No</td>
      <td>Female</td>
      <td>55 to 59</td>
      <td>5.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>60629</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>25.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>60 to 64</td>
      <td>6.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>163859</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>28.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>50 to 54</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>179387</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>31.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Male</td>
      <td>55 to 59</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>6258</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>21.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>No</td>
      <td>Female</td>
      <td>25 to 29</td>
      <td>4.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>153576</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>24.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>No</td>
      <td>Female</td>
      <td>35 to 39</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>187540</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>27.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>25 to 29</td>
      <td>4.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>158320</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>23.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Male</td>
      <td>45 to 49</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>185003</th>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>32.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>Male</td>
      <td>60 to 64</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>72397</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>23.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>70 to 74</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
<p>152208 rows × 22 columns</p>
</div>




```python
val_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HeartDiseaseorAttack</th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>Diabetes</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>Veggies</th>
      <th>HvyAlcoholConsump</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>177961</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>28.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>No</td>
      <td>Male</td>
      <td>65 to 69</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>105626</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>27.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Type2</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>4.0</td>
      <td>8.0</td>
      <td>20.0</td>
      <td>Yes</td>
      <td>Female</td>
      <td>65 to 69</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>136759</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>47.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>Male</td>
      <td>40 to 44</td>
      <td>6.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>181637</th>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>26.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Male</td>
      <td>45 to 49</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>245214</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>23.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>Male</td>
      <td>55 to 59</td>
      <td>6.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>250516</th>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>29.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>No</td>
      <td>Male</td>
      <td>55 to 59</td>
      <td>6.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>161301</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>22.0</td>
      <td>No</td>
      <td>No</td>
      <td>Type2</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>No</td>
      <td>Female</td>
      <td>60 to 64</td>
      <td>5.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>31718</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>24.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>No</td>
      <td>Female</td>
      <td>55 to 59</td>
      <td>6.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>152320</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>26.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>60 to 64</td>
      <td>5.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>87111</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>33.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>45 to 49</td>
      <td>5.0</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
<p>50736 rows × 22 columns</p>
</div>




```python
test_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HeartDiseaseorAttack</th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>Diabetes</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>Veggies</th>
      <th>HvyAlcoholConsump</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>219620</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>21.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>No</td>
      <td>Female</td>
      <td>50 to 54</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>132821</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>28.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>80+</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>151862</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>24.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Male</td>
      <td>18 to 24</td>
      <td>4.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>139717</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>27.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Male</td>
      <td>25 to 29</td>
      <td>4.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>239235</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>31.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>4.0</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>Yes</td>
      <td>Female</td>
      <td>55 to 59</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>169513</th>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>29.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>Type2</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>No</td>
      <td>Female</td>
      <td>60 to 64</td>
      <td>6.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>182415</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>25.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>No</td>
      <td>Female</td>
      <td>65 to 69</td>
      <td>5.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>109739</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>28.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Male</td>
      <td>45 to 49</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>181671</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>24.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Male</td>
      <td>80+</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>202118</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>23.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>40 to 44</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
<p>50736 rows × 22 columns</p>
</div>



### Identifying Features and Targets


```python
input_cols = list(train_df.columns)[1:]
target_col = 'HeartDiseaseorAttack'
print(input_cols)
target_col
```

    ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Diabetes', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
    




    'HeartDiseaseorAttack'



### Copying Inputs and Targets for Further Processing


```python
train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()
train_inputs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>Diabetes</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>Veggies</th>
      <th>HvyAlcoholConsump</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>132776</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>25.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>Type2</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>No</td>
      <td>Female</td>
      <td>55 to 59</td>
      <td>5.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>60629</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>25.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>60 to 64</td>
      <td>6.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>163859</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>28.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>50 to 54</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>179387</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>31.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Male</td>
      <td>55 to 59</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>6258</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>21.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>No</td>
      <td>Female</td>
      <td>25 to 29</td>
      <td>4.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>153576</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>24.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>No</td>
      <td>Female</td>
      <td>35 to 39</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>187540</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>27.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>25 to 29</td>
      <td>4.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>158320</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>23.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Male</td>
      <td>45 to 49</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>185003</th>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>32.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>Male</td>
      <td>60 to 64</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>72397</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>23.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>2.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>No</td>
      <td>Female</td>
      <td>70 to 74</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
<p>152208 rows × 21 columns</p>
</div>




```python
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()
train_inputs[numeric_cols].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BMI</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>152208.000000</td>
      <td>152208.000000</td>
      <td>152208.000000</td>
      <td>152208.000000</td>
      <td>152208.000000</td>
      <td>152208.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>28.379566</td>
      <td>2.511471</td>
      <td>3.192598</td>
      <td>4.238266</td>
      <td>5.050891</td>
      <td>6.058847</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.614460</td>
      <td>1.067363</td>
      <td>7.421910</td>
      <td>8.703617</td>
      <td>0.987088</td>
      <td>2.069661</td>
    </tr>
    <tr>
      <th>min</th>
      <td>12.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>24.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>27.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>31.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>98.000000</td>
      <td>5.000000</td>
      <td>30.000000</td>
      <td>30.000000</td>
      <td>6.000000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Checking if Feature Scaling is Necessary


```python
raw_df[numeric_cols].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BMI</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
      <td>253680.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>28.382364</td>
      <td>2.511392</td>
      <td>3.184772</td>
      <td>4.242081</td>
      <td>5.050434</td>
      <td>6.053875</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.608694</td>
      <td>1.068477</td>
      <td>7.412847</td>
      <td>8.717951</td>
      <td>0.985774</td>
      <td>2.071148</td>
    </tr>
    <tr>
      <th>min</th>
      <td>12.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>24.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>27.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>31.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>98.000000</td>
      <td>5.000000</td>
      <td>30.000000</td>
      <td>30.000000</td>
      <td>6.000000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>



Scaling numeric features to smaller ranges


```python
scaler = MinMaxScaler()
```


```python
scaler.fit(raw_df[numeric_cols])
```




<style>#sk-container-id-14 {color: black;}#sk-container-id-14 pre{padding: 0;}#sk-container-id-14 div.sk-toggleable {background-color: white;}#sk-container-id-14 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-14 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-14 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-14 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-14 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-14 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-14 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-14 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-14 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-14 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-14 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-14 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-14 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-14 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-14 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-14 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-14 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-14 div.sk-item {position: relative;z-index: 1;}#sk-container-id-14 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-14 div.sk-item::before, #sk-container-id-14 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-14 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-14 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-14 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-14 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-14 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-14 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-14 div.sk-label-container {text-align: center;}#sk-container-id-14 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-14 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-14" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>MinMaxScaler()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" checked><label for="sk-estimator-id-14" class="sk-toggleable__label sk-toggleable__label-arrow">MinMaxScaler</label><div class="sk-toggleable__content"><pre>MinMaxScaler()</pre></div></div></div></div></div>



Inspecting min and max values for each column


```python
print('Minimum:')
list(scaler.data_min_)
```

    Minimum:
    




    [12.0, 1.0, 0.0, 0.0, 1.0, 1.0]




```python
print('Maximum:')
list(scaler.data_max_)
```

    Maximum:
    




    [98.0, 5.0, 30.0, 30.0, 6.0, 8.0]



### Scaling the Train, Test and Val sets with scaler.transform


```python
train_inputs.columns
```




    Index(['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
           'Diabetes', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
           'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth',
           'DiffWalk', 'Sex', 'Age', 'Education', 'Income'],
          dtype='object')




```python
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])
train_inputs[numeric_cols].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BMI</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>152208.000000</td>
      <td>152208.000000</td>
      <td>152208.000000</td>
      <td>152208.000000</td>
      <td>152208.000000</td>
      <td>152208.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.190460</td>
      <td>0.377868</td>
      <td>0.106420</td>
      <td>0.141276</td>
      <td>0.810178</td>
      <td>0.722692</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.076912</td>
      <td>0.266841</td>
      <td>0.247397</td>
      <td>0.290121</td>
      <td>0.197418</td>
      <td>0.295666</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.139535</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.600000</td>
      <td>0.571429</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.174419</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.800000</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.220930</td>
      <td>0.500000</td>
      <td>0.066667</td>
      <td>0.100000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('train_inputs:', train_inputs.shape)
print('train_targets:', train_targets.shape)
print('val_inputs:', val_inputs.shape)
print('val_targets:', val_targets.shape)
print('test_inputs:', test_inputs.shape)
print('test_targets:', test_targets.shape)
```

    train_inputs: (152208, 21)
    train_targets: (152208,)
    val_inputs: (50736, 21)
    val_targets: (50736,)
    test_inputs: (50736, 21)
    test_targets: (50736,)
    

### Encoding Categorical Data


```python
raw_df[categorical_cols].nunique()
```




    HighBP                2
    HighChol              2
    CholCheck             2
    Smoker                2
    Stroke                2
    Diabetes              3
    PhysActivity          2
    Fruits                2
    Veggies               2
    HvyAlcoholConsump     2
    AnyHealthcare         2
    NoDocbcCost           2
    DiffWalk              2
    Sex                   2
    Age                  13
    dtype: int64




```python
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
```


```python
encoder.fit(raw_df[categorical_cols])
```




<style>#sk-container-id-15 {color: black;}#sk-container-id-15 pre{padding: 0;}#sk-container-id-15 div.sk-toggleable {background-color: white;}#sk-container-id-15 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-15 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-15 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-15 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-15 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-15 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-15 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-15 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-15 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-15 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-15 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-15 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-15 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-15 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-15 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-15 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-15 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-15 div.sk-item {position: relative;z-index: 1;}#sk-container-id-15 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-15 div.sk-item::before, #sk-container-id-15 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-15 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-15 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-15 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-15 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-15 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-15 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-15 div.sk-label-container {text-align: center;}#sk-container-id-15 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-15 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-15" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;, sparse_output=False)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" checked><label for="sk-estimator-id-15" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;, sparse_output=False)</pre></div></div></div></div></div>




```python
encoder.categories_
```




    [array(['No', 'Yes'], dtype=object),
     array(['No', 'Yes'], dtype=object),
     array(['No', 'Yes'], dtype=object),
     array(['No', 'Yes'], dtype=object),
     array(['No', 'Yes'], dtype=object),
     array(['No', 'Type1', 'Type2'], dtype=object),
     array(['No', 'Yes'], dtype=object),
     array(['No', 'Yes'], dtype=object),
     array(['No', 'Yes'], dtype=object),
     array(['No', 'Yes'], dtype=object),
     array(['No', 'Yes'], dtype=object),
     array(['No', 'Yes'], dtype=object),
     array(['No', 'Yes'], dtype=object),
     array(['Female', 'Male'], dtype=object),
     array(['18 to 24', '25 to 29', '30 to 34', '35 to 39', '40 to 44',
            '45 to 49', '50 to 54', '55 to 59', '60 to 64', '65 to 69',
            '70 to 74', '75 to 79', '80+'], dtype=object)]




```python
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
print(encoded_cols)
```

    ['HighBP_No', 'HighBP_Yes', 'HighChol_No', 'HighChol_Yes', 'CholCheck_No', 'CholCheck_Yes', 'Smoker_No', 'Smoker_Yes', 'Stroke_No', 'Stroke_Yes', 'Diabetes_No', 'Diabetes_Type1', 'Diabetes_Type2', 'PhysActivity_No', 'PhysActivity_Yes', 'Fruits_No', 'Fruits_Yes', 'Veggies_No', 'Veggies_Yes', 'HvyAlcoholConsump_No', 'HvyAlcoholConsump_Yes', 'AnyHealthcare_No', 'AnyHealthcare_Yes', 'NoDocbcCost_No', 'NoDocbcCost_Yes', 'DiffWalk_No', 'DiffWalk_Yes', 'Sex_Female', 'Sex_Male', 'Age_18 to 24', 'Age_25 to 29', 'Age_30 to 34', 'Age_35 to 39', 'Age_40 to 44', 'Age_45 to 49', 'Age_50 to 54', 'Age_55 to 59', 'Age_60 to 64', 'Age_65 to 69', 'Age_70 to 74', 'Age_75 to 79', 'Age_80+']
    


```python
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])
```


```python
pd.set_option('display.max_columns', None)
test_inputs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>Diabetes</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>Veggies</th>
      <th>HvyAlcoholConsump</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
      <th>HighBP_No</th>
      <th>HighBP_Yes</th>
      <th>HighChol_No</th>
      <th>HighChol_Yes</th>
      <th>CholCheck_No</th>
      <th>CholCheck_Yes</th>
      <th>Smoker_No</th>
      <th>Smoker_Yes</th>
      <th>Stroke_No</th>
      <th>Stroke_Yes</th>
      <th>Diabetes_No</th>
      <th>Diabetes_Type1</th>
      <th>Diabetes_Type2</th>
      <th>PhysActivity_No</th>
      <th>PhysActivity_Yes</th>
      <th>Fruits_No</th>
      <th>Fruits_Yes</th>
      <th>Veggies_No</th>
      <th>Veggies_Yes</th>
      <th>HvyAlcoholConsump_No</th>
      <th>HvyAlcoholConsump_Yes</th>
      <th>AnyHealthcare_No</th>
      <th>AnyHealthcare_Yes</th>
      <th>NoDocbcCost_No</th>
      <th>NoDocbcCost_Yes</th>
      <th>DiffWalk_No</th>
      <th>DiffWalk_Yes</th>
      <th>Sex_Female</th>
      <th>Sex_Male</th>
      <th>Age_18 to 24</th>
      <th>Age_25 to 29</th>
      <th>Age_30 to 34</th>
      <th>Age_35 to 39</th>
      <th>Age_40 to 44</th>
      <th>Age_45 to 49</th>
      <th>Age_50 to 54</th>
      <th>Age_55 to 59</th>
      <th>Age_60 to 64</th>
      <th>Age_65 to 69</th>
      <th>Age_70 to 74</th>
      <th>Age_75 to 79</th>
      <th>Age_80+</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>219620</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.104651</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.50</td>
      <td>0.100000</td>
      <td>0.233333</td>
      <td>No</td>
      <td>Female</td>
      <td>50 to 54</td>
      <td>0.6</td>
      <td>0.142857</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>132821</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0.186047</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.50</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>No</td>
      <td>Female</td>
      <td>80+</td>
      <td>1.0</td>
      <td>0.714286</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>151862</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.139535</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>No</td>
      <td>Male</td>
      <td>18 to 24</td>
      <td>0.6</td>
      <td>0.857143</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>139717</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.174419</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.25</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>No</td>
      <td>Male</td>
      <td>25 to 29</td>
      <td>0.6</td>
      <td>0.857143</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>239235</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0.220930</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0.75</td>
      <td>0.900000</td>
      <td>0.900000</td>
      <td>Yes</td>
      <td>Female</td>
      <td>55 to 59</td>
      <td>0.4</td>
      <td>0.142857</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>169513</th>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.197674</td>
      <td>Yes</td>
      <td>No</td>
      <td>Type2</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.50</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>No</td>
      <td>Female</td>
      <td>60 to 64</td>
      <td>1.0</td>
      <td>0.857143</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>182415</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.151163</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.25</td>
      <td>0.033333</td>
      <td>0.333333</td>
      <td>No</td>
      <td>Female</td>
      <td>65 to 69</td>
      <td>0.8</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>109739</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0.186047</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.50</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>No</td>
      <td>Male</td>
      <td>45 to 49</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>181671</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.139535</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0.75</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>No</td>
      <td>Male</td>
      <td>80+</td>
      <td>0.6</td>
      <td>0.571429</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>202118</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.127907</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.25</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>No</td>
      <td>Female</td>
      <td>40 to 44</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>50736 rows × 63 columns</p>
</div>




```python
print('train_inputs:', train_inputs.shape)
print('train_targets:', train_targets.shape)
print('val_inputs:', val_inputs.shape)
print('val_targets:', val_targets.shape)
print('test_inputs:', test_inputs.shape)
print('test_targets:', test_targets.shape)
```

    train_inputs: (152208, 63)
    train_targets: (152208,)
    val_inputs: (50736, 63)
    val_targets: (50736,)
    test_inputs: (50736, 63)
    test_targets: (50736,)
    

#Saving Processed Data As Parquet Files for Optimal Storage


```python
%%time
train_inputs.to_parquet('data/train_inputs.parquet')
val_inputs.to_parquet('data/val_inputs.parquet')
test_inputs.to_parquet('data/test_inputs.parquet')

pd.DataFrame(train_targets).to_parquet('data/train_targets.parquet')
pd.DataFrame(val_targets).to_parquet('data/val_targets.parquet')
pd.DataFrame(test_targets).to_parquet('data/test_targets.parquet')
```

    CPU times: total: 438 ms
    Wall time: 1.37 s
    


```python
%%time

train_inputs = pd.read_parquet('data/train_inputs.parquet')
val_inputs = pd.read_parquet('data/val_inputs.parquet')
test_inputs = pd.read_parquet('data/test_inputs.parquet')

train_targets = pd.read_parquet('data/train_targets.parquet')[target_col]
val_targets = pd.read_parquet('data/val_targets.parquet')[target_col]
test_targets = pd.read_parquet('data/test_targets.parquet')[target_col]
```

    CPU times: total: 62.5 ms
    Wall time: 335 ms
    


```python
train_inputs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>Diabetes</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>Veggies</th>
      <th>HvyAlcoholConsump</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
      <th>HighBP_No</th>
      <th>HighBP_Yes</th>
      <th>HighChol_No</th>
      <th>HighChol_Yes</th>
      <th>CholCheck_No</th>
      <th>CholCheck_Yes</th>
      <th>Smoker_No</th>
      <th>Smoker_Yes</th>
      <th>Stroke_No</th>
      <th>Stroke_Yes</th>
      <th>Diabetes_No</th>
      <th>Diabetes_Type1</th>
      <th>Diabetes_Type2</th>
      <th>PhysActivity_No</th>
      <th>PhysActivity_Yes</th>
      <th>Fruits_No</th>
      <th>Fruits_Yes</th>
      <th>Veggies_No</th>
      <th>Veggies_Yes</th>
      <th>HvyAlcoholConsump_No</th>
      <th>HvyAlcoholConsump_Yes</th>
      <th>AnyHealthcare_No</th>
      <th>AnyHealthcare_Yes</th>
      <th>NoDocbcCost_No</th>
      <th>NoDocbcCost_Yes</th>
      <th>DiffWalk_No</th>
      <th>DiffWalk_Yes</th>
      <th>Sex_Female</th>
      <th>Sex_Male</th>
      <th>Age_18 to 24</th>
      <th>Age_25 to 29</th>
      <th>Age_30 to 34</th>
      <th>Age_35 to 39</th>
      <th>Age_40 to 44</th>
      <th>Age_45 to 49</th>
      <th>Age_50 to 54</th>
      <th>Age_55 to 59</th>
      <th>Age_60 to 64</th>
      <th>Age_65 to 69</th>
      <th>Age_70 to 74</th>
      <th>Age_75 to 79</th>
      <th>Age_80+</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>132776</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.151163</td>
      <td>Yes</td>
      <td>No</td>
      <td>Type2</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.033333</td>
      <td>No</td>
      <td>Female</td>
      <td>55 to 59</td>
      <td>0.8</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>60629</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.151163</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>No</td>
      <td>Female</td>
      <td>60 to 64</td>
      <td>1.0</td>
      <td>0.857143</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>163859</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.186047</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>No</td>
      <td>Female</td>
      <td>50 to 54</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>179387</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.220930</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>No</td>
      <td>Male</td>
      <td>55 to 59</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6258</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.104651</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.066667</td>
      <td>No</td>
      <td>Female</td>
      <td>25 to 29</td>
      <td>0.6</td>
      <td>0.714286</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>153576</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0.139535</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.066667</td>
      <td>No</td>
      <td>Female</td>
      <td>35 to 39</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>187540</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.174419</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>0.50</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>No</td>
      <td>Female</td>
      <td>25 to 29</td>
      <td>0.6</td>
      <td>0.714286</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>158320</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.127907</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>No</td>
      <td>Male</td>
      <td>45 to 49</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>185003</th>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.232558</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.50</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>Yes</td>
      <td>Male</td>
      <td>60 to 64</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>72397</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0.127907</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>No</td>
      <td>Female</td>
      <td>70 to 74</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>152208 rows × 63 columns</p>
</div>




```python
val_inputs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>Diabetes</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>Veggies</th>
      <th>HvyAlcoholConsump</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
      <th>HighBP_No</th>
      <th>HighBP_Yes</th>
      <th>HighChol_No</th>
      <th>HighChol_Yes</th>
      <th>CholCheck_No</th>
      <th>CholCheck_Yes</th>
      <th>Smoker_No</th>
      <th>Smoker_Yes</th>
      <th>Stroke_No</th>
      <th>Stroke_Yes</th>
      <th>Diabetes_No</th>
      <th>Diabetes_Type1</th>
      <th>Diabetes_Type2</th>
      <th>PhysActivity_No</th>
      <th>PhysActivity_Yes</th>
      <th>Fruits_No</th>
      <th>Fruits_Yes</th>
      <th>Veggies_No</th>
      <th>Veggies_Yes</th>
      <th>HvyAlcoholConsump_No</th>
      <th>HvyAlcoholConsump_Yes</th>
      <th>AnyHealthcare_No</th>
      <th>AnyHealthcare_Yes</th>
      <th>NoDocbcCost_No</th>
      <th>NoDocbcCost_Yes</th>
      <th>DiffWalk_No</th>
      <th>DiffWalk_Yes</th>
      <th>Sex_Female</th>
      <th>Sex_Male</th>
      <th>Age_18 to 24</th>
      <th>Age_25 to 29</th>
      <th>Age_30 to 34</th>
      <th>Age_35 to 39</th>
      <th>Age_40 to 44</th>
      <th>Age_45 to 49</th>
      <th>Age_50 to 54</th>
      <th>Age_55 to 59</th>
      <th>Age_60 to 64</th>
      <th>Age_65 to 69</th>
      <th>Age_70 to 74</th>
      <th>Age_75 to 79</th>
      <th>Age_80+</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>177961</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0.186047</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.25</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>No</td>
      <td>Male</td>
      <td>65 to 69</td>
      <td>0.6</td>
      <td>0.571429</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>105626</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0.174419</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Type2</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.75</td>
      <td>0.266667</td>
      <td>0.666667</td>
      <td>Yes</td>
      <td>Female</td>
      <td>65 to 69</td>
      <td>0.6</td>
      <td>0.142857</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>136759</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0.406977</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.50</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>Yes</td>
      <td>Male</td>
      <td>40 to 44</td>
      <td>1.0</td>
      <td>0.571429</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>181637</th>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.162791</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>No</td>
      <td>Male</td>
      <td>45 to 49</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>245214</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.127907</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.25</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>Yes</td>
      <td>Male</td>
      <td>55 to 59</td>
      <td>1.0</td>
      <td>0.857143</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>250516</th>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.197674</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.25</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>No</td>
      <td>Male</td>
      <td>55 to 59</td>
      <td>1.0</td>
      <td>0.857143</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>161301</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0.116279</td>
      <td>No</td>
      <td>No</td>
      <td>Type2</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.25</td>
      <td>0.066667</td>
      <td>0.066667</td>
      <td>No</td>
      <td>Female</td>
      <td>60 to 64</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>31718</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.139535</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.25</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>No</td>
      <td>Female</td>
      <td>55 to 59</td>
      <td>1.0</td>
      <td>0.857143</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>152320</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0.162791</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.00</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>No</td>
      <td>Female</td>
      <td>60 to 64</td>
      <td>0.8</td>
      <td>0.857143</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>87111</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0.244186</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.50</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>No</td>
      <td>Female</td>
      <td>45 to 49</td>
      <td>0.8</td>
      <td>0.714286</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>50736 rows × 63 columns</p>
</div>



### Training the Regression Model


```python
model = LogisticRegression(solver='liblinear')
```


```python
model.fit(train_inputs[numeric_cols + encoded_cols], train_targets)
```




<style>#sk-container-id-16 {color: black;}#sk-container-id-16 pre{padding: 0;}#sk-container-id-16 div.sk-toggleable {background-color: white;}#sk-container-id-16 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-16 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-16 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-16 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-16 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-16 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-16 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-16 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-16 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-16 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-16 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-16 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-16 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-16 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-16 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-16 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-16 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-16 div.sk-item {position: relative;z-index: 1;}#sk-container-id-16 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-16 div.sk-item::before, #sk-container-id-16 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-16 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-16 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-16 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-16 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-16 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-16 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-16 div.sk-label-container {text-align: center;}#sk-container-id-16 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-16 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-16" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(solver=&#x27;liblinear&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-16" type="checkbox" checked><label for="sk-estimator-id-16" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(solver=&#x27;liblinear&#x27;)</pre></div></div></div></div></div>




```python
print(numeric_cols + encoded_cols)
```

    ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Education', 'Income', 'HighBP_No', 'HighBP_Yes', 'HighChol_No', 'HighChol_Yes', 'CholCheck_No', 'CholCheck_Yes', 'Smoker_No', 'Smoker_Yes', 'Stroke_No', 'Stroke_Yes', 'Diabetes_No', 'Diabetes_Type1', 'Diabetes_Type2', 'PhysActivity_No', 'PhysActivity_Yes', 'Fruits_No', 'Fruits_Yes', 'Veggies_No', 'Veggies_Yes', 'HvyAlcoholConsump_No', 'HvyAlcoholConsump_Yes', 'AnyHealthcare_No', 'AnyHealthcare_Yes', 'NoDocbcCost_No', 'NoDocbcCost_Yes', 'DiffWalk_No', 'DiffWalk_Yes', 'Sex_Female', 'Sex_Male', 'Age_18 to 24', 'Age_25 to 29', 'Age_30 to 34', 'Age_35 to 39', 'Age_40 to 44', 'Age_45 to 49', 'Age_50 to 54', 'Age_55 to 59', 'Age_60 to 64', 'Age_65 to 69', 'Age_70 to 74', 'Age_75 to 79', 'Age_80+']
    


```python
#higher weights the greater impact on predictions
print(model.coef_.tolist())
```

    [[0.06837168233845196, 1.9829660878849276, 0.06884743385008901, 0.013986704150432506, 0.07746985646872094, -0.32959946081554337, -0.48478529258801495, 0.047426076037158815, -0.5106042978151092, 0.07324508127108663, -0.41888620715730035, -0.018473009395780726, -0.3976898621504877, -0.03966935439685527, -0.7062132675731577, 0.2688540510226979, -0.2632130562169323, -0.22507910986859872, 0.0509329495309535, -0.24576940838308015, -0.19158980817737004, -0.21876394175133856, -0.21859527480076815, -0.24960824971183898, -0.18775096683963338, -0.047489229268531864, -0.38986998728299, -0.2071476866006668, -0.23021152994782043, -0.3441366470619852, -0.09322256948943805, -0.36632580606214526, -0.07103341049600201, -0.5828700645478256, 0.14551084800184833, -1.7482487472366113, -1.2002006123817965, -1.0167969142186655, -0.9241296477321187, -0.5605242077169111, -0.20827890610990402, 0.010407777941339303, 0.16730949635744496, 0.501676118515311, 0.7756982824574709, 1.055405052966805, 1.214848412173744, 1.4954746784354858]]
    


```python
print(model.intercept_)
```

    [-0.43735922]
    

### Make Predictions on Training Test Set


```python
X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]
```


```python
train_preds = model.predict(X_train)
```


```python
train_preds
```




    array(['No', 'No', 'No', ..., 'No', 'No', 'No'], dtype=object)




```python
train_targets
```




    132776    No
    60629     No
    163859    No
    179387    No
    6258      No
              ..
    153576    No
    187540    No
    158320    No
    185003    No
    72397     No
    Name: HeartDiseaseorAttack, Length: 152208, dtype: object




```python
#probablistic prediction with predict_proba
train_probs = model.predict_proba(X_train)
train_probs
```




    array([[0.98033226, 0.01966774],
           [0.9850001 , 0.0149999 ],
           [0.99103061, 0.00896939],
           ...,
           [0.9793025 , 0.0206975 ],
           [0.86014228, 0.13985772],
           [0.8944115 , 0.1055885 ]])




```python
model.classes_
```




    array(['No', 'Yes'], dtype=object)




```python
acc = accuracy_score(train_targets, train_preds)
print("Accuracy: {:.2f}%".format(acc * 100))
```

    Accuracy: 90.71%
    

This model receives a 90.71% Accuracy


```python
confusion_matrix(train_targets, train_preds, normalize='true')
```




    array([[0.98874586, 0.01125414],
           [0.87506948, 0.12493052]])




```python
def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)

    accuracy = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sn.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name));

    return preds
```

### Training Predictions


```python
train_preds = predict_and_plot(X_train, train_targets, 'Training')
```

    Accuracy: 90.71%
    






### Validations Predictions


```python
val_preds = predict_and_plot(X_val, val_targets, 'Validatiaon')
```

    Accuracy: 90.87%
    





### Test Predictions


```python
test_preds = predict_and_plot(X_test, test_targets, 'Test')
```

    Accuracy: 90.84%
    




The model performs slightly above 90% accuracy

### Comparing Results to a "Random Model" to verify quality
One model will guess randomly and the other will return 'No' no matter what



```python
def random_guess(inputs):
    return np.random.choice(["No", "Yes"], len(inputs))
```


```python
def all_no(inputs):
    return np.full(len(inputs), "No")
```


```python
accuracy_score(test_targets, random_guess(X_test))
```




    0.5009854935351624




```python
accuracy_score(test_targets, all_no(X_test))
```




    0.9060233364869127



### Making Predictions on a Random Input


```python
rand_input = {'HighBP' : np.random.choice(["No", "Yes"]),
              'HighChol' : np.random.choice(["No", "Yes"]),
              'CholCheck' : np.random.choice(["No", "Yes"]),
              'BMI' : np.random.choice(range(1,100)),
              'Smoker' : np.random.choice(["No", "Yes"]),
              'Stroke' : np.random.choice(["No", "Yes"]),
              'Diabetes' : np.random.choice(["No", "Type1", "Type2"]),
              'PhysActivity' : np.random.choice(["No", "Yes"]),
              'Fruits' : np.random.choice(["No", "Yes"]),
              'Veggies' : np.random.choice(["No", "Yes"]),
              'HvyAlcoholConsump' : np.random.choice(["No", "Yes"]),
              'AnyHealthcare' : np.random.choice(["No", "Yes"]),
              'NoDocbcCost' : np.random.choice(["No", "Yes"]),
              'GenHlth' : np.random.choice(range(1,5)),
              'MentHlth' : np.random.choice(range(0,31)),
              'PhysHlth' : np.random.choice(range(0,31)),
              'DiffWalk' : np.random.choice(["No", "Yes"]),
              'Education' : np.random.choice(range(1,6)),
              'Income' : np.random.choice(range(1,8)),
              'Age' : np.random.choice(["50 to 54", "80+", "18 to 24", "25 to 29", "55 to 59","65 to 69", "70 to 74", "30 to 34","35 to 39", "75 to 79"]),
              'Sex' : np.random.choice(["Female", "Male"])
}

rand_input_df = pd.DataFrame([rand_input])
rand_input_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>Diabetes</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>Veggies</th>
      <th>HvyAlcoholConsump</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Education</th>
      <th>Income</th>
      <th>Age</th>
      <th>Sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>46</td>
      <td>Yes</td>
      <td>No</td>
      <td>Type1</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>4</td>
      <td>13</td>
      <td>11</td>
      <td>Yes</td>
      <td>2</td>
      <td>1</td>
      <td>35 to 39</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>




```python
rand_input_df[numeric_cols] = scaler.transform(rand_input_df[numeric_cols])
rand_input_df[encoded_cols] = encoder.transform(rand_input_df[categorical_cols])
```


```python
X_new_input = rand_input_df[numeric_cols + encoded_cols]
X_new_input
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BMI</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>Education</th>
      <th>Income</th>
      <th>HighBP_No</th>
      <th>HighBP_Yes</th>
      <th>HighChol_No</th>
      <th>HighChol_Yes</th>
      <th>CholCheck_No</th>
      <th>CholCheck_Yes</th>
      <th>Smoker_No</th>
      <th>Smoker_Yes</th>
      <th>Stroke_No</th>
      <th>Stroke_Yes</th>
      <th>Diabetes_No</th>
      <th>Diabetes_Type1</th>
      <th>Diabetes_Type2</th>
      <th>PhysActivity_No</th>
      <th>PhysActivity_Yes</th>
      <th>Fruits_No</th>
      <th>Fruits_Yes</th>
      <th>Veggies_No</th>
      <th>Veggies_Yes</th>
      <th>HvyAlcoholConsump_No</th>
      <th>HvyAlcoholConsump_Yes</th>
      <th>AnyHealthcare_No</th>
      <th>AnyHealthcare_Yes</th>
      <th>NoDocbcCost_No</th>
      <th>NoDocbcCost_Yes</th>
      <th>DiffWalk_No</th>
      <th>DiffWalk_Yes</th>
      <th>Sex_Female</th>
      <th>Sex_Male</th>
      <th>Age_18 to 24</th>
      <th>Age_25 to 29</th>
      <th>Age_30 to 34</th>
      <th>Age_35 to 39</th>
      <th>Age_40 to 44</th>
      <th>Age_45 to 49</th>
      <th>Age_50 to 54</th>
      <th>Age_55 to 59</th>
      <th>Age_60 to 64</th>
      <th>Age_65 to 69</th>
      <th>Age_70 to 74</th>
      <th>Age_75 to 79</th>
      <th>Age_80+</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.395349</td>
      <td>0.75</td>
      <td>0.433333</td>
      <td>0.366667</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Making Prediction on Random Input


```python
prediction = model.predict(X_new_input)[0]
prediction
```




    'No'



Probability of the above prediction


```python
prob = model.predict_proba(X_new_input)[0]
prob
```




    array([0.93933341, 0.06066659])




```python
def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob
```


```python
random_df = pd.DataFrame(predict_input(rand_input))
random_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.939333</td>
    </tr>
  </tbody>
</table>
</div>




```python
random_df.to_csv('data/random_prediction.csv')
```


```python
heart_predictor = {
    'model': model,
    'scaler': scaler,
    'encoder': encoder,
    'input_cols': input_cols,
    'target_col': target_col,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'encoded_cols': encoded_cols
}
```


```python
joblib.dump(heart_predictor, 'heart_predictor.joblib')
```




    ['heart_predictor.joblib']




```python
heart_predictor2 = joblib.load('heart_predictor.joblib')

test_preds2 = heart_predictor2['model'].predict(X_test)
accuracy_score(test_targets, test_preds2)
```




    0.9084082308420057




```python

```
