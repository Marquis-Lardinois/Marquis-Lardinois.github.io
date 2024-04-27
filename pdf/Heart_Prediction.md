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


<div>                            <div id="2839a2d3-d941-4d43-b826-16c27900f60f" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("2839a2d3-d941-4d43-b826-16c27900f60f")) {                    Plotly.newPlot(                        "2839a2d3-d941-4d43-b826-16c27900f60f",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003elabel=%{label}\u003cbr\u003evalue=%{value}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["No Heart Disease","Heart Disease"],"labels":["No Heart Disease","Heart Disease"],"legendgroup":"","name":"","showlegend":true,"values":[229787,23893],"type":"pie","text":["No Heart Disease","Heart Disease"],"textinfo":"percent+label"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"marker":{"line":{"color":"#283442"}},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#f2f5fa"},"error_y":{"color":"#f2f5fa"},"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"marker":{"line":{"color":"#283442"}},"type":"scattergl"}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"baxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#506784"},"line":{"color":"rgb(17,17,17)"}},"header":{"fill":{"color":"#2a3f5f"},"line":{"color":"rgb(17,17,17)"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#f2f5fa"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"rgb(17,17,17)","plot_bgcolor":"rgb(17,17,17)","polar":{"bgcolor":"rgb(17,17,17)","angularaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"radialaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"ternary":{"bgcolor":"rgb(17,17,17)","aaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"baxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"caxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2},"yaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2},"zaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2}},"shapedefaults":{"line":{"color":"#f2f5fa"}},"annotationdefaults":{"arrowcolor":"#f2f5fa","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"rgb(17,17,17)","landcolor":"rgb(17,17,17)","subunitcolor":"#506784","showland":true,"showlakes":true,"lakecolor":"rgb(17,17,17)"},"title":{"x":0.05},"updatemenudefaults":{"bgcolor":"#506784","borderwidth":0},"sliderdefaults":{"bgcolor":"#C8D4E3","borderwidth":1,"bordercolor":"rgb(17,17,17)","tickwidth":0},"mapbox":{"style":"dark"}}},"legend":{"tracegroupgap":0},"title":{"text":"Distribution of Heart Diagnosis"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('2839a2d3-d941-4d43-b826-16c27900f60f');
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


<div>                            <div id="cc369c62-6bdd-4e2a-a810-433038e28b4e" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("cc369c62-6bdd-4e2a-a810-433038e28b4e")) {                    Plotly.newPlot(                        "cc369c62-6bdd-4e2a-a810-433038e28b4e",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003elabel=%{label}\u003cbr\u003evalue=%{value}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["No","Yes"],"labels":["No","Yes"],"legendgroup":"","name":"","showlegend":true,"values":[144851,108829],"type":"pie","text":["No","Yes"],"textinfo":"percent+label"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"marker":{"line":{"color":"#283442"}},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#f2f5fa"},"error_y":{"color":"#f2f5fa"},"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"marker":{"line":{"color":"#283442"}},"type":"scattergl"}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"baxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#506784"},"line":{"color":"rgb(17,17,17)"}},"header":{"fill":{"color":"#2a3f5f"},"line":{"color":"rgb(17,17,17)"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#f2f5fa"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"rgb(17,17,17)","plot_bgcolor":"rgb(17,17,17)","polar":{"bgcolor":"rgb(17,17,17)","angularaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"radialaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"ternary":{"bgcolor":"rgb(17,17,17)","aaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"baxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"caxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2},"yaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2},"zaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2}},"shapedefaults":{"line":{"color":"#f2f5fa"}},"annotationdefaults":{"arrowcolor":"#f2f5fa","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"rgb(17,17,17)","landcolor":"rgb(17,17,17)","subunitcolor":"#506784","showland":true,"showlakes":true,"lakecolor":"rgb(17,17,17)"},"title":{"x":0.05},"updatemenudefaults":{"bgcolor":"#506784","borderwidth":0},"sliderdefaults":{"bgcolor":"#C8D4E3","borderwidth":1,"bordercolor":"rgb(17,17,17)","tickwidth":0},"mapbox":{"style":"dark"}}},"legend":{"tracegroupgap":0},"title":{"text":"Distribution of HighBP"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('cc369c62-6bdd-4e2a-a810-433038e28b4e');
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



<div>                            <div id="acfaea62-73a3-47a6-80af-deeeaca6a8dd" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("acfaea62-73a3-47a6-80af-deeeaca6a8dd")) {                    Plotly.newPlot(                        "acfaea62-73a3-47a6-80af-deeeaca6a8dd",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003elabel=%{label}\u003cbr\u003evalue=%{value}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["No","Yes"],"labels":["No","Yes"],"legendgroup":"","name":"","showlegend":true,"values":[141257,112423],"type":"pie","text":["No","Yes"],"textinfo":"percent+label"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"marker":{"line":{"color":"#283442"}},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#f2f5fa"},"error_y":{"color":"#f2f5fa"},"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"marker":{"line":{"color":"#283442"}},"type":"scattergl"}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"baxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#506784"},"line":{"color":"rgb(17,17,17)"}},"header":{"fill":{"color":"#2a3f5f"},"line":{"color":"rgb(17,17,17)"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#f2f5fa"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"rgb(17,17,17)","plot_bgcolor":"rgb(17,17,17)","polar":{"bgcolor":"rgb(17,17,17)","angularaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"radialaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"ternary":{"bgcolor":"rgb(17,17,17)","aaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"baxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"caxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2},"yaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2},"zaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2}},"shapedefaults":{"line":{"color":"#f2f5fa"}},"annotationdefaults":{"arrowcolor":"#f2f5fa","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"rgb(17,17,17)","landcolor":"rgb(17,17,17)","subunitcolor":"#506784","showland":true,"showlakes":true,"lakecolor":"rgb(17,17,17)"},"title":{"x":0.05},"updatemenudefaults":{"bgcolor":"#506784","borderwidth":0},"sliderdefaults":{"bgcolor":"#C8D4E3","borderwidth":1,"bordercolor":"rgb(17,17,17)","tickwidth":0},"mapbox":{"style":"dark"}}},"legend":{"tracegroupgap":0},"title":{"text":"Distribution of Smoker"}},                        {"responsive": true}                    ).then(function(){

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



<div>                            <div id="14f66727-40f7-481e-868a-1d6a112ce1e1" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("14f66727-40f7-481e-868a-1d6a112ce1e1")) {                    Plotly.newPlot(                        "14f66727-40f7-481e-868a-1d6a112ce1e1",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003elabel=%{label}\u003cbr\u003evalue=%{value}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["No","Yes"],"labels":["No","Yes"],"legendgroup":"","name":"","showlegend":true,"values":[243388,10292],"type":"pie","text":["No","Yes"],"textinfo":"percent+label"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"marker":{"line":{"color":"#283442"}},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#f2f5fa"},"error_y":{"color":"#f2f5fa"},"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"marker":{"line":{"color":"#283442"}},"type":"scattergl"}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"baxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#506784"},"line":{"color":"rgb(17,17,17)"}},"header":{"fill":{"color":"#2a3f5f"},"line":{"color":"rgb(17,17,17)"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#f2f5fa"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"rgb(17,17,17)","plot_bgcolor":"rgb(17,17,17)","polar":{"bgcolor":"rgb(17,17,17)","angularaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"radialaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"ternary":{"bgcolor":"rgb(17,17,17)","aaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"baxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"caxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2},"yaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2},"zaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2}},"shapedefaults":{"line":{"color":"#f2f5fa"}},"annotationdefaults":{"arrowcolor":"#f2f5fa","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"rgb(17,17,17)","landcolor":"rgb(17,17,17)","subunitcolor":"#506784","showland":true,"showlakes":true,"lakecolor":"rgb(17,17,17)"},"title":{"x":0.05},"updatemenudefaults":{"bgcolor":"#506784","borderwidth":0},"sliderdefaults":{"bgcolor":"#C8D4E3","borderwidth":1,"bordercolor":"rgb(17,17,17)","tickwidth":0},"mapbox":{"style":"dark"}}},"legend":{"tracegroupgap":0},"title":{"text":"Distribution of Stroke"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('14f66727-40f7-481e-868a-1d6a112ce1e1');
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



<div>                            <div id="bc9a6147-3131-4381-94be-48f65985e3c4" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("bc9a6147-3131-4381-94be-48f65985e3c4")) {                    Plotly.newPlot(                        "bc9a6147-3131-4381-94be-48f65985e3c4",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003elabel=%{label}\u003cbr\u003evalue=%{value}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["No",null,"Yes"],"labels":["No",null,"Yes"],"legendgroup":"","name":"","showlegend":true,"values":[213703,35346,4631],"type":"pie","text":["No",null,"Yes"],"textinfo":"percent+label"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"marker":{"line":{"color":"#283442"}},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#f2f5fa"},"error_y":{"color":"#f2f5fa"},"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"marker":{"line":{"color":"#283442"}},"type":"scattergl"}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"baxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#506784"},"line":{"color":"rgb(17,17,17)"}},"header":{"fill":{"color":"#2a3f5f"},"line":{"color":"rgb(17,17,17)"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#f2f5fa"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"rgb(17,17,17)","plot_bgcolor":"rgb(17,17,17)","polar":{"bgcolor":"rgb(17,17,17)","angularaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"radialaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"ternary":{"bgcolor":"rgb(17,17,17)","aaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"baxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"caxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2},"yaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2},"zaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2}},"shapedefaults":{"line":{"color":"#f2f5fa"}},"annotationdefaults":{"arrowcolor":"#f2f5fa","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"rgb(17,17,17)","landcolor":"rgb(17,17,17)","subunitcolor":"#506784","showland":true,"showlakes":true,"lakecolor":"rgb(17,17,17)"},"title":{"x":0.05},"updatemenudefaults":{"bgcolor":"#506784","borderwidth":0},"sliderdefaults":{"bgcolor":"#C8D4E3","borderwidth":1,"bordercolor":"rgb(17,17,17)","tickwidth":0},"mapbox":{"style":"dark"}}},"legend":{"tracegroupgap":0},"title":{"text":"Distribution of Diabetes"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('bc9a6147-3131-4381-94be-48f65985e3c4');
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



<div>                            <div id="2adc99df-2178-4b70-bfa0-0c55971c5e63" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("2adc99df-2178-4b70-bfa0-0c55971c5e63")) {                    Plotly.newPlot(                        "2adc99df-2178-4b70-bfa0-0c55971c5e63",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003elabel=%{label}\u003cbr\u003evalue=%{value}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["No","Yes"],"labels":["No","Yes"],"legendgroup":"","name":"","showlegend":true,"values":[239424,14256],"type":"pie","text":["No","Yes"],"textinfo":"percent+label"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"marker":{"line":{"color":"#283442"}},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#f2f5fa"},"error_y":{"color":"#f2f5fa"},"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"marker":{"line":{"color":"#283442"}},"type":"scattergl"}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"baxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#506784"},"line":{"color":"rgb(17,17,17)"}},"header":{"fill":{"color":"#2a3f5f"},"line":{"color":"rgb(17,17,17)"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#f2f5fa"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"rgb(17,17,17)","plot_bgcolor":"rgb(17,17,17)","polar":{"bgcolor":"rgb(17,17,17)","angularaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"radialaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"ternary":{"bgcolor":"rgb(17,17,17)","aaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"baxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"caxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2},"yaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2},"zaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3","gridwidth":2}},"shapedefaults":{"line":{"color":"#f2f5fa"}},"annotationdefaults":{"arrowcolor":"#f2f5fa","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"rgb(17,17,17)","landcolor":"rgb(17,17,17)","subunitcolor":"#506784","showland":true,"showlakes":true,"lakecolor":"rgb(17,17,17)"},"title":{"x":0.05},"updatemenudefaults":{"bgcolor":"#506784","borderwidth":0},"sliderdefaults":{"bgcolor":"#C8D4E3","borderwidth":1,"bordercolor":"rgb(17,17,17)","tickwidth":0},"mapbox":{"style":"dark"}}},"legend":{"tracegroupgap":0},"title":{"text":"Distribution of HvyAlcoholConsump"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('2adc99df-2178-4b70-bfa0-0c55971c5e63');
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
    



<div style="display: inline-block;">
    <div class="jupyter-widgets widget-label" style="text-align: center;">
        Figure
    </div>
    <img src='data:image/png;base64,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' width=640.0/>
</div>



### Validations Predictions


```python
val_preds = predict_and_plot(X_val, val_targets, 'Validatiaon')
```

    Accuracy: 90.87%
    



<div style="display: inline-block;">
    <div class="jupyter-widgets widget-label" style="text-align: center;">
        Figure
    </div>
    <img src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgCAYAAAA10dzkAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/OQEPoAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA33UlEQVR4nO3deXxjZ33v8e/RvtqyJEsejz3DZCaBJDchAW4IpQuXpXCbm0AplO2yNJQLlAIFCgVKoBQocCFdWEoXoLzgFig0l6Zs5bIUWiBAKVtYJyQTz3iTbFmyLGuXzv1j0GE08mTOzNiWfM7nzcsvrKMj6ZHG4K9/v+d5jiHJFAAAAFzDM+wBAAAAYHcRAAEAAFyGAAgAAOAyBEAAAACXIQACAAC4DAEQAADAZQiAAAAALkMABAAAcBkCIAAAgMsQAAEAAFyGAAgAAOAyBEAAAACXIQACAAC4DAEQAADAZQiAAAAALkMABAAAcBkCIAAAgMsQAAEAAFyGAAgAAOAyBEAAAACXIQACAAC4DAEQAADAZQiAAAAALkMABAAAcBkCIAAAgMsQAAEAAFyGAAgAAOAyBEAAAACXIQACAAC4DAEQAADAZQiAAAAALkMABAAAcBkCIAAAgMsQAAEAAFyGAAgAAOAyBEAAAACX8Q17AMC52r9/v0KhkO68884znnPw4EF1Oh3Nz8+f9fkOHTqkarWqXC4nn8+niy66SMvLyyqXy7YeY1coFFIqldLCwoIk2X6tYQiHw0okEgqHw/J4PGq326pUKioWi+p0Ojvymul0WuPj4zIMQ7lcThsbGxf8nGNjY5qamtJdd92ldru9DaM8+2tJ0rFjx9RqtQbOiUQimpmZkSQdPXrU9nN7PB5lMhmtr6+rVqud8bxR/pkCMFqoAGLPKZfL8nq9ikajW94fDAYVDAa1vr5+zs/d6XR0/PhxbW5uXugwB4yPjysQCOzKa12IdDpthZR8Pq+FhQWVSiWNjY1pdnZWPt/2/90YCASUTCa1sbGhhYUFVavVbXnezc1NHT9+fMdC61ZM01Q8Ht/yvjMdP5tgMKixsbGznjeqP1MARg8BEHtOpVJRp9M54y/EsbExdTodVSqVc35u0zRVr9d3JTDs5mvZFY/HlUwmtbKyoqWlJVUqFdVqNZVKJZ04cUI+n0+Tk5Pb/rper1eStLGxoVqttm2fSafTUb1el2ma2/J8dtRqtS2DnmEYisViqtfrO/bao/gzBWA00QLGnmOapjY2NjQ2NiaPx6Nut9t3fzwe18bGhkzTlMfjUTqdVjQalc/nU7fbVa1WUz6f37IluFULLRAIaHJyUuFwWJ1OR6urqwOPO9vrZLNZjY+PS5IuueQSLS8vq1qtDrxWOBxWMplUKBSyWq/lclmFQqHvtVKplPVarVZLxWKxr+V36NAhlctlGYZhfU698WzVmuyZmJhQo9FQqVQauK/VamllZUUez8//bjQMQxMTExobG5PP51O73db6+rqKxaJ1zszMjFqtlprNphKJhLxerxqNhlZWVlSv15VKpZRKpSRJs7OzarVaOnbsmC655BIVCoW+9947t9c+9Xq9mpycVCQSkcfjUbPZVLFYtNrHW7WAI5GIksmkgsGgpJNVwtXVVev+sbExZbNZnThxQpOTkwoGg+p0OiqVSn3v60w2NjaUzWbl9/v7PutIJGK9XigU6nvM2NiYEomEVSFuNptaW1tTpVJROBzW7Oys9flUq1XNz89rZmZG7XZbhmEoGo2qVqspl8v1/UwdOHBAfr9fx44ds/53ks1mFY/HNTc3d48/CwCcjQog9qT19XV5PB7FYrG+471Q1Gv/7t+/X5FIRKurq5qfn1ehUFA4HFY2m7X1Oj6fT7Ozs/J6vVpaWtLq6qrS6fRAG/Rsr9P7Zd5ut8/YogsEApqZmVGn09HS0pIWFhZUq9WUSqWsipJhGJqdnVU8HlexWNTi4qJqtZqmpqaUTCb7nq8XKJaXl5XL5RQKhaw5alvxer0KhUL32D48PdxNT08rmUxqfX1di4uL2tjYUDqdViaT6XtcLBZTLBZTPp/X8vKyvF6v9u3bZz1nby5lLpfT4uLiGV//dFNTUwoEAsrlclpYWFCj0dC+ffsUDoe3PD8ej1vBaXl5Wfl83gpYvSpkz759+6yWdK1Ws4Lm2VSrVXU6nYEqYDwe37IqPT4+rmw2q0qlooWFBS0vL8s0Te3bt08+n0+NRqPv88nn833P2e12tbCwsGU4XV5elmEYVtU2Go1qfHxcKysrhD/A5agAYk9qNBqq1+uKx+N9la+xsTE1Gg01Gg15vV6Zpqnl5WWr7Var1eT3+61q3NkkEgkZhqH5+XmrgtJqtXTgwAHrHDuv02q11Ol0rBadpIEQGQwGVa1Wtby8bB2rVquKRqMKh8NW1TMYDOr48ePW81SrVRmGoWQyqVKpZI2z2+32hSm/3690Or1l1fTU8dgNBpFIRNFoVEtLS1bFrVqtyjRNpdNplUolNZtNSSeD68LCgvW6hmFo3759CgaDajQa1nnNZlONRsPW60snK6Zra2tWaO21j8/U8p2cnNTm5mbfZ1yv13Xw4EFNTExY1V3DMFQoFKyfrXq9rlgspmg0etb5iaZpqlKpKB6Pa21tzXq+WCymxcXFgXDq9/tVLBatc6WT/wYHDx60/t1P/Xx63/deK5/PW+/39J+pZrOpQqFgve9MJqNKpXJe82MBOAsBEHtWuVzW5OSkvF6vOp2OPB6PotGo9Uv81FXAPp9PgUBAgUDAWtlqRzgcVq1W6wtM9Xq9LyRtx+tIJ1uHGxsbMgxDfr9fgUBAwWBQhmHIMAxJJ0NXq9UamEdWLpc1Pj6ucDhshaHTz+m1OM8UAM9VJBKx2vGnjyWdTiscDvcFl1Nf89SxXIhehTQYDGpzc9Nq524lEAjI5/MN3N/7PE+v7p36+Zmmaf2M2bGxsaHx8XGrDRyNRtXtdlWtVgcCYG88Ho9HgUBAfr/fGkvv3/1Mms3mWec3FotFxWIx7du3T51O55xWrgNwLgIg9qxe0IjH4yqVSlbL7dSKYDweVzqdlt/vtxYEnEv48Xq9W1bETp8/eKGvI538ZZ/JZBSPx2UYhhVMTv0F35sXeKbxnBpQzvX1W62WTNOU3+8/4zkej0emaco0TSt4n6537NSW6nYEzq0sLS0pmUwqHo8rHo/LNE1re57TP6feZ3Omz+/0eXkXMuZqtap2u21VAXvzUrfi9/uVyWSskNhqtWxXQe2OcWNjQ+FwmAUiACwEQOxZ3W5XlUpFY2Nj1jYllUrF+qXYm/PWm7zf+8WfTqdtzeWSToaZrbY9OTXcbMfrSCfbk7FYTEtLS1YrVZIuuuiivvd86lYyPb0xXsgv9263q0aj0VdFPV0qldL4+LiOHTumTqczMG9O+vlnsxNB4/SKWLfb1erqqlZXV+X3+xWLxZRKpZTJZAbmEvZ+Lrb69/T5fNs+3kqlolgspmKxqGg0esY9Kffv3y/TNDU3N2cFv0AgYGvbFzu8Xq9SqZTVxo7FYue1Qh6As7AIBHtauVxWKBRSOBxWOBzuq/6Fw2FrLtepVZ9zCWXValWhUKgvNPTadNv5Or3nqdVq2tzctMJfMBiUz+ezgk+1WpXf799yFWm3273gLUbW1tYUDAaVSCQG7uuFkt4ih1qtJsMwBhY79ILLPW1YbMdW4fvU9qnP59OhQ4eshUC91dC9z+h0zWbTqsqdyu/3W5/9dtrY2FAoFFIymbSqwqfzer0KBAJaX1/vq/qdaY/L85HNZmWapubn51WpVJTJZLYM7gDchQog9rRqtapWq6VsNqtms9k3Qb/3C7d3BQWv16tEImFt/2EYhq35U+Pj49q/f7+1HUk6ne57nN3X6VXMIpHIli2+3qKW8fFxNZtNBYNBJZNJmaZpBcByuaxEIqHp6WkVCgW1Wi3FYjGNj4+rUChccKu1t0Agk8koFApZ2+mEQiFNTEz0zSHb3NxUtVpVNpu1Vqv2trFZX1/vW6xwPjY3NxWPx1Wv19VsNq05dT3tdlvtdluZTEYej0etVkuhUEiRSOSM27Wsrq5qampKU1NT1obiqVRKnU7H1hYv56JWq6ndbiuZTJ7xuTudjlqtlhKJhNrttjqdjqLRqCYmJiT9vG3dq05Go1F1Oh3bn208Hreqyt1uV/l8XgcPHlQmk9HS0tI2vEsAexUBEHteuVxWKpUaaFv29kWbmJhQLBZTp9NRtVpVsVi0tm052xUTut2utR/c1NSUut2uNafrXF+nXC4rGo1q//79Wl1dHZgTtrKyIsMwlE6nrTmAa2trCgQCVpXLNE1rPKlUytr7bjsv/ZXL5VStVq3tSXrhan19XWtra30hc2FhQalUytrfr91ua3V1dVvCVO/zmJyctBab9AJcz+LiotLptLW6ud1ua21trW9F7anK5bK63a6SyaSmp6ethRmrq6s70rLe2NjQxMTEPV7WbmFhQZlMRlNTUzJNU41GQwsLC9bek73V1L3wH41GNTc3d9bX9nq91qrf3uu3220VCoWB4wDcx5C0e1vkAwAAYOiYAwgAAOAyBEAAAACXIQACAAC4DAEQAADAZQiAAAAALkMABAAAcBkCIAAAgMsQAAEAAFyGK4FcoObKncMeAjBywtO/NOwhACOp01rc8dfYrt9LgcnD2/I8GE0EQAAAnKS7/Zc1hPMQAAEAcBKze/Zz4HrMAQQAAHAZKoAAADhJlwogzo4ACACAg5i0gGEDLWAAAACXoQIIAICT0AKGDQRAAACchBYwbKAFDAAA4DJUAAEAcBI2goYNBEAAAJyEFjBsoAUMAADgMlQAAQBwElYBwwYCIAAADsJG0LCDAAgAgJNQAYQNzAEEAABwGSqAAAA4CS1g2EAABADASdgHEDbQAgYAAHAZKoAAADgJLWDYQAAEAMBJWAUMG2gBAwAAuAwVQAAAnIQWMGwgAAIA4CS0gGEDLWAAAACXoQIIAICDmCb7AOLsCIAAADgJcwBhAwEQAAAnYQ4gbGAOIAAAgMtQAQQAwEloAcMGAiAAAE7SZREIzo4WMAAAgMtQAQQAwEloAcMGAiAAAE7CKmDYQAsYAADAZagAAgDgJLSAYQMBEAAAJ6EFDBtoAQMAALgMFUAAAJyECiBsIAACAOAgpslG0Dg7AiAAAE5CBRA2MAcQAADAZagAAgDgJGwDAxsIgAAAOAktYNhACxgAAMBlqAACAOAktIBhAwEQAAAnoQUMG2gBAwAAuAwVQAAAnIQWMGwgAAIA4CS0gGEDLWAAAACXoQIIAICTUAGEDQRAAACchDmAsIEACACAk1ABhA3MAQQAAHAZKoAAADgJLWDYQAAEAMBJaAHDBlrAAAAALkMFEAAAJ6EFDBsIgAAAOAktYNhACxgAAMBlqAACAOAkVABhAwEQAAAnMc1hjwB7AC1gAAAAl6ECCACAkwyhBWwYhjKZjGKxmEzTVLFYVLFY3PLcWCymVColv9+vRqOhfD6vRqOxyyMGARAAACcZQgBMp9MKhUKan5+X3+9XNptVq9VSpVLpOy8QCGhqakq5XE71el0TExPav3+/jh07JpPW9a4iAAIA4CS7vA+gYRgaHx/XwsKCGo2GGo2GAoGAEonEQACMRCJqNpva2NiQJK2srCiRSCgQCFAF3GXMAQQAAOctGAzKMAzVajXrWK1WUygUGji30+koEAhY942Pj6vT6ajVau3aeHESFUAAAJxkl1vAPp9PnU6n71in05HH45HX6+27r1KpKBaL6cCBA1bLd2FhQV22rtl1BEAAAJxkm+bSGYYhwzBOe2pzYK6eYRgDx3q3T3+8x+ORz+ez5gAmEglls1kdP358IERiZxEAAQDAgGQyqVQq1XesUCioUCj0HTNNcyDo9W6fXtmbnJxUo9HQ+vq6JCmXy+le97qXxsbGzrhqGDuDAAgAgJNsUzt1bW1tIJRttVK33W7L6/X2HfN6vep2uwMBMBgMqlQq9R1rNBry+/3bMmbYRwAEAMBJtikAbtXu3Uqj0ZBpmgqFQqrX65KkcDhsfX+qdrutQCDQdywQCKhcLm/LmGEfq4ABAMB5M01T5XJZ2WxWwWBQ0WhUExMTVqXP6/VaLeH19XWNj48rHo/L7/crnU7L5/MRAIeACiAAAE6yy/sASif388tkMpqdnVW321WhULD2ADx8+LCWl5dVLpdVqVSUz+eVTCbl9/tVr9c1Pz/PApAhIAACAOAgZnf3r6hhmqZyuZxyudzAfUePHu27XS6XqfiNAFrAAAAALkMFEAAAJ2FTZdhAAAQAwEmGMAcQew8BEAAAJxnCHEDsPcwBBAAAcBkqgAAAOAlzAGEDARAAACchAMIGWsAAAAAuQwDEyGo0mrrpjX+mBz3ycXrIDU/W+z50yxnP/crX/1OPffrv6L8+/Nf12y98hY7NzVv3maapv/vgP+qRj3uGHvTIx+lVb/hTVau13XgLwLYIBoP6m79+q1bzP9SJuW/pRb/37DOee9VVl+urX/64yqWf6ravflL3u/qKLc97xctfoPe8+8+2vO/Tn/ygnvbU39yWsWMITHN7vuBoBECMrJvf+W794Md36D1ve5Ne9ZLn6V3v/Xv9v3/994HzfnrXnJ730tfoob94rT7ynrfr0kuO6JkveLkV8j5666f1l+/9e73w2c/QB951s3Irq3rZH715t98OcN7e/KZX6f73v68e8au/qd99wSt106tepMc+9rqB8yKRsD5+6wf05S9/Q9dc+yjddts39c+3vl+RSLjvvCc84dF6zatfMvB4wzD053/2Oj3iEb+yY+8Fu6Db3Z4vOBoBECOpWqvrlo9/Ri9/4XN02b2P6OG/8mDd+JTH64O3fHzg3H/42Cd11RWX6nef9TQdOjijF//OjYrFIvrE//tXSdIH//Gf9fQnPla/9oiH6MhFB/Unr/p9femr3+irEgKjKhIJ65k3PkkvfvGr9e3vfF+33voveuvN79LznvuMgXN/8/E3qFar62Uvf51+/OOf6sUveY02Njb1uN+4XpLk9Xr1jre/Ue/+m5t1511zfY+dnp7SZz/zEV3/P35VxWJpF94ZgGEiAGIk/eSnd6ndaevqKy61jl195eW6/Qc/Ufe0v0znF5d0xWX3sW4bhqGLLzqk737/Rz+7f1lXXnZv6/7JdFITiXHrfmCU3ffKy+X3+/XV275pHfvKV76ha665WoZh9J37wAfeT1/56n/0Hfvqbf+ha6+9vyQpFovqyisu1S/84vX62tf+s++8+119hU7ML+qaax+l9fWNHXo32BVdc3u+4GiuWwXs8XhkGIZM0xwIEhgdq6trSoyPy+/3W8dSyYQazaZK62UlJxKnHJ9QfmW17/HL+RWNj8Wtx+VXC9Z91Vpd5fKGiuvrO/smgG0wtS+j1dU1tVot61guv6JwOKxUakKrq2vW8X37svrhD3/S9/h8fkWX/+wPpPX1sn75IY/Z8nU+8cnP6hOf/Oz2vwHsPq4EAhtcUQGMxWKamZnRkSNHdPjwYV100UU6fPiwjhw5opmZGUWj0WEPEaepNRoKnBL+JFm3m6f8IpSkRz3sl/WZf/2yvviVr6vd7ujWT31WP/jRUesX5qMe9st69wc+ojvvPq5Go6m3vP1vJEmtVnsX3glwYSKRsBqNZt+x3u1gMNh/bnjrc4PBwM4OEsCe4/gKYCKRUCqVUrFYVKFQULvdlmmaMgxDPp9P4XBYU1NTKhQKKpVKwx4ufiYYCAwEvd7tcCjUd/wXr32Annvjk/WiP3y9Op2urrnflbr+vz9MlcqmJOnZz3iy5heX9Zj/+Rz5fF49/tG/pntffJFi0cjuvBngAtTrjYEA17t9+mr2M51brbHq3VVo38IGxwfAZDKp5eVlbW5uDtzXarVUq9XUaDSUyWQIgCMkM5lSaX1d7XZHPp9XklQoFBUKBhWPDVZsn/30J+m3nvQb2tisKjWR0Etu+hNN78tKkiLhkG5+3Su1UdmUYUixaFS/fN0TrfuBUba4sKx0Oimv16tOpyNJmspmVK3WVCr1T2NYWFxSNpvpO5bNZrS0lN+18WL4TKY3wQbHt4ANw+ibO7OVdrstj8fxH8Wecp+LL5LP69P3fvDzhRrf+t4P9F8uvXjg3+pTn/2i3vTnf6VAIKDUREL1RkPf+NZ3dc397itJuvmd79Gtn/qs4rGoYtGobv/RT7Sxuamrr7hsV98TcD6+893vq9Vq6doH3s869uAHX6NvfvM7Mk/bq+3rX/+WHvSgB/Qd+4UHPUBf/3r/gg84HItAYIPjU0+lUtHU1JTC4fCW94dCIU1NTalSqezyyHBPwqGQbvjvD9cfv+Uduv1HP9Hn/+2ret+HbtFTHv8YSdJqYU31RkOSdHB2vz5y66f02S9+RXMnFvSyP3qzpjKT+qVrT/4izKSTetfffVC3/+gn+sGP79ArXvsWPeEx11mLRIBRVqvV9f4P/KPe+c436QH3v69uuOGRevGLnq23veM9kqRsdlKhn02LuOX/flKJ8TH96c2v1aWXXqw/vfm1ikYj+ug/Dm6fBMDdHN8CzufzSqfT2r9/vwzDUKfTseYAer1emaapcrmslZWVYQ8Vp3nZC56l173lHbrx+S9XPBrV8575P/WIhzxYkvSQG56i17/yxXrMdY/Q5fe5WDf9/u/qre/4W5XWy3rgA67SX77lj61K4ZMfd4MWlnJ67kteLY9h6PpHPUwveu6Nw3xrwDn5/Zf+kd75jjfpc5/9qNbXy3rtH9+sf/qnT0uSFk58Rzc+80V6/wc+oo2Nih79mKfrne98k57120/R7bf/SNc/+qlc+cZtWAUMGwxJrqjzGoahYDAon89nbQPTbrfVaDQG2ijnorly5zaOEnCG8PQvDXsIwEjqtBZ3/DUqr33ytjxP7DUf3JbnwWhyfAWwxzRN1ev1YQ8DAABg6FwTAAEAcAVWAcMGAiAAAE7CCl7Y4PhVwAAAAOhHBRAAACdhFTBsIAACAOAktIBhAy1gAAAAl6ECCACAg3AtYNhBAAQAwEloAcMGAiAAAE5CAIQNzAEEAABwGSqAAAA4CdvAwAYCIAAATkILGDbQAgYAAHAZKoAAADiISQUQNhAAAQBwEgIgbKAFDAAA4DJUAAEAcBKuBAIbCIAAADgJLWDYQAsYAADAZagAAgDgJFQAYQMBEAAABzFNAiDOjgAIAICTUAGEDcwBBAAAcBkqgAAAOAkVQNhAAAQAwEG4FBzsoAUMAADgMlQAAQBwEiqAsIEACACAk3AlONhACxgAAMBlqAACAOAgLAKBHQRAAACchAAIG2gBAwAAuAwVQAAAnIRFILCBAAgAgIMwBxB2EAABAHASKoCwgTmAAAAALkMFEAAAB6EFDDsIgAAAOAktYNhACxgAAMBlqAACAOAgJhVA2EAABADASQiAsIEWMAAAgMtQAQQAwEFoAcMOAiAAAE4yhABoGIYymYxisZhM01SxWFSxWNzy3EAgoGw2q2AwqFarpXw+r1qttssjBi1gAABwQdLptEKhkObn55XP55VMJhWLxQbO83g8mpmZUaPR0NzcnCqViqanp+X1eocwanejAggAgIPsdgvYMAyNj49rYWFBjUZDjUZDgUBAiURClUql79yxsTF1u13l83lJUqFQUDQaVSgU0ubm5u4O3OUIgAAAOMhuB8BgMCjDMPrauLVaTclkcuDccDg8EAqPHz++42PEIAIgAAAOstsB0OfzqdPp9B3rdDryeDzyer199/n9ftXrdWu+YKvV0srKiur1+u4OGswBBAAAgwzDkMfj6fsyDGPL80yz//rDvdunn+/xeJRMJtXpdLSwsKBaraaZmRn5fNSjdhufOAAATmIOhrTzkUwmlUql+o4VCgUVCoX+lzPNgaDXu93tDpYjG42G9RyNRkORSERjY2NaW1vblnHDHgIgAAAOsl0t4LW1tYGtXE6v9ElSu90eWMXr9XrV7XYHAmC73Vaz2ew71mq1qAAOAS1gAAAwwDRNK8T1vrYKgI1GQ6ZpKhQKWcfC4fCW8/rq9bqCwWDfsUAgoFartf1vAPeIAAgAgIOYXWNbvmy/nmmqXC5bmztHo1FNTEyoVCpJOlkN7LWES6WSgsGgUqmU/H6/9d8bGxs78VHgHlBzBQDAQYZxKbiVlRVlMhnNzs6q2+2qUChY270cPnxYy8vLKpfLarfbmp+fVyaT0cTEhJrNphYWFtRut3d/0C5HAAQAABfENE3lcjnlcrmB+44ePdp3u16vs/ffCCAAAgDgIOY2rQKGsxEAAQBwkGG0gLH3sAgEAADAZagAAgDgIOeyghfuRQAEAMBBttiqDxhAAAQAwEGoAMIO5gACAAC4DBVAAAAchAog7CAAAgDgIMwBhB20gAEAAFyGCiAAAA5CCxh2EAABAHAQLgUHO2gBAwAAuAwVQAAAHIRrAcOOka4AJpNJGcZgKdvj8SidTg9hRAAAjLauaWzLF5xt5CqAfr9fPt/JYaVSKTUaDXW7/X/OBAIBJRIJra6uDmOIAAAAe9rIBUCfz6eZmRnr9vT09MA5pmmqWCzu5rAAANgTWAQCO0YuANZqNd1xxx2SpEOHDmlubm6gAggAALbGNjCwY+QC4KmOHTsmSTIMQ4FAQM1mU4ZhEAgBADgDrgQCO0Y6ABqGoUwmo7GxMUnS3XffrXQ6LY/Ho6WlJYIgAADAeRjpVcDpdFqBQEBzc3Myf/YnTaFQkNfrVSaTGfLoAAAYPWbX2JYvONtIVwBjsZgWFxfVbDatY81mU7lcrm+hCAAAOIktXGDHSFcAPR6PVfkDAADA9hjpALi5ual0Om1tBm2apnw+nzKZjDY3N4c8OgAARo9pGtvyBWcb6RZwPp9XNpvVkSNHJEkHDx6Ux+NRtVpVPp8f8ugAABg9NM5gx0gHwG63q6WlJfn9fgUCAUkn5wC2Wq0hjwwAAGDvGukAGA6Hre97W774fD75fD6Zpql2u612uz2s4QEAMHJYBAI7RjoAZrNZ+f1+ST8PgB5P/7TFer2uxcVFdTqdXR8fAACjhvl7sGOkA2C5XFY0GtXy8rLV9vX7/cpms6pUKiqXy8pms8pkMlpaWhryaAEAAPaGkV4FnEgklMvl+ub8tVot5fN5JZNJdbtdFQoFRSKRIY4SAIDRYZrb8wVnG+kKoCR5vd4tj/W2hgEAAD/HHEDYMdIBsFwua2pqSoVCQfV6XZIUCoWUSqVULpfl8XiUTqdVrVaHNsb2dz83tNcGRtXlyYPDHgIwkr6XW9zx12AOIOwY6QC4urqqbrerVColn+/kUNvttkqlkorFoiKRiEzTZE9AAACAczDSATAej6tUKmltbc1a/dtbDSxJ1Wp1qNU/AABGDS1g2DHSi0AymYw1B7Db7faFPwAAMMjcpi8420gHwGq1qrGxMRZ8AAAAbKORbgH7fD7FYjElk0l1Op2BCuDdd989nIEBADCiaAHDjpEOgOvr61pfXx/2MAAA2DNYBQw7RjoAlsvlYQ8BAADAcUY6AHq9XiWTSQUCgb55gIZhKBAI6M477xzi6AAAGD0sl4QdI70IJJvNKhqNql6vKxwOq16vq9PpKBQKqVAoDHt4AACMHFPGtnzB2Ua6AhiJRDQ/P696va5oNKpKpaJ6va6JiQlFo1GVSqVhDxEAAGDPGekKoHTyyh+S1Gg0FAqFJEkbGxvW9wAA4Oe65vZ8wdlGLgCGw2Hr+3q9rrGxMUknA2AkEpEk+f3+oYwNAIBR15WxLV9wtpELgDMzM9bVP1ZXVzUxMaFEIqFyuaxQKKSDBw9qenpaGxsbQx4pAACjhzmAsGOk5wDW63UdO3ZMhmGo2+1qbm5OsVhM3W6XAAgAAHCeRjoASuq7+ken02FjaAAA7gHbwMCOkQyABw4ckGmefQYql4IDAKAf7VvYMZIBsFgsDlz3FwAAANtjJAPgxsaGOp3OsIcBAMCeQ/kEdoxkAAQAAOeHAAg7Rm4bmHK5TPsXAABgB41cBTCXyw17CAAA7FksAoEdIxcAAQDA+euS/2DDyLWAAQAAsLOoAAIA4CBcxxd2EAABAHCQs19GASAAAgDgKOyjATuYAwgAAOAyVAABAHCQrsEcQJwdARAAAAdhDiDsoAUMAADgMlQAAQBwEBaBwA4CIAAADsKVQGAHARAAAFwQwzCUyWQUi8VkmqaKxaKKxeI9Psbn8+le97qXFhYWVKvVdmmk6CEAAgDgIMO4Ekg6nVYoFNL8/Lz8fr+y2axarZYqlcoZH5PNZuXxsBRhWAiAAAA4yG6vAjYMQ+Pj41pYWFCj0VCj0VAgEFAikThjAIzH44S/IePTBwDAQbrG9nzZFQwGZRhGXxu3VqspFApteb7H49Hk5KRyudyFvlVcACqAAABggGEYMk7bVNo0TZlmf43R5/Op0+n0Het0OvJ4PPJ6vQP3TU5Oan19Xc1mc2cGDlsIgAAAOMh2bQOTTCaVSqX6jhUKBRUKhb5jhmEMhMLe7dMDZCQSUTgc1tzc3DaNEueLAAgAgINs1xzAtbW1gZW8pwe93rHTg17vdrfb7TuWyWSUz+e3fB7sLgIgAAAYsFW7dyvtdlter7fvmNfrVbfb7QuAoVBIgUBA09PTfefu379f5XJZ+Xx+ewYOWwiAAAA4yG5vBN1oNGSapkKhkOr1uiQpHA5b3/fU63UdO3as79ihQ4eUy+VUrVZ3bbw4iQAIAICD7Pal4EzTVLlcVjab1fLysnw+nyYmJqxVvr1qoGmaarVaA49vt9sDC0Ww89gGBgAAXJCVlRXV63XNzs4qm82qUChYewAePnxY8Xh8yCPE6agAAgDgILtdAZROVgFzudyWe/sdPXr0jI+7p/uwswiAAAA4iLn7V4LDHkQLGAAAwGWoAAIA4CDDaAFj7yEAAgDgIARA2EEABADAQbjGBuxgDiAAAIDLUAEEAMBBdvtKINibCIAAADgIcwBhBy1gAAAAl6ECCACAg1ABhB0EQAAAHIRVwLCDFjAAAIDLUAEEAMBBWAUMOwiAAAA4CHMAYQctYAAAAJehAggAgIOwCAR2EAABAHCQLhEQNhAAAQBwEOYAwg7mAAIAALgMFUAAAByEBjDsIAACAOAgtIBhBy1gAAAAl6ECCACAg3AlENhBAAQAwEHYBgZ20AIGAABwGSqAAAA4CPU/2EEABADAQVgFDDtoAQMAALgMFUAAAByERSCwgwAIAICDEP9gBwEQAAAHYQ4g7GAOIAAAgMtQAQQAwEGYAwg7CIAAADgI8Q920AIGAABwGSqAAAA4CItAYAcBEAAABzFpAsMGWsAAAAAuQwUQAAAHoQUMOwiAAAA4CNvAwA5awAAAAC5DBRAjq9Fq640f/rw+9507FPL79LSHP0BPe/gDtjz3C9+5Q2+/9ctaLm3o3jOT+oPHP1SXHshqobCu625695aPec+LnqD7Xzyzk28B2BGBYECvfONL9LDrHqJGvaH3v+tDev9ffegeH3P1NVfq9W+/Sdc98PHWMY/Ho+e/4tm64Qm/pnAkpC9/4Wt60yv/VGurxZ1+C9hB1P9gBwEQI+vP/u+X9MPjOf3tCx+vpbWybnr/v2hfckyPuN8lfef9dHFVr/i7T+lVT3q4rjq8X//nC/+p5//lx/TxP36mpibi+twbn9N3/ltv+aJOrJR05UX7dvPtANvmxa9+ni677330rMc9X9MzU3rd227S4vyyPveJf93y/CP3uUhvffcb1Gg0+47f+Pyn6pGPebhe+r9uUmmtpD94/Yv0J+94jZ7zxN/bhXeBnUILGHbQAsZIqjVa+thXv6+XPv6/6dIDWT30qov1jEf8V334S98eOPe2H83p8L6Urr/2cs1OJvSCR/+SVsubumupIK/Ho/R41PqaXy3p89+5Q69/+qPk93qH8M6ACxOOhPTrT75B//umP9ePbz+qL3z63/S+d/69nnjjb2x5/uOe+mi9/xN/rcLK2sB9Xp9Xb331X+hbX/uO7jp6tz747o/qqmuu3Om3gB3W3aYvOBsBECPpJ/N5tTsdXXXRtHXsqsP79f27l9Xt9v91m4iGdOdSQd++c0Hdrqlbb/u+YqGAZicTA8/7tlv/XY998BU6NJXa6bcA7IhLLjsin9+r7/zH7daxb3/ju7ri6stlGMbA+Q9+6IN00wter//zN/8wcN9f3/xefeHT/yZJSqYn9NinXK9v3vatnRs8gJFBCxgjabW8qUQsLL/v51W6VDyiRqut0mZNyXjEOv7I+99bX7r9Tv3WzR+W12PIMAy9/Xd+XWORUN9zfvvOBX3vriW98cbrdu19ANstnU2rtLaudqttHSusrCkUDiqRHFexUOo7/0W/9XJJ0g1P+LUzPudzX/pMPeclz9R6sayn3/DsHRk3dg8bQcMOKoAYSfVmWwFff4s24D95u9Xu9B0vbda1Wt7Uy5/wUH3gZU/R9Q+8TK/5wGe0tlHtO++WL39PD73qiLKJ+M4OHthB4XBIzWar71izcfK2P+A/r+f8xEf/RU965I362r//h/7qw3+uaCxy9gdhZNEChh0EQIykgN+r5mlBr9k6eTsU6C9c/8U//Zsunp7UE3/lal12IKubnvyrCgf8uvW271vntDtdffF7d+q6ay7b+cEDO6jRaChwWtALBE/ertfq5/WcJ+5e0A+/+2O96vmvUzAU1MOue8iFDhPAiHNFCzgcDts+t1ar7eBIYFdmPKZSpaZ2pyuf9+TfKYXypkJ+n+Lh/tbuj47n9KSH3M+67fEYumRmUouFsnXse8cW1e50dO2lB3fnDQA7JL+0okRyXF6vV53OyT+K0pmUatW6NtYr5/Rcv/yIX9CPbz+q/PKqJKnZaGrh+KISyfFtHzd2Dy1g2OGKAJjJZBQIBGyde8cdd+zwaGDHvWcz8nm9uv3Yoq4+cnKvvm/fuaDLD07J4+mf6D45HtNdy4W+Y3O5oi6/Zsq6ffuxJV12IKug3xU/8nCwn/zgDrVbHV15/8v17W98T9LJPf5+8N0fyTTP7Rf/i1/9fP3zRz6l9779A5KkSDSiAxfN6tgdc9s+buwe2rewwxW/DY8fP66pqSn5/X6dOHHinP9PErsvHPDr+gdeptd/6HN67VMfpXxpQ+//3Df12qc+SpK0ur6pWDigUMCvxz74Cr36A5/R5QendOWhffrYV2/X4lpZN1x7ufV8P10q6CJW/sIB6rWGPv6RT+lV//tlevXvvUGZqUk97blP1mt+7w2SpNRkUpWNihr15lmeSfqH992i5/7+b+voD3+qpfllPf8Vz9GJu+f15c/fttNvA8CQuSIAmqap5eVlzc7OKpVKaXV1ddhDgg0vedxD9IYPfU7P+ouPKBYK6jn/4xf0sKsvliQ9/BV/pdc+9ZF69IP+ix75gPuo2mjpPf/ydeVKG7r3TEZ/+8LH960UXitv6t4zmWG9FWBbvfWP3qY/fPNL9e5b3q5KeVPvesu79flPfUmS9IXbP6GbXvh6/fM/fOqsz/Ph996icDisP3zzSzWRTOi2L31DL3zaH/BH8h7X5d8PNhhy0VVjAoGAwuGw1tfXt+05q5/76217LsAprn3K+4Y9BGAkfS+389XVpxz49W15nr8//rFteR6MJldUAHuazaaazbO3RQAAAJzMVQEQAACn41rAsIMACACAg7ANDOwgAAIA4CBsAwM7uBIIAACAy1ABBADAQZgDCDsIgAAAOAhzAGEHLWAAAACXoQIIAICDsAgEdhAAAQBwEC7lBzsIgAAA4IIYhqFMJqNYLCbTNFUsFlUsFrc8NxqNKpVKKRAIqNVqaXV1VZubm7s8YhAAAQBwkGGsAk6n0wqFQpqfn5ff71c2m1Wr1VKlUuk7LxAIaN++fVboi0Qimp6e1tzcHJdq3WUsAgEAwEG62/Rll2EYGh8fVz6fV6PRUKVSUbFYVCKRGDh3bGxMtVpNpVJJrVZL6+vrqlarisfj5/t2cZ6oAAIAgPMWDAZlGIZqtZp1rFarKZlMDpxbLpe3fA6v17tj48PWqAACAOAg5jb9xy6fz6dOp9N3rNPpyOPxDAS7ZrPZ1+oNBAKKRCKqVqsX9qZxzqgAAgDgINs1B9AwDBmG0XfMNM2BVcaGYQwc690+/fGn8ng8mp6eVq1WG5griJ1HAAQAwEG2axuYZDKpVCrVd6xQKKhQKAy83ulBr3e72916NqHX69XMzIwkaWlpaVvGi3NDAAQAAAPW1tYGtnLZKly22+2BVq/X61W3290yAPp8Piv8nThxYqB9jN1BAAQAwEG260ogW7V7t9JoNGSapkKhkOr1uiQpHA5b35/KMAzt379fpmlqfn6e8DdELAIBAMBBdnsRiGmaKpfLymazCgaDikajmpiYUKlUknSyGthrCSeTSfn9fuVyOes+r9crj4c4stuoAAIAgAuysrKiTCaj2dlZdbtdFQoFa2HH4cOHtby8rHK5rHg8Lo/HowMHDvQ9fn193QqF2B0EQAAAHGQYVwIxTVO5XG7LEHf06FHr+7vvvnsXR4V7QgAEAMBBtmsVMJyNpjsAAIDLUAEEAMBBhtECxt5DAAQAwEHOZQUv3IsWMAAAgMtQAQQAwEG6LAKBDQRAAAAchPgHOwiAAAA4CItAYAdzAAEAAFyGCiAAAA5CBRB2EAABAHAQrgQCO2gBAwAAuAwVQAAAHIQWMOwgAAIA4CBcCQR20AIGAABwGSqAAAA4CItAYAcBEAAAB2EOIOygBQwAAOAyVAABAHAQWsCwgwAIAICD0AKGHQRAAAAchG1gYAdzAAEAAFyGCiAAAA7SZQ4gbCAAAgDgILSAYQctYAAAAJehAggAgIPQAoYdBEAAAByEFjDsoAUMAADgMlQAAQBwEFrAsIMACACAg9AChh20gAEAAFyGCiAAAA5CCxh2EAABAHAQWsCwgwAIAICDmGZ32EPAHsAcQAAAAJehAggAgIN0aQHDBgIgAAAOYrIIBDbQAgYAAHAZKoAAADgILWDYQQAEAMBBaAHDDlrAAAAALkMFEAAAB+FKILCDAAgAgINwJRDYQQsYAADAZagAAgDgICwCgR0EQAAAHIRtYGAHARAAAAehAgg7mAMIAADgMlQAAQBwELaBgR0EQAAAHIQWMOygBQwAAOAyVAABAHAQVgHDDgIgAAAOQgsYdtACBgAAcBkqgAAAOAirgGEHARAAAAcxmQMIG2gBAwAAuAwVQAAAHIQWMOwgAAIA4CCsAoYdBEAAAByEOYCwgzmAAAAALkMFEAAAB6EFDDsIgAAAOAgBEHbQAgYAAHAZKoAAADgI9T/YYYifFQAAAFehBQwAAOAyBEAAAACXIQACAAC4DAEQAADAZQiAAAAALkMABAAAcBkCIAAAgMsQAAEAAFyGAAgAAOAyXAoOe55hGMpkMorFYjJNU8ViUcVicdjDAkaCYRg6cOCA8vm8arXasIcDYEQQALHnpdNphUIhzc/Py+/3K5vNqtVqqVKpDHtowFAZhqGpqSkFg8FhDwXAiCEAYk8zDEPj4+NaWFhQo9FQo9FQIBBQIpEgAMLVAoGApqamZBjGsIcCYAQxBxB7WjAYlGEYfa2tWq2mUCg0xFEBwxcOh1Wr1XT8+PFhDwXACKICiD3N5/Op0+n0Het0OvJ4PPJ6vQP3AW6xvr4+7CEAGGFUALGnGYYh0zT7jvVu0/oCAGBrBEDsaaZpDgS93u1utzuMIQEAMPIIgNjT2u22vF5v3zGv16tut0sABADgDAiA2NMajYZM0+xb9BEOh1Wv14c4KgAARhsBEHuaaZoql8vKZrMKBoOKRqOamJhQqVQa9tAAABhZrALGnreysqJMJqPZ2Vl1u10VCgX2AAQA4B4YksyzngUAAADHoAUMAADgMgRAAAAAlyEAAgAAuAwBEAAAwGUIgAAAAC5DAAQAAHAZAiAAAIDLsBE04HCHDh2S3++3bpumqVarpVKptG1XTJmZmVGtVlOhUFA2m5Uk5XK5sz5ufHxc6+vrA88BANhZBEDABfL5vDY2NiRJhmEoEokom82q0+lYx7fLysqKrfPGxsaUTCatALi4uCjTZF96ANgNtIABF+h2u+p0Oup0Omq32yqXy6pWq4rH4zvyWt1u97weRwAEgN1BBRBwKdM0ZZqmZmZm1Gg0FI1GZRiG7r77bnm9XmUyGUUiEXU6Ha2vr2ttbc16bCwWUzqdls/nU7lc7nve01vA8XhcqVRKPp9PjUZD+XxeHo9HU1NTkqRLLrlEd911l6ampvpawGNjY5qYmJDf71ez2dTKyopqtZqkk23ttbU1jY2NKRgMqtlsKpfLqdFo7PjnBgBOQAUQcKFYLKZoNKpKpSLp5Fy85eVlqw07PT2tTqejubk5LS8vW+1aSQoEAtq3b59KpZLm5uYkSZFIZMvXiUQimpqaUrFY1NzcnOr1uvbv369araZ8Pq9Wq6U777xT7Xa773FjY2PKZDJaW1vT3NycqtWq9u/fL5/v53+zplIp6/5ut6tMJrMTHxUAOBIVQMAFMpmMFZAMw5BpmioWi9rY2ND4+LgqlYrq9bokKRwOy+fz6fjx45KkVqullZUVTU1NWVW3Wq1mLSDJ5/OKxWJbvu74+LjK5bI1z29lZUWmacrr9Vpt4k6nM/C4RCJhjU+SVldXFQ6HlUgktLq6Kkkql8va3NyUJBWLRU1PT2/HRwUArkAABFygUChY1T7TNAcqbqfeDgaD8nq9OnLkSN85Ho9HHo9HgUDACos9p9/uCQQCVvjr6QW4exIIBAZWA9frdQUCAet2q9Wyvu92uzIM46zPCwA4iQAIuECn0+kLTKc7ffFFs9nU4uLiwHm9qp3dsHW+izrsPI4FIwBw/pgDCKBPs9mU3++3QmOr1ZLf71cqlbLuDwaDfY85tTJ3qlarNXDuoUOHFAqFzjqGcDjcdywUCqnZbJ7r2wEAbIEACKBPtVpVq9XS1NSUAoGAwuGwstmsVXFbX19XKBRSMpmU3+/X5ORk30bTpyqVSorH4xobG7POlaRGo6FutyuPx7PlY4vFohKJhOLxuPx+v9LptILB4EA7GQBwfmgBAxiwuLioyclJHThwQN1uV5VKxdrgudVqWfcnk0lVKhVrMcbpeqt9k8mkfD6f6vW6FhYWZJqmFTQPHjyoEydO9D2uUqlodXVV6XRaXq9XjUZDCwsL99jGBgDYZ0hiIg0AAICL0AIGAABwGQIgAACAyxAAAQAAXIYACAAA4DIEQAAAAJchAAIAALgMARAAAMBlCIAAAAAuQwAEAABwGQIgAACAyxAAAQAAXIYACAAA4DL/H0evu3n3aFpDAAAAAElFTkSuQmCC' width=640.0/>
</div>



### Test Predictions


```python
test_preds = predict_and_plot(X_test, test_targets, 'Test')
```

    Accuracy: 90.84%
    



<div style="display: inline-block;">
    <div class="jupyter-widgets widget-label" style="text-align: center;">
        Figure
    </div>
    <img src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgCAYAAAA10dzkAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/OQEPoAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA1p0lEQVR4nO3deXhjd33v8c/RvtnWZsljeyaZzCSQ5CYkwA2htJTLUrhNEyiFsl2WhnKBUqBAoUAJlAIFLqQLS+kClAdaoNBcmrKVy1JogQClbGGdkGQ8Y3ss2bIWy9auc/8YdBiNPJkzM7KlOef9yqMn1tHR0e9o8mQ+/n5/v3MMSaYAAADgGp5RDwAAAAC7iwAIAADgMgRAAAAAlyEAAgAAuAwBEAAAwGUIgAAAAC5DAAQAAHAZAiAAAIDLEAABAABchgAIAADgMgRAAAAAlyEAAgAAuAwBEAAAwGUIgAAAAC5DAAQAAHAZAiAAAIDLEAABAABchgAIAADgMgRAAAAAlyEAAgAAuAwBEAAAwGUIgAAAAC5DAAQAAHAZAiAAAIDLEAABAABchgAIAADgMgRAAAAAlyEAAgAAuAwBEAAAwGUIgAAAAC5DAAQAAHAZAiAAAIDLEAABAABchgAIAADgMgRAAAAAlyEAAgAAuIxv1AMAxlE2m9XU1NQ97rO1taXFxcVz/qxkMinTNFUsFk+7bzAYVCKRUDgcltfrVbvd1tbWltbX19Vut895LNuJx+NKJpPyeDxaX1/X+vr6OR8zHA5r7969Onr0qGq12hBGefrPkqTFxUVtbW0N7OP3+7V//35J0l133WX7uzQMQ+l0WvV6XRsbG/e47yWXXKJCoaBCoXCGZwAAw0cABLaxvr6ucrlsPU8mkwqFQlpeXra2dbvdoXxWOp22FQqmpqaUyWS0tbWltbU1tdttBQIBJRIJTUxM6OjRo2o2m0MZU4/H49H09LQ2NzdVLBbVarWGctxGo6EjR44Mfbz3xDRNTUxMbBsAJycnz+qYXq9XiURCKysrp933yJEjOxbSAeBMEQCBbbRarb6w0+l0ZJqm6vX6SMYTCoWUyWRUKpW0urpqba/VaqpWq9q3b59mZmZ05MiRoX6ux+ORYRiqVqtDrdR1u91d/y5rtZpisZhyudzAa7FYTPV6XaFQaMc+f1T/7QDAdgiAwDkIh8NKpVIKhUIyTVPValVra2vqdDrWPqlUSpOTk/J6vep0OtrY2NDa2pqk423B3j6pVEqHDh3a9nOSyaS63a71vhN1Oh2trq4qEAjIMAyZpinpeMUwHo/L7/dbn1soFKzXs9msfD6fNjY2lEwm5fP51Gw2tba2pq2tLU1OTmpmZkaSNDMzo5mZGR06dEj79+/X1tZWX5Dq7dtrnxqGoenpaUWjUatVXS6XrTb3di3gYDCodDpthbBaraa1tTWrSnjie5LJpMLhsLrdriqVyrbfy8k2NjYUiUQUiUT6qoCBQECBQEBra2sDATAajSqRSFjb2+22isWiyuWyfD6fLrroIuv7SaVSuvvuu5XNZuX3+9VsNjUxMaF2u62FhYW+FvCePXsUjUa1sLBg/aKRSqWUTCa1uLi4421xAGARCHCWwuGw5ufnZZqmjh07pnw+r0gkovn5eRmGIUlKJBKKx+MqFApaWlpSqVRSIpFQKpWSJKtiVy6X77F61wstvfB2smq1qvX19b5wl8lkVK1Wtby8rFKppHg8rtnZ2b73hUIhJRIJFQoFq709Ozsrj8ejzc1Na1uhUDij6mIv/K2trWlpaUnValXT09OnbLWGw2Ht27dPkrSysqJcLiefz6e9e/fK7/f37btnzx7VajUtLS1Z4fV08zUlqdlsqtFoaGJiom/75OSkarVaX2iXjoe/ubk5NRoNLS0t6dixY2q1WspmswqFQup0On3fz4nTA8LhsHw+n5aXl7cNp/l8XqZpKpvNSjoefpPJpIrFIuEPwK6gAgicpXQ6rWazqaWlJWtbvV7XhRdeqMnJSZXLZUUiEdXrdVUqFUnHq1qmaVpho9cWbLfbp2wRer1eeTwe2/PvAoGApqamtLq6alXctra21G63rcrT5uamdewjR45Yx15dXdXevXsViURUrVatMbVarTNqYUYiEW1ubloLI2q1mrrd7kDI6tnuu9za2tL+/fuVTqd17Ngxa3u5XLYWotRqNUWjUUWj0b45m6eysbGhRCLRV72MxWLbLmwJBAIql8sDLfeDBw8qHA6rXq/3fT+NRsPazzAM5fP5U87563Q6yuVymp2d1eTkpBKJhBqNhq1KJgAMAxVA4CwYhqFQKGQFqZ5Wq6Vms6loNCrpeIiJRqOan59XIpFQIBBQqVQ67YrRE52q6ncq4XBYkgY+Y2NjQ6ZpWq9Lx4PnicGyF1h6FcyztbW1pXg8rrm5OcXjcfl8Pq2vrw98X73PCoVCA+PtdruqVqt945UG59K12215PPb+V7axsSGv16tIJCLpeAXU5/OpWq0O7FssFpXL5WQYhoLBoGKxmJLJpDXme9LpdE674KNarWpjY8NqGdtZSAIAw0IFEDgLXq9XhmEomUxaoeBEvdBWLBbV7XY1NTWldDqt6elpNRoN5fN5262+XuXs5FboiQzDkGEY6na78nq9krRtta3T6fSFpZPDZe/5uQbA1dVVtdttTUxMKJPJKJPJqFarKZfLDaz87X2Xpxpv73x6zmX1da+S2VsN3Pv3dsf0eDzKZrOKxWKSjreQe39mp/t+7I6xUqloYmJCjUZjV1dEAwABEDgLvVXBxWJx22reicGqXC6rXC7L6/UqGo0qmUxqdnZWd955p+3P29raUjgc7lvkcaKpqSlNT0/ryJEjVpDqLb44UW8hyrk6OQCdXIEzTdO6ZqDP51M0GlUqldKePXu0sLDQt2/vuzw56A1zvCfqzRvM5XKKxWKnbLvu2bNHgUBAi4uLqtfrMk1ThmEoHo8PZRy9hTKNRsO6vqOda0ECwDDQAgbOgmmaajQaCgQCajQa1qPZbCqVSllty71792p6elrS8aBTqVRUKpWseX29Y51OsViU1+u1Fo+cqHctut4ih16V6uTFDhMTEzIM45wvR9LtduXz9f/ueOLqWcMwdOGFFyqRSEiStQJ4Y2Nj2ypm77s8ebwej0exWGzoiyKq1ar1XXq93m3bv9LxVnrv8je9P6Nea38Y0um0tVCkVCoplUopEAgM7fgAcE+oAAJnaW1tTXNzc5qZmbGqgL1Lhpy4SCGRSKjT6ahWq8nn8ymRSPS1HbvdrkKhkMLh8CnDTr1eV6FQUDqdViAQUKVSUafTsSpHHo/HWkDRbDZVLpeVTqfl8XhUq9UUDAaVSqW0tbW17Ty8M1GtVpVMJpVIJFSv1xWLxaw5dZKs6yX27nDSC8qTk5OnnPu4urqq+fl5zc3NqVQqWe11wzCGfueMXhs4mUxa8yK302sV1+t1tdtthcNh65x6FdDen2EkElGz2bQdrsPhsOLxuNbW1tRqtbS2tqZYLKZsNqujR48O50QB4B4QAIGz1LsVXK+12Qs7vZahdDwkmqapyclJ61p+m5ubfStLC4WCUqmU5ubmdPjw4VMuHlhfX1e9Xlc8Hlcmk5HH41G73dbm5ubAreByuZxarZb1ub3r1w3jNm7r6+vyer1WQKtWq8rlcpqbm+v7/HQ6rUQiYbVxy+XyKcNcrVYb+C5rtZpWVlZ2ZG7cxsbGtgtPTrSysmLNX5SOB8dcLqfJyUkr8Ha7Xa2vrysejysajdpq6xuGoWw2q2azabV8TdNUPp/X3NwcrWAAu8KQdGZLDAEAAHBeYw4gAACAyxAAAQAAXIYACAAA4DIEQAAAAJchAAIAALgMARAAAMBlCIAAAAAuQwAEAABwGe4Eco6aq6e/8j/gNuHZXxr1EICx1Gkt7/hnDOvvpcD0gaEcB+OJAAgAgJN0O6MeAc4DBEAAAJzE7I56BDgPMAcQAADAZagAAgDgJF0qgDg9AiAAAA5i0gKGDbSAAQAAXIYKIAAATkILGDYQAAEAcBJawLCBFjAAAIDLUAEEAMBJuBA0bCAAAgDgJLSAYQMtYAAAAJehAggAgJOwChg2EAABAHAQLgQNOwiAAAA4CRVA2MAcQAAAAJehAggAgJPQAoYNBEAAAJyE6wDCBlrAAAAALkMFEAAAJ6EFDBsIgAAAOAmrgGEDLWAAAACXoQIIAICT0AKGDQRAAACchBYwbKAFDAAA4DJUAAEAcBDT5DqAOD0CIAAATsIcQNhAAAQAwEmYAwgbmAMIAADgMlQAAQBwElrAsIEACACAk3RZBILTowUMAADgMlQAAQBwElrAsIEACACAk7AKGDbQAgYAAHAZKoAAADgJLWDYQAAEAMBJaAHDBlrAAAAALkMFEAAAJ6ECCBsIgAAAOIhpciFonB4BEAAAJ6ECCBuYAwgAAOAyVAABAHASLgMDGwiAAAA4CS1g2EALGAAAwGWoAAIA4CS0gGEDARAAACehBQwbaAEDAAC4DBVAAACchBYwbCAAAgDgJLSAYQMtYAAAAJehAggAgJNQAYQNBEAAAJyEOYCwgQAIAICTUAGEDcwBBAAAcBkqgAAAOAktYNhAAAQAwEloAcMGWsAAAAAuQwUQAAAnoQUMGwiAAAA4CS1g2EALGAAAwGWoAAIA4CRUAGEDARAAACcxzVGPAOcBWsAAAAAuQwUQAAAnoQUMGwiAAAA4yQgCoGEYymQyisViMk1TxWJRxWJx231jsZhSqZT8fr8ajYby+bwajcYujxi0gAEAcBKzO5zHGUin0wqFQlpcXFQ+n1cymVQsFhvYLxAIaGZmRuvr61pYWFCj0dDc3JwMwxjW2cMmAiAAADhrhmFoamrKquRVq1UVi0XF4/GBfSORiJrNpjY2NtRqtbS6uiqfz6dAILD7A3c5WsAAADjJLreAg8GgDMNQrVazttVqNSWTyYF9O52OAoGAQqGQ6vW6pqam1Ol01Gq1dnPIEAEQAABnGdJlYAzDGGjNmqYp86Tj+3w+dTqdvm2dTkcej0der7fvtWq1qlgspn379lnHWVpaUpeFK7uOAAgAAAYkk0mlUqm+bYVCQYVCoW+bYRgDobD3/OQA6fF45PP5lMvlVK/XFY/Hlc1mdeTIkYEQiZ1FAAQAwEmGVE1bX18fWMl7ctDrbTs56PWen1zZm56eVqPRULlcliTlcjldeOGFmpycPOWqYewMAiAAAE4ypAC4Xbt3O+12W16vt2+b1+tVt9sdCIDBYFClUqlvW6PRkN/vP+fx4sywChgAAJy1RqMh0zQVCoWsbeFwWPV6fWDfdrs9sOI3EAiwCGQECIAAADjJLl8H0DRNVSoVZbNZBYNBRaNRJRIJq9Ln9XqtlnC5XNbU1JQmJibk9/uVTqfl8/lUqVR24pvAPaAFDACAg5jd4awCPhOrq6vKZDLau3evut2uCoWCqtWqJOnAgQNaWVlRpVJRtVq1LhTt9/tVr9e1uLjIApARIAACAIBzYpqmcrmccrncwGuHDh3qe16pVKj4jQECIAAATsI19WADARAAACc5w/v4wp0IgAAAOMkI5gDi/MMqYAAAAJehAggAgJMwBxA2EAABAHASAiBsoAUMAADgMgRAjK1Go6mb3vhneuAjH6eH3PBkve9Dt5xy3698/b/02Kf/jv77w39dv/3CV+juhUXrNdM09Xcf/Cc98nHP0AMf+Ti96g1/qq2t2m6cAjAUwWBQf/PXb9Va/oc6uvAtvej3nn3Kfa+66nJ99csfV6X0U9321U/qvldfse1+r3j5C/Sed//Ztq99+pMf1NOe+ptDGTtGwDSH84CjEQAxtm5+57v1gx/fofe87U161Uuep3e99x/0//7tPwb2++ldC3reS1+jh/7itfrIe96uSy85qGe+4OVWyPvorZ/WX773H/TCZz9DH3jXzcqtrullf/Tm3T4d4Ky9+U2v0v3udx894ld+U7/7glfqple9SI997HUD+0UiYX381g/oy1/+hq659lG67bZv6l9ufb8ikXDffk94wqP1mle/ZOD9hmHoz//sdXrEI355x84Fu6DbHc4DjkYAxFjaqtV1y8c/o5e/8Dm67F4H9fBffpBufMrj9cFbPj6w7z9+7JO66opL9bvPepr2XzCvF//OjYrFIvrE//s3SdIH/+lf9PQnPla/+oiH6OBFF+hPXvX7+tJXv9FXJQTGVSQS1jNvfJJe/OJX69vf+b5uvfVf9dab36XnPfcZA/v+5uNvUK1W18te/jr9+Mc/1Ytf8hptbGzqcb9xvaTj92R9x9vfqHf/zc26866FvvfOzs7os5/5iK7/tV9RsVjahTMDMEoEQIyln/z0LrU7bV19xaXWtquvvFy3/+An6p70m+ni8jFdcdm9reeGYejii/bru9//0c9eX9GVl93Len06nVQiPmW9Doyz+1x5ufx+v7562zetbV/5yjd0zTVXyzCMvn0f8ID76itf/c++bV+97T917bX3kyTFYlFdecWl+oVfvF5f+9p/9e1336uv0NHFZV1z7aNULm/s0NlgV3TN4TzgaK5bBezxeGQYhkzTHAgSGB9ra+uKT03J7/db21LJuBrNpkrlipKJ+AnbE8qvrvW9fyW/qqnJCet9+bWC9dpWra5KZUPFcnlnTwIYgpk9Ga2travValnbcvlVhcNhpVIJra2tW9v37Mnqhz/8Sd/78/lVXf6zX5DK5Yoe/JDHbPs5n/jkZ/WJT352+CeA3cedQGCDKyqAsVhM8/PzOnjwoA4cOKCLLrpIBw4c0MGDBzU/P69oNDrqIeIktUZDgRPCnyTrefOEvwgl6VEPe7A+829f1he/8nW12x3d+qnP6gc/OmT9hfmohz1Y7/7AR3Tn4SNqNJp6y9v/RpLUarV34UyAcxOJhNVoNPu29Z4Hg8H+fcPb7xsMBnZ2kADOO46vAMbjcaVSKRWLRRUKBbXbbZmmKcMw5PP5FA6HNTMzo0KhoFKpNOrh4meCgcBA0Os9D4dCfdt/8dr767k3Plkv+sPXq9Pp6pr7Xqnr/+fDVK1uSpKe/Ywna3F5RY/5X8+Rz+fV4x/9q7rXxRcpFo3szskA56BebwwEuN7zk1ezn2rfrRqr3l2F9i1scHwATCaTWllZ0ebm5sBrrVZLtVpNjUZDmUyGADhGMtMplcpltdsd+XxeSVKhUFQoGNREbLBi++ynP0m/9aTf0MbmllKJuF5y059odk9WkhQJh3Tz616pjeqmDEOKRaN68HVPtF4Hxtny0orS6aS8Xq86nY4kaSab0dZWTaVS/zSGpeVjymYzfduy2YyOHcvv2ngxeibTm2CD41vAhmH0zZ3ZTrvdlsfj+K/ivHLviy+Sz+vT937w84Ua3/reD/TfLr144M/qU5/9ot7053+lQCCgVCKueqOhb3zru7rmvveRJN38zvfo1k99VhOxqGLRqG7/0U+0sbmpq6+4bFfPCTgb3/nu99VqtXTtA+5rbXvQg67RN7/5HZknXavt61//lh74wPv3bfuFB95fX/96/4IPOByLQGCD41NPtVrVzMyMwuHwtq+HQiHNzMyoWq3u8shwT8KhkG74nw/XH7/lHbr9Rz/R5//9q3rfh27RUx7/GEnSWmFd9UZDknTB3jl95NZP6bNf/IoWji7pZX/0Zs1kpvVL1x7/izCTTupdf/dB3f6jn+gHP75Dr3jtW/SEx1xnLRIBxlmtVtf7P/BPeuc736T73+8+uuGGR+rFL3q23vaO90iSstlphX42LeKW//tJxacm9ac3v1aXXnqx/vTm1yoajeij/zR4+SQA7ub4FnA+n1c6ndbc3JwMw1Cn07HmAHq9XpmmqUqlotXV1VEPFSd52Quepde95R268fkv10Q0quc983/pEQ95kCTpITc8Ra9/5Yv1mOseocvvfbFu+v3f1Vvf8bcqlSt6wP2v0l++5Y+tSuGTH3eDlo7l9NyXvFoew9D1j3qYXvTcG0d5asAZ+f2X/pHe+Y436XOf/ajK5Ype+8c365//+dOSpKWj39GNz3yR3v+Bj2hjo6pHP+bpeuc736Rn/fZTdPvtP9L1j34qd75xG1YBwwZDkivqvIZhKBgMyufzWZeBabfbajQaA22UM9FcvXOIowScITz7S6MeAjCWOq3lHf+M6mufPJTjxF7zwaEcB+PJ8RXAHtM0Va/XRz0MAACAkXNNAAQAwBVYBQwbCIAAADgJK3hhg+NXAQMAAKAfFUAAAJyEVcCwgQAIAICT0AKGDbSAAQAAXIYKIAAADsK9gGEHARAAACehBQwbCIAAADgJARA2MAcQAADAZagAAgDgJFwGBjYQAAEAcBJawLCBFjAAAIDLUAEEAMBBTCqAsIEACACAkxAAYQMtYAAAAJehAggAgJNwJxDYQAAEAMBJaAHDBlrAAAAALkMFEAAAJ6ECCBsIgAAAOIhpEgBxegRAAACchAogbGAOIAAAgMtQAQQAwEmoAMIGAiAAAA7CreBgBy1gAAAAl6ECCACAk1ABhA0EQAAAnIQ7wcEGWsAAAAAuQwUQAAAHYREI7CAAAgDgJARA2EALGAAAwGWoAAIA4CQsAoENBEAAAByEOYCwgwAIAICTUAGEDcwBBAAAcBkqgAAAOAgtYNhBAAQAwEloAcMGWsAAAAAuQwUQAAAHMakAwgYCIAAATkIAhA20gAEAAFyGCiAAAA5CCxh2EAABAHCSEQRAwzCUyWQUi8VkmqaKxaKKxeK2+wYCAWWzWQWDQbVaLeXzedVqtV0eMWgBAwCAc5JOpxUKhbS4uKh8Pq9kMqlYLDawn8fj0fz8vBqNhhYWFlStVjU7Oyuv1zuCUbsbFUAAABxkt1vAhmFoampKS0tLajQaajQaCgQCisfjqlarfftOTk6q2+0qn89LkgqFgqLRqEKhkDY3N3d34C5HAAQAwEF2OwAGg0EZhtHXxq3VakomkwP7hsPhgVB45MiRHR8jBhEAAQBwkN0OgD6fT51Op29bp9ORx+OR1+vte83v96ter1vzBVutllZXV1Wv13d30GAOIAAAGGQYhjweT9/DMIxt9zPN/vsP956fvL/H41EymVSn09HS0pJqtZrm5+fl81GP2m184wAAOIk5GNLORjKZVCqV6ttWKBRUKBT6P840B4Je73m3O1iObDQa1jEajYYikYgmJye1vr4+lHHDHgIgAAAOMqwW8Pr6+sClXE6u9ElSu90eWMXr9XrV7XYHAmC73Vaz2ezb1mq1qACOAC1gAAAwwDRNK8T1HtsFwEajIdM0FQqFrG3hcHjbeX31el3BYLBvWyAQUKvVGv4J4B4RAAEAcBCzawzlYfvzTFOVSsW6uHM0GlUikVCpVJJ0vBrYawmXSiUFg0GlUin5/X7r3xsbGzvxVeAeUHMFAMBBRnEruNXVVWUyGe3du1fdbleFQsG63MuBAwe0srKiSqWidrutxcVFZTIZJRIJNZtNLS0tqd1u7/6gXY4ACAAAzolpmsrlcsrlcgOvHTp0qO95vV7n2n9jgAAIAICDmENaBQxnIwACAOAgo2gB4/zDIhAAAACXoQIIAICDnMkKXrgXARAAAAfZ5lJ9wAACIAAADkIFEHYwBxAAAMBlqAACAOAgVABhBwEQAAAHYQ4g7KAFDAAA4DJUAAEAcBBawLCDAAgAgINwKzjYQQsYAADAZagAAgDgINwLGHaMdQUwmUzKMAZL2R6PR+l0egQjAgBgvHVNYygPONvYVQD9fr98vuPDSqVSajQa6nb7f50JBAKKx+NaW1sbxRABAADOa2MXAH0+n+bn563ns7OzA/uYpqlisbibwwIA4LzAIhDYMXYBsFar6Y477pAk7d+/XwsLCwMVQAAAsD0uAwM7xi4Anujuu++WJBmGoUAgoGazKcMwCIQAAJwCdwKBHWMdAA3DUCaT0eTkpCTp8OHDSqfT8ng8OnbsGEEQAADgLIz1KuB0Oq1AIKCFhQWZP/uVplAoyOv1KpPJjHh0AACMH7NrDOUBZxvrCmAsFtPy8rKazaa1rdlsKpfL9S0UAQAAx3EJF9gx1hVAj8djVf4AAAAwHGMdADc3N5VOp62LQZumKZ/Pp0wmo83NzRGPDgCA8WOaxlAecLaxbgHn83lls1kdPHhQknTBBRfI4/Foa2tL+Xx+xKMDAGD80DiDHWMdALvdro4dOya/369AICDp+BzAVqs14pEBAACcv8Y6AIbDYevn3iVffD6ffD6fTNNUu91Wu90e1fAAABg7LAKBHWMdALPZrPx+v6SfB0CPp3/aYr1e1/Lysjqdzq6PDwCAccP8Pdgx1gGwUqkoGo1qZWXFavv6/X5ls1lVq1VVKhVls1llMhkdO3ZsxKMFAAA4P4z1KuB4PK5cLtc356/VaimfzyuZTKrb7apQKCgSiYxwlAAAjA/THM4DzjbWFUBJ8nq9227rXRoGAAD8HHMAYcdYB8BKpaKZmRkVCgXV63VJUigUUiqVUqVSkcfjUTqd1tbW1sjG2P7u50b22cC4ujx5waiHAIyl7+WWd/wzmAMIO8Y6AK6tranb7SqVSsnnOz7UdrutUqmkYrGoSCQi0zS5JiAAAMAZGOsAODExoVKppPX1dWv1b281sCRtbW2NtPoHAMC4oQUMO8Z6EUgmk7HmAHa73b7wBwAABplDesDZxjoAbm1taXJykgUfAAAAQzTWLWCfz6dYLKZkMqlOpzNQATx8+PBoBgYAwJiiBQw7xjoAlstllcvlUQ8DAIDzBquAYcdYB8BKpTLqIQAAADjOWAdAr9erZDKpQCDQNw/QMAwFAgHdeeedIxwdAADjh+WSsGOsF4Fks1lFo1HV63WFw2HV63V1Oh2FQiEVCoVRDw8AgLFjyhjKA8421hXASCSixcVF1et1RaNRVatV1et1JRIJRaNRlUqlUQ8RAADgvDPWFUDp+J0/JKnRaCgUCkmSNjY2rJ8BAMDPdc3hPOBsYxcAw+Gw9XO9Xtfk5KSk4wEwEolIkvx+/0jGBgDAuOvKGMoDzjZ2AXB+ft66+8fa2poSiYTi8bgqlYpCoZAuuOACzc7OamNjY8QjBQBg/DAHEHaM9RzAer2uu+++W4ZhqNvtamFhQbFYTN1ulwAIAABwlsY6AErqu/tHp9PhwtAAANwDLgMDO8YyAO7bt0+mefoZqNwKDgCAfrRvYcdYBsBisThw318AAAAMx1gGwI2NDXU6nVEPAwCA8w7lE9gxlgEQAACcHQIg7Bi7y8BUKhXavwAAADto7CqAuVxu1EMAAOC8xSIQ2DF2ARAAAJy9LvkPNoxdCxgAAAA7iwogAAAOwn18YQcBEAAABzn9bRQAAiAAAI7CdTRgB3MAAQAAXIYKIAAADtI1mAOI0yMAAgDgIMwBhB20gAEAAFyGCiAAAA7CIhDYQQAEAMBBuBMI7CAAAgCAc2IYhjKZjGKxmEzTVLFYVLFYvMf3+Hw+XXjhhVpaWlKtVtulkaKHAAgAgIOM4k4g6XRaoVBIi4uL8vv9ymazarVaqlarp3xPNpuVx8NShFEhAAIA4CC7vQrYMAxNTU1paWlJjUZDjUZDgUBA8Xj8lAFwYmKC8DdifPsAADhI1xjOw65gMCjDMPrauLVaTaFQaNv9PR6PpqenlcvlzvVUcQ6oAAIAgAGGYcg46aLSpmnKNPtrjD6fT51Op29bp9ORx+OR1+sdeG16elrlclnNZnNnBg5bCIAAADjIsC4Dk0wmlUql+rYVCgUVCoW+bYZhDITC3vOTA2QkElE4HNbCwsKQRomzRQAEAMBBhjUHcH19fWAl78lBr7ft5KDXe97tdvu2ZTIZ5fP5bY+D3UUABAAAA7Zr926n3W7L6/X2bfN6vep2u30BMBQKKRAIaHZ2tm/fubk5VSoV5fP54QwcthAAAQBwkN2+EHSj0ZBpmgqFQqrX65KkcDhs/dxTr9d19913923bv3+/crmctra2dm28OI4ACACAg+z2reBM01SlUlE2m9XKyop8Pp8SiYS1yrdXDTRNU61Wa+D97XZ7YKEIdh6XgQEAAOdkdXVV9Xpde/fuVTabVaFQsK4BeODAAU1MTIx4hDgZFUAAABxktyuA0vEqYC6X2/bafocOHTrl++7pNewsAiAAAA5i7v6d4HAeogUMAADgMlQAAQBwkFG0gHH+IQACAOAgBEDYQQAEAMBBuMcG7GAOIAAAgMtQAQQAwEF2+04gOD8RAAEAcBDmAMIOWsAAAAAuQwUQAAAHoQIIOwiAAAA4CKuAYQctYAAAAJehAggAgIOwChh2EAABAHAQ5gDCDlrAAAAALkMFEAAAB2ERCOwgAAIA4CBdIiBsIAACAOAgzAGEHcwBBAAAcBkqgAAAOAgNYNhBAAQAwEFoAcMOWsAAAAAuQwUQAAAH4U4gsIMACACAg3AZGNhBCxgAAMBlqAACAOAg1P9gBwEQAAAHYRUw7KAFDAAA4DJUAAEAcBAWgcAOAiAAAA5C/IMdBEAAAByEOYCwgzmAAAAALkMFEAAAB2EOIOwgAAIA4CDEP9hBCxgAAMBlqAACAOAgLAKBHQRAAAAcxKQJDBtoAQMAALgMFUAAAByEFjDsIAACAOAgXAYGdtACBgAAcBkqgBhbjVZbb/zw5/W579yhkN+npz38/nraw++/7b5f+M4devutX9ZKaUP3mp/WHzz+obp0X1ZLhbKuu+nd277nPS96gu538fxOngKwIwLBgF75xpfoYdc9RI16Q+9/14f0/r/60D2+5+prrtTr336TrnvA461tHo9Hz3/Fs3XDE35V4UhIX/7C1/SmV/6p1teKO30K2EHU/2AHARBj68/+75f0wyM5/e0LH69j6xXd9P5/1Z7kpB5x30v69vvp8ppe8Xef0que9HBddWBOf/+F/9Lz//Jj+vgfP1MziQl97o3P6dv/rbd8UUdXS7ryoj27eTrA0Lz41c/TZfe5t571uOdrdn5Gr3vbTVpeXNHnPvFv2+5/8N4X6a3vfoMajWbf9huf/1Q98jEP10v/900qrZf0B69/kf7kHa/Rc574e7twFtgptIBhBy1gjKVao6WPffX7eunj/4cu3ZfVQ6+6WM94xH/Xh7/07YF9b/vRgg7sSen6ay/X3um4XvDoX9JaZVN3HSvI6/EoPRW1HotrJX3+O3fo9U9/lPxe7wjODDg34UhIv/7kG/R/bvpz/fj2Q/rCp/9d73vnP+iJN/7Gtvs/7qmP1vs/8dcqrK4PvOb1efXWV/+FvvW17+iuQ4f1wXd/VFddc+VOnwJ2WHdIDzgbARBj6SeLebU7HV110ay17aoDc/r+4RV1u/2/3cajId15rKBv37mkbtfUrbd9X7FQQHun4wPHfdut/6HHPugK7Z9J7fQpADvikssOyuf36jv/ebu17dvf+K6uuPpyGYYxsP+DHvpA3fSC1+vv/+YfB17765vfqy98+t8lScl0Qo99yvX65m3f2rnBAxgbtIAxltYqm4rHwvL7fl6lS01E1Gi1VdqsKTkRsbY/8n730pduv1O/dfOH5fUYMgxDb/+dX9dkJNR3zG/fuaTv3XVMb7zxul07D2DY0tm0SutltVtta1thdV2hcFDx5JSKhVLf/i/6rZdLkm54wq+e8pjPfekz9ZyXPFPlYkVPv+HZOzJu7B4uBA07qABiLNWbbQV8/S3agP/481a707e9tFnXWmVTL3/CQ/WBlz1F1z/gMr3mA5/R+sZW3363fPl7euhVB5WNT+zs4IEdFA6H1Gy2+rY1G8ef+wP+szrmJz76r3rSI2/U1/7jP/VXH/5zRWOR078JY4sWMOwgAGIsBfxeNU8Kes3W8eehQH/h+i/++d918ey0nvjLV+uyfVnd9ORfUTjg1623fd/ap93p6ovfu1PXXXPZzg8e2EGNRkOBk4JeIHj8eb1WP6tjHj28pB9+98d61fNfp2AoqIdd95BzHSaAMeeKFnA4HLa9b61W28GRwK7MVEylak3tTlc+7/HfUwqVTYX8Pk2E+1u7PzqS05Mecl/rucdj6JL5aS0XKta27929rHano2svvWB3TgDYIfljq4onp+T1etXpHP+lKJ1JqbZV10a5ekbHevAjfkE/vv2Q8itrkqRmo6mlI8uKJ6eGPm7sHlrAsMMVATCTySgQCNja94477tjh0cCOe+3NyOf16va7l3X1wePX6vv2nUu6/IIZeTz9E92np2K6a6XQt20hV9Tl18xYz2+/+5gu25dV0O+K/+ThYD/5wR1qtzq68n6X69vf+J6k49f4+8F3fyTTPLO/+F/86ufrXz7yKb337R+QJEWiEe27aK/uvmNh6OPG7qF9Cztc8bfhkSNHNDMzI7/fr6NHj57x/ySx+8IBv65/wGV6/Yc+p9c+9VHKlzb0/s99U6996qMkSWvlTcXCAYUCfj32QVfo1R/4jC6/YEZX7t+jj331di2vV3TDtZdbx/vpsYIuYuUvHKBea+jjH/mUXvV/XqZX/94blJmZ1tOe+2S95vfeIElKTSdV3aiqUW+e5kjSP77vFj33939bh374Ux1bXNHzX/EcHT28qC9//radPg0AI+aKAGiaplZWVrR3716lUimtra2Nekiw4SWPe4je8KHP6Vl/8RHFQkE959d+QQ+7+mJJ0sNf8Vd67VMfqUc/8L/pkfe/t7YaLb3nX7+uXGlD95rP6G9f+Pi+lcLrlU3daz4zqlMBhuqtf/Q2/eGbX6p33/J2VSubetdb3q3Pf+pLkqQv3P4J3fTC1+tf/vFTpz3Oh997i8LhsP7wzS9VIhnXbV/6hl74tD/gl+TzXJc/P9hgyEV3jQkEAgqHwyqXy0M75tbn/npoxwKc4tqnvG/UQwDG0vdyO19dfcq+Xx/Kcf7hyMeGchyMJ1dUAHuazaaazdO3RQAAAJzMVQEQAACn417AsIMACACAg3AZGNhBAAQAwEG4DAzs4E4gAAAALkMFEAAAB2EOIOwgAAIA4CDMAYQdtIABAABchgogAAAOwiIQ2EEABADAQbiVH+wgAAIAgHNiGIYymYxisZhM01SxWFSxWNx232g0qlQqpUAgoFarpbW1NW1ubu7yiEEABADAQUaxCjidTisUCmlxcVF+v1/ZbFatVkvVarVvv0AgoD179lihLxKJaHZ2VgsLC9yqdZexCAQAAAfpDulhl2EYmpqaUj6fV6PRULVaVbFYVDweH9h3cnJStVpNpVJJrVZL5XJZW1tbmpiYONvTxVmiAggAAM5aMBiUYRiq1WrWtlqtpmQyObBvpVLZ9hher3fHxoftUQEEAMBBzCH9Y5fP51On0+nb1ul05PF4BoJds9nsa/UGAgFFIhFtbW2d20njjFEBBADAQYY1B9AwDBmG0bfNNM2BVcaGYQxs6z0/+f0n8ng8mp2dVa1WG5griJ1HAAQAwEGGdRmYZDKpVCrVt61QKKhQKAx83slBr/e8291+NqHX69X8/Lwk6dixY0MZL84MARAAAAxYX18fuJTLduGy3W4PtHq9Xq+63e62AdDn81nh7+jRowPtY+wOAiAAAA4yrDuBbNfu3U6j0ZBpmgqFQqrX65KkcDhs/XwiwzA0Nzcn0zS1uLhI+BshFoEAAOAgu70IxDRNVSoVZbNZBYNBRaNRJRIJlUolScergb2WcDKZlN/vVy6Xs17zer3yeIgju40KIAAAOCerq6vKZDLau3evut2uCoWCtbDjwIEDWllZUaVS0cTEhDwej/bt29f3/nK5bIVC7A4CIAAADjKKO4GYpqlcLrdtiDt06JD18+HDh3dxVLgnBEAAABxkWKuA4Ww03QEAAFyGCiAAAA4yihYwzj8EQAAAHORMVvDCvWgBAwAAuAwVQAAAHKTLIhDYQAAEAMBBiH+wgwAIAICDsAgEdjAHEAAAwGWoAAIA4CBUAGEHARAAAAfhTiCwgxYwAACAy1ABBADAQWgBww4CIAAADsKdQGAHLWAAAACXoQIIAICDsAgEdhAAAQBwEOYAwg5awAAAAC5DBRAAAAehBQw7CIAAADgILWDYQQAEAMBBuAwM7GAOIAAAgMtQAQQAwEG6zAGEDQRAAAAchBYw7KAFDAAA4DJUAAEAcBBawLCDAAgAgIPQAoYdtIABAABchgogAAAOQgsYdhAAAQBwEFrAsIMWMAAAgMtQAQQAwEFoAcMOAiAAAA5CCxh2EAABAHAQ0+yOegg4DzAHEAAAwGWoAAIA4CBdWsCwgQAIAICDmCwCgQ20gAEAAFyGCiAAAA5CCxh2EAABAHAQWsCwgxYwAACAy1ABBADAQbgTCOwgAAIA4CDcCQR20AIGAABwGSqAAAA4CItAYAcBEAAAB+EyMLCDAAgAgINQAYQdzAEEAABwGSqAAAA4CJeBgR0EQAAAHIQWMOygBQwAAOAyVAABAHAQVgHDDgIgAAAOQgsYdtACBgAAcBkqgAAAOAirgGEHARAAAAcxmQMIG2gBAwAAuAwVQAAAHIQWMOwgAAIA4CCsAoYdBEAAAByEOYCwgzmAAAAALkMFEAAAB6EFDDsIgAAAOAgBEHbQAgYAAHAZKoAAADgI9T/YYYj/VgAAAFyFFjAAAIDLEAABAABchgAIAADgMgRAAAAAlyEAAgAAuAwBEAAAwGUIgAAAAC5DAAQAAHAZAiAAAIDLcCs4nPcMw1Amk1EsFpNpmioWiyoWi6MeFjAWDMPQvn37lM/nVavVRj0cAGOCAIjzXjqdVigU0uLiovx+v7LZrFqtlqrV6qiHBoyUYRiamZlRMBgc9VAAjBkCIM5rhmFoampKS0tLajQaajQaCgQCisfjBEC4WiAQ0MzMjAzDGPVQAIwh5gDivBYMBmUYRl9rq1arKRQKjXBUwOiFw2HVajUdOXJk1EMBMIaoAOK85vP51Ol0+rZ1Oh15PB55vd6B1wC3KJfLox4CgDFGBRDnNcMwZJpm37bec1pfAABsjwCI85ppmgNBr/e82+2OYkgAAIw9AiDOa+12W16vt2+b1+tVt9slAAIAcAoEQJzXGo2GTNPsW/QRDodVr9dHOCoAAMYbARDnNdM0ValUlM1mFQwGFY1GlUgkVCqVRj00AADGFquAcd5bXV1VJpPR3r171e12VSgUuAYgAAD3wJBknnYvAAAAOAYtYAAAAJchAAIAALgMARAAAMBlCIAAAAAuQwAEAABwGQIgAACAyxAAAQAAXIYLQQMOt3//fvn9fuu5aZpqtVoqlUpDu2PK/Py8arWaCoWCstmsJCmXy532fVNTUyqXywPHAADsLAIg4AL5fF4bGxuSJMMwFIlElM1m1el0rO3Dsrq6amu/yclJJZNJKwAuLy/LNLkuPQDsBlrAgAt0u111Oh11Oh21221VKhVtbW1pYmJiRz6r2+2e1fsIgACwO6gAAi5lmqZM09T8/LwajYai0agMw9Dhw4fl9XqVyWQUiUTU6XRULpe1vr5uvTcWiymdTsvn86lSqfQd9+QW8MTEhFKplHw+nxqNhvL5vDwej2ZmZiRJl1xyie666y7NzMz0tYAnJyeVSCTk9/vVbDa1urqqWq0m6Xhbe319XZOTkwoGg2o2m8rlcmo0Gjv+vQGAE1ABBFwoFospGo2qWq1KOj4Xb2VlxWrDzs7OqtPpaGFhQSsrK1a7VpICgYD27NmjUqmkhYUFSVIkEtn2cyKRiGZmZlQsFrWwsKB6va65uTnVajXl83m1Wi3deeedarfbfe+bnJxUJpPR+vq6FhYWtLW1pbm5Ofl8P/+dNZVKWa93u11lMpmd+KoAwJGoAAIukMlkrIBkGIZM01SxWNTGxoampqZUrVZVr9clSeFwWD6fT0eOHJEktVotra6uamZmxqq61Wo1awFJPp9XLBbb9nOnpqZUqVSseX6rq6syTVNer9dqE3c6nYH3xeNxa3yStLa2pnA4rHg8rrW1NUlSpVLR5uamJKlYLGp2dnYYXxUAuAIBEHCBQqFgVftM0xyouJ34PBgMyuv16uDBg337eDweeTweBQIBKyz2nPy8JxAIWOGvpxfg7kkgEBhYDVyv1xUIBKznrVbL+rnb7cowjNMeFwBwHAEQcIFOp9MXmE528uKLZrOp5eXlgf16VTu7YetsF3XYeR8LRgDg7DEHEECfZrMpv99vhcZWqyW/369UKmW9HgwG+95zYmXuRK1Wa2Df/fv3KxQKnXYM4XC4b1soFFKz2TzT0wEAbIMACKDP1taWWq2WZmZmFAgEFA6Hlc1mrYpbuVxWKBRSMpmU3+/X9PR034WmT1QqlTQxMaHJyUlrX0lqNBrqdrvyeDzbvrdYLCoej2tiYkJ+v1/pdFrBYHCgnQwAODu0gAEMWF5e1vT0tPbt26dut6tqtWpd4LnValmvJ5NJVatVazHGyXqrfZPJpHw+n+r1upaWlmSaphU0L7jgAh09erTvfdVqVWtra0qn0/J6vWo0GlpaWrrHNjYAwD5DEhNpAAAAXIQWMAAAgMsQAAEAAFyGAAgAAOAyBEAAAACXIQACAAC4DAEQAADAZQiAAAAALkMABAAAcBkCIAAAgMsQAAEAAFyGAAgAAOAyBEAAAACX+f+3T0B2BkrO7wAAAABJRU5ErkJggg==' width=640.0/>
</div>



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
