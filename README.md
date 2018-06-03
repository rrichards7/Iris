# Deep Neural Network (DNN) for Iris Dataset

Ryan J. Richards

Machine Learning using TensorFlow 1.8

## 1. Introduction/Methodology

This readme page will discuss how flowers contained within the Iris dataset are classified with a Deep Neural Network (DNN) using TensorFlow's high level APIs for Python. This is my broken down and "linear" version of TensorFlow's example located here: https://www.tensorflow.org/get_started/premade_estimators

## 2. Code

This section will break down and dicuss each section of the code in further detail.

### 2.1 Import Libraries

Import the following libs:

 ```python
#imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
```


### 2.2 Dataset

#### 2.2.1 Understanding the Iris Dataset

There are many other resources online regarding this dataset that will explain in the features/labels, point of the experiment, etc in more detail but here is a quick breakdown. The training and testing dataset is arranged as follows: first 4 columns are features (Sepal Length, Sepal Width, Pedal Length, Pedal Width) and the 5th (last) column is the label column (Iris species). The table below illustrates this:


| Sepal Length  | Sepal Width   | Pedal Length  | Pedal Width | Species|
| ------------- |:-------------:| :-----:|:-------------:|-------------:|
| test     | test | test |
| test      | test      |   test |
| test | test      |    test |

#### 2.2.2 Download Dataset using pandas

#### 2.2.3 Parse Dataset


### 2.3 Select Model Type




### 2.4 Train the Model




### 2.5 Evaluate/Test the Model



### 2.6 Predictions



### 2.7 Visualization using TensorBoard



## 3. Conclusion

To conclude...
