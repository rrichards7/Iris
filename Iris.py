'''
Ryan J. Richards
Iris Flower classification/prediction

TensorFlow v1.8
Python v3.6
'''

#imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd

#feature column names
feature_column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

#bring in the data...
train_dataset = pd.read_csv(filepath_or_buffer="/Users/Ryan/Downloads/iris_training.csv", names=CSV_COLUMN_NAMES, header=0)
test_dataset = pd.read_csv(filepath_or_buffer="/Users/Ryan/Downloads/iris_test.csv", names=CSV_COLUMN_NAMES, header=0)

#split up the training data and define as features/labels
train_features = train_dataset.drop(columns=train_dataset.columns[4], axis = 1)
train_labels = train_dataset.drop(columns=train_dataset.columns[[0,1,2,3]], axis = 1)

#split up the testing data and define as features/labels
test_features = test_dataset.drop(columns=test_dataset.columns[4], axis = 1)
test_labels = test_dataset.drop(columns=test_dataset.columns[[0,1,2,3]], axis = 1)

# Feature columns describe how to use the input.
my_feature_columns = []

my_feature_columns.append(tf.feature_column.numeric_column(key="SepalLength"))
my_feature_columns.append(tf.feature_column.numeric_column(key="SepalWidth"))
my_feature_columns.append(tf.feature_column.numeric_column(key="PetalLength"))
my_feature_columns.append(tf.feature_column.numeric_column(key="PetalWidth"))

#initialize classifier
classifier = tf.estimator.DNNClassifier\
(
    feature_columns = my_feature_columns,
    hidden_units = [10, 10],
    n_classes = 3,
    model_dir="/Users/Ryan/Desktop/TB_Model_Directory"
)

#define the TRAINING input function
def input_fn_train(features, labels, batch_size):
    #convert features/labels to dataset object
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    #return dataset
    return dataset

#train the DNN
classifier.train\
(
    input_fn = lambda : input_fn_train(features=train_features, labels=train_labels, batch_size=100),
    steps = 100
)


print('fit done')

#define the TESTING input function
def input_fn_test(features, labels, batch_size):
    #convert features/labels to dataset object
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    #return dataset
    return dataset

eval_result = classifier.evaluate\
(
    input_fn= lambda : input_fn_test(features=test_features, labels=test_labels, batch_size=10),
    steps=100
)

print(eval_result)

predict_x = \
    {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


predictions = classifier.predict \
(
    input_fn = lambda : eval_input_fn(features = predict_x, labels = None, batch_size = 100)
)

# Generate predictions from the model
expected = ['Setosa', 'Versicolor', 'Virginica']

template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(template.format(SPECIES[class_id], 100 * probability, expec))
