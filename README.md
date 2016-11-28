# Tensorflow LSTM RNN Example

## Running it.

### Basic train and test

```
python train.py
python test.py
python test.py # To see it's deterministic
```

Install the obvious dependencies if it fails because they're missing.

### Syslog prediction server (only designed to work with dummy data)

```
python train.py # unless you've already done so
python dummy_prediction_server.py &
echo 01010101010101 | nc 127.0.0.1 5140
# => prediction: [7]
echo 01010101000001 | nc 127.0.0.1 5140
# => prediction: [5]
kill %1
```

The number of binary digits of the `echo`ed strings, and the port for `nc` are determined by
the `NUM_DIGITS` and `PORT` variables in `load_data.py` and `dummy_prediction_server.py`,
respectively.

## Tweaking it.

Any variable in ALL_CAPS in any file might be of interest to tweak, and safe-ish to tweak
independently.  Hopefully all variables are sufficiently descriptively named.

## What is it?

`train.py` imports training data from `load_data.py`, and uses Tensorflow to train an RNN model
based off a single LSTM cell, then saves the model to the path determined by `save_location.py`.
It uses Tensorflow's "Adam Optimizer" to learn the RNN model's parameters by minimizing a
certain loss function, which is a function of the actual classes of the training data and the
predicted classes of the training data according to the intermediate RNN model. It actually
feeds the data to the training process multiple times (for a more accurate model) in batches
(for efficiency).

`test.py` loads testing data from `load_data.py`, and restores the Tensorflow computation graph
from the paths determined by `save_location.py`.  It then tests the accuracy of the trained
model against the test data.

Currently, the loaded data is generated data, which is then randomly (but deterministically)
shuffled. You can change the `NUM_CLASSES` in `load_data.py` but by default it is `16`.  Each
"observation" corresponds to a number in `range(2**(NUM_CLASSES-1))`.  The "class" associated
to a number is the number of "1"s appearing in its `NUM_CLASSES-1` digit binary expansion.
RNN's are often used for cases where an observation of features occurs over time.  In this
case, the points in "time" for a given observation correspond to the individual digits in its
binary expansion.  And the "feature vector" associated with a given observation at a given
digit/"point-in-time" is just the single {0,1}-valued feature of whether that digit is 0 or 1.

## Why is it?

1. It's a complete working example of Tensorflow's RNN capabilities but only takes seconds to
  run, so you can start tweaking it without having to wait minutes or hours to figure out if you
  have basic dependencies installed correctly or other things wired up correctly.
2. It shows how loading data, training, and testing/predicting can be decoupled. `train.py`
  only depends on the resulting `X_train` and `y_train` from `load_data.py`, and doesn't
  depend on the process for how the data was loaded. `test.py` only depends on `X_test` and
  `y_test`. Most importantly, however, is that `test.py` doesn't depend on `train.py` at all.
