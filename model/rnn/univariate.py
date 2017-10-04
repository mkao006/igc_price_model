from __future__ import print_function
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def split_data(data, response_col, train_prop, valid_prop):
    ''' Function to split complete data into train, valid and prediction set.

    The data is split based on the assumption it is a time series.
    '''

    # Determine sequential end points
    series = data[response_col]
    total_length = len(series)
    model_prop = (train_prop + valid_prop)
    train_end_ind = int(total_length * train_prop)
    val_end_ind = int(total_length * model_prop)

    # Create the datasets
    train_data = series[:train_end_ind]
    valid_data = series[train_end_ind:val_end_ind]
    pred_data = series[val_end_ind:]

    return train_data, valid_data, pred_data


def batch_generator(data, batch_size, interval_size):
    ''' Returns a generator function of data for training and prediction.
    '''
    series_length = len(data)
    series = np.array(data, dtype=np.float32)
    num_slide = series_length - interval_size
    num_batch = num_slide / batch_size
    # truncated_slide = num_batch * batch_size
    truncated_sample_size = num_batch * batch_size
    truncated_series = series[-truncated_sample_size:]

    feature_matrix = np.zeros([truncated_sample_size, interval_size, 1])
    target = np.zeros([truncated_sample_size, 1])
    for slide in range(truncated_sample_size - interval_size):
        slide_end = slide + interval_size
        feature_matrix[slide, ] = truncated_series[slide:slide_end].reshape(
            1, interval_size, 1)
        target[slide] = truncated_series[slide_end]

    for batch in range(num_batch):
        batch_end = batch + batch_size
        batch_feature = feature_matrix[batch:batch_end, ]
        batch_target = target[batch:batch_end]
        yield batch_feature, batch_target


def scaler(batch):
    '''The function normalise the input for training, and also returns a
    denormaliser for reverse scaling.

    '''
    mean, var = tf.nn.moments(batch, [1], keep_dims=True)
    normalised_input = tf.div(tf.subtract(batch, mean), tf.sqrt(var))
    mean_squeezed = tf.squeeze(mean, 2)
    var_squeezed = tf.squeeze(var, 2)

    def reverse_scaler(scaled_prediction):
        original_scale = tf.add(tf.multiply(scaled_prediction,
                                            tf.sqrt(var_squeezed)),
                                mean_squeezed)
        return original_scale

    return normalised_input, reverse_scaler


# Read the data
goi_price = pd.read_csv('../../data/goi_price.csv')
response_name = 'GOI'

# Split the data
#
# NOTE (Michael): 50% for training, 47% for validating, and 3% for
#                 prediction. The 3% was chosen to approximate 180
#                 days for forecasting period.
train_data, test_data, pred_data = split_data(data=goi_price,
                                              response_col=response_name,
                                              train_prop=0.5, valid_prop=0.47)


# Initialise parameters
cell_size = 5
num_layers = 1
learning_rate = 0.001
epochs = 100
batch_size = 100
interval_size = 30

# Build the graph
graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    # Create input and target placeholder
    input_ = tf.placeholder(tf.float32, shape=[None, None, 1], name='input')
    target_ = tf.placeholder(tf.float32, shape=[None, 1], name='target')
    batch_size_ = tf.placeholder(tf.int32, shape=(), name='batch_size')

    # normalise the data and create the denormaliser
    scaled_input, denomarliser = scaler(input_)

    # Define the LSTM
    lstm = tf.contrib.rnn.MultiRNNCell(
        cells=[tf.contrib.rnn.BasicLSTMCell(num_units=cell_size)
               for l in range(num_layers)])
    initial_state = lstm.zero_state(batch_size=batch_size_,
                                    dtype=tf.float32)
    output, final_state = tf.nn.dynamic_rnn(cell=lstm,
                                            inputs=scaled_input,
                                            initial_state=initial_state)

    # Define the fully connected layer
    normalised_pred = tf.contrib.layers.fully_connected(inputs=output[:, -1],
                                                        num_outputs=1,
                                                        activation_fn=None)

    # Revert the prediction to the original scale
    pred = denomarliser(normalised_pred)

    # Define training operation
    loss = tf.losses.mean_squared_error(labels=target_,
                                        predictions=pred)

    optimiser = (tf.train.AdamOptimizer(learning_rate=learning_rate)
                 .minimize(loss=loss))


# Train the LSTM
#
# NOTE (Michael): The large difference between the train and test loss
#                 suggests the model is over-fitting.
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for e in range(epochs):
        train_generator = batch_generator(
            data=train_data, interval_size=interval_size,
            batch_size=batch_size)
        train_state = sess.run(initial_state,
                               feed_dict={batch_size_: batch_size})
        for train_batch, (train_input, train_target) in enumerate(train_generator):
            train_loss, train_pred, _, train_state = sess.run(
                [loss, pred, optimiser, final_state],
                feed_dict={input_: train_input,
                           target_: train_target,
                           batch_size_: batch_size,
                           initial_state: train_state})

        test_state = train_state
        test_generator = batch_generator(
            data=test_data, interval_size=interval_size,
            batch_size=batch_size)

        for test_batch, (test_input, test_target) in enumerate(test_generator):
            test_loss, test_pred, test_state = sess.run(
                [loss, pred, final_state],
                feed_dict={input_: test_input,
                           target_: test_target,
                           batch_size_: batch_size,
                           initial_state: test_state})

        print('Epoch: {}/{}'.format(e + 1, epochs),
              'Train loss: {:.3f}'.format(train_loss),
              'Test loss: {:.3f}'.format(test_loss))

    if not os.path.exists('checkpoints/'):
        os.mkdir('checkpoints')

    saver.save(sess, 'checkpoints/univariate/univariate.ckpt')

# Make rolling prediction
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints/univariate'))
    prediction = np.empty(0)
    initial_input = np.array(pred_data.iloc[:interval_size])
    num_slide = len(pred_data) - interval_size
    for slide in range(num_slide):
        current_input = np.concatenate(
            [initial_input, prediction])[-interval_size:].reshape(1, -1, 1)
        current_pred = sess.run(pred,
                                feed_dict={input_: current_input,
                                           batch_size_: 1})
        prediction = np.append(prediction, current_pred)


# Plot the prediction
#
# NOTE (Michael): The prediction is non-sensical, further extremely
#                 unstable. However, this is what one would expect to
#                 make a rolling forecast of any model.
prediction_df = pd.DataFrame({'pred': prediction.reshape(1, -1).tolist()[0],
                              'actual': pred_data[interval_size:].tolist()})

plt.plot(prediction_df['pred'])
plt.plot(prediction_df['actual'])
plt.show()
