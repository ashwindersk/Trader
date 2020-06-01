import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import csv
import itertools




#Loading data from files
def load_data(midprices_fname, transactions_fname):
    midprices_x = []
    midprices_y = []
        
    with open(midprices_fname, 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter = ",")
            for row in plots:
                    midprices_x.append(float(row[0]))
                    midprices_y.append(float(row[1]))
                    
    transactions_x = []
    transactions_y = []
    with open(transactions_fname,'r') as csvfile:
            plots = csv.reader(csvfile, delimiter = ',')
            for row in plots:
                    transactions_x.append(float(row[0]))
                    transactions_y.append(float(row[1]))
        
    
    return (midprices_x, midprices_y),(transactions_x, transactions_y)



def normalise_data(data):
    data = tf.keras.utils.normalize(data)
    return data[0]

def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)

def create_time_steps(length):
  return list(range(-length, 0))

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  plt.show()



def main():
    #midprice_data, transaction_data = [], []
    #for i in range(10)
    midprice_data, transaction_data = load_data(f'../BSE/midprice.csv', f'../BSE/transactions.csv')
    
    midprice_y = np.array(midprice_data[1])
    midprice_x = np.array(midprice_data[0])

    transaction_y = np.array(transaction_data[1])
    transaction_x = np.array(transaction_data[0])
    
    
    TRAIN_SPLIT = 1500
    
    
    
    transactions = normalise_data(transaction_data[1])
    midprice_y = normalise_data(midprice_data[1])
    print(midprice_y)    
    univariate_past_history = 40
    univariate_future_target = 0
    

    x_train_uni, y_train_uni = univariate_data(midprice_y, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
    x_val_uni, y_val_uni = univariate_data(midprice_y, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)
    
    
    BATCH_SIZE = 256
    BUFFER_SIZE = 10000

    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
    
    simple_lstm_model = tf.keras.models.Sequential([
                        tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
                        tf.keras.layers.Dense(1)
                    ])

    simple_lstm_model.compile(optimizer='adam', loss='mae')
    
    EVALUATION_INTERVAL = 300
    EPOCHS = 10

    simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)
    
    for x, y in val_univariate.take(3):
        plot = show_plot([x[0].numpy(), y[0].numpy(),
                    simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
    plot.show()
    
    #midprice_data.append(midprice)
    #transaction_data.append(transaction)

    

    
    
if __name__ == '__main__':
    main()

