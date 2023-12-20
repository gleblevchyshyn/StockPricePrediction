import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def create_sequences(data, seq_size=1):
    x = np.zeros(shape=(len(data) - seq_size, seq_size))

    for i in range(seq_size):
        x[:, i] = data[i: len(data) - seq_size + i].reshape(-1)

    y = data[seq_size:]

    print(x.shape)
    print(y.shape)
    return x, y


sequence_size = 1
train_df = pd.read_csv('train.csv')

train_size = int(0.8 * len(train_df))

train = train_df.iloc[:train_size]
val = train_df.iloc[train_size:]

min_max_scaler = MinMaxScaler()

x_train, y_train = create_sequences(train['Close'].values.reshape((-1, 1)), sequence_size)
x_val, y_val = create_sequences(val['Close'].values.reshape((-1, 1)), sequence_size)

scaled_train = min_max_scaler.fit_transform(x_train)
scaled_val = min_max_scaler.transform(x_val)


model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(1, sequence_size), return_sequences=True),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(8, activation='elu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, verbose=1)
history = model.fit(scaled_train, y_train, validation_data=[scaled_val, y_val], verbose=1, epochs=100,
                    callbacks=[early_stopping])

pd.DataFrame(history.history).plot()
plt.savefig('training_history.png')
plt.show()


plt.title('Model Predictions')
plt.plot(y_train, label='Train Data')
plt.plot(model.predict(scaled_train), label='Predicted Train')
plt.plot(range(len(y_train), len(y_train) + len(y_val)), y_val, label='Validation Data')
plt.plot(range(len(y_train), len(y_train) + len(y_val)), model.predict(scaled_val), label='Predicted Validation')

plt.legend()

plt.savefig('predictions.png')
plt.show()
