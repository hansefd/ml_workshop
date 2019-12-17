import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Read and check
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)

bank_dataframe = pd.read_csv(os.path.join('data', 'bank-full.csv'), sep=';')

print("Original shape: \n", bank_dataframe.shape)
print("Original head: \n", bank_dataframe.head())

# Drop duration columns as advised
bank_dataframe.drop(['duration'], axis=1, inplace=True)
print("Shape after drop: \n", bank_dataframe.shape)

# Column names
var_names = bank_dataframe.columns.tolist()
# Categorical vars
categs = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day', 'poutcome', 'y']
# Quantitative vars
quantit = [i for i in var_names if i not in categs]
print("Numerical values: \n", quantit)

# Format categorical vars
print("Marital categories: \n", bank_dataframe.marital.unique())
# Look and see
bank_dataframe['marital'].value_counts().plot(kind='bar')
plt.show()

# Factorization
marital_factorized, uniques = pd.factorize(bank_dataframe['marital'])
print("Marital factorized: \n", marital_factorized)
print("Unique factorized: \n", uniques)

# Map and modify
marital_map = bank_dataframe['marital'].map({'married': 1, 'single': 0, 'divorced': 0})
print("Marital map: \n", marital_map)

# Dummy variables
marital_dummy = pd.get_dummies(bank_dataframe['marital'])
print('Dummy: ', marital_dummy[:8])

marital = pd.get_dummies(bank_dataframe['marital'], drop_first=True)
print("Dummy cleared: ", marital[:8])

# Rest
job = pd.get_dummies(bank_dataframe['job'], drop_first=True)
education = pd.get_dummies(bank_dataframe['education'], drop_first=True)
default = pd.get_dummies(bank_dataframe['default'], drop_first=True)
housing = pd.get_dummies(bank_dataframe['housing'], drop_first=True)
loan = pd.get_dummies(bank_dataframe['loan'], drop_first=True)
contact = pd.get_dummies(bank_dataframe['contact'], drop_first=True)
month = pd.get_dummies(bank_dataframe['month'], drop_first=True)
day = pd.get_dummies(bank_dataframe['day'], drop_first=True)
poutcome = pd.get_dummies(bank_dataframe['poutcome'], drop_first=True)

# Encode output column
label = bank_dataframe['y'].map({'yes': 1, 'no': 0})

# Creating Quantitative DataFrame
bank_dataframe_quantit = bank_dataframe[quantit]
# Column names
df_names = bank_dataframe_quantit.keys().tolist()
# Normalizing, not quite how it should be
min_max_scaler = preprocessing.MinMaxScaler()
quantit_scaled = min_max_scaler.fit_transform(bank_dataframe_quantit)

print("Before scaling: \n", bank_dataframe_quantit)
print("After scaling: \n", quantit_scaled)

# Creating final data frame
df = pd.DataFrame(quantit_scaled)
df.columns = df_names
# Get final df
final_df = pd.concat([df,
                      job,
                      marital,
                      education,
                      default,
                      housing,
                      loan,
                      contact,
                      month,
                      day,
                      poutcome,
                      label], axis=1)

# Quick check
print("Final dataframe head: \n", final_df.head())
Y = final_df['y']

# Division to x and y
final_df.drop(['y'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(final_df, label, test_size=0.33, random_state=42, shuffle=True)
print("Training shape: \n", X_train.shape)
print("Testing shape: \n", X_test.shape)

# Input layer, hidden, Dense, num of neurons, activation fun, loss, optimizer-learning rate, fit args, kernel init
model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=54, input_shape=X_train.shape[1:], activation='elu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(np.array(X_train), np.array(y_train), epochs=10, verbose=2, validation_split=0.1, batch_size=128)

# Plots
plt.figure()
training_acc = np.array(history.history['acc'])
val_acc = np.array(history.history['val_acc'])
plt.plot(history.epoch, training_acc, 'g-')
plt.plot(history.epoch, val_acc, 'r-')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show(block=True)

training_loss = history.history['loss']
valid_loss = history.history['val_loss']
plt.plot(history.epoch, training_loss, 'b--')
plt.plot(history.epoch, valid_loss, 'r--')
plt.legend(['Training loss', 'Validation loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show(block=True)

[loss, accuracy] = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
print("Testing set accuracy: %.2f%%" % (100 * accuracy))
