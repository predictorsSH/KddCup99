#GPU 할당
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


import tensorflow as tf
import pandas as pd

#data load
df=pd.read_csv('kddcup99_csv.csv')

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['label']=encoder.fit_transform(df['label'])

#features
x=df[df.columns.difference(['label'])]
#target
y=df['label']
x=pd.get_dummies(x)

# 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=False,random_state=1004)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

#minmax scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

len(df['label'].unique())

#하이퍼 파라미터 설정
learning_rate = 0.001
training_epochs = 3
batch_size = 30

#모델 생성
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(23, activation='softmax')
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

import numpy as np
model.fit(X_train, np.array(y_train),epochs=training_epochs,validation_split=0.1,batch_size=batch_size)
model.evaluate(X_test,np.array(y_test), verbose=2)
#148206/148206 - 10s - loss: 1.7095 - accuracy: 0.8600
