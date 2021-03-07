#GPU 할당
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

#data laod
import pandas as pd
df=pd.read_csv('kddcup99_csv.csv')

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['label']=encoder.fit_transform(df['label'])
print(df['label'])

#features
x=df[df.columns.difference(['label'])]
#target
y=df['label']

#모델 학습때 'categorical_crossentropy' 를 loss로 쓰기 위해 다시 변환
from tensorflow.keras.utils import to_categorical
y=to_categorical(y)


# 독립변수 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x=pd.get_dummies(x)
x=scaler.fit_transform(x)

# 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=False)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

#하이퍼 파라미터 설정
learning_rate = 0.001
training_epochs = 3
batch_size = 25

#CNN 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=16, kernel_size=10, activation='relu',padding="SAME" ,input_shape=(118,1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=32, kernel_size=10, activation="relu",padding="SAME" ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(23,activation='softmax')

])

#모델 컴파일
model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#학습시키기 위해서 데이터 shape 맞춰주기
X_train=X_train.reshape((-1,118,1))
model.fit(X_train,y_train, epochs=training_epochs, validation_split=0.1, verbose=2,batch_size=batch_size)


#테스트 데이터로 평가
X_test.reshape((-1,118,1))
model.evaluate(X_test,y_test,verbose=2)
#[loss,accuracy]=[3.0902209281921387, 0.8582648634910583]
