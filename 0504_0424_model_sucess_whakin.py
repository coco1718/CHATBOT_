#====  파일나눌때

# 이미 설치돼있음 !pip install pyyaml h5py  # Required to save models in HDF5 format

# https://www.youtube.com/watch?v=WTul6LIjIBA 참고
# pip install gTTS
# pip install playsound==1.2.2
#cmd명령어
#C:\AISpeaker>python -m venv myenv
#C:\AISpeaker>.\myenv\Scripts\activate
#(myenv) C:\AISpeaker>pip install gTTS
#(myenv) C:\AISpeaker>pip install playsound==1.2.2

#마이크설치
#(myenv) C:\AISpeaker>pip install SpeechRecognition
#(myenv) C:\AISpeaker>pip install PyAudio

import pandas as pd
import urllib.request
import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import re




import os

import tensorflow as tf
from tensorflow import keras

#print(tf.version.VERSION)


#====  파일나눌때

# 이미 설치돼있음 !pip install pyyaml h5py  # Required to save models in HDF5 format

import pandas as pd
import urllib.request
import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import re
from tensorflow import keras

import os

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# tf.__version__

class PositionalEncoding(tf.keras.layers.Layer):
  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)

    # 배열의 짝수 인덱스(2i)에는 사인 함수 적용
    sines = tf.math.sin(angle_rads[:, 0::2])

    # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
    cosines = tf.math.cos(angle_rads[:, 1::2])

    angle_rads = np.zeros(angle_rads.shape)
    angle_rads[:, 0::2] = sines
    angle_rads[:, 1::2] = cosines
    pos_encoding = tf.constant(angle_rads)
    pos_encoding = pos_encoding[tf.newaxis, ...]

    print(pos_encoding.shape)
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

sample_pos_encoding = PositionalEncoding(50, 128)

plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 128))
plt.ylabel('Position')
plt.colorbar()
plt.show()

def scaled_dot_product_attention(query, key, value, mask):
  # query 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
  # key 크기 : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
  # value 크기 : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
  # padding_mask : (batch_size, 1, 1, key의 문장 길이)

  # Q와 K의 곱. 어텐션 스코어 행렬.
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # 스케일링
  # dk의 루트값으로 나눠준다.
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # 마스킹. 어텐션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
  # 매우 작은 값이므로 소프트맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 된다.
  if mask is not None:
    logits += (mask * -1e9)

  # 소프트맥스 함수는 마지막 차원인 key의 문장 길이 방향으로 수행된다.
  # attention weight : (batch_size, num_heads, query의 문장 길이, key의 문장 길이)
  attention_weights = tf.nn.softmax(logits, axis=-1)

  # output : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
  output = tf.matmul(attention_weights, value)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    # d_model을 num_heads로 나눈 값.
    # 논문 기준 : 64
    self.depth = d_model // self.num_heads

    # WQ, WK, WV에 해당하는 밀집층 정의
    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)

    # WO에 해당하는 밀집층 정의
    self.dense = tf.keras.layers.Dense(units=d_model)

  # num_heads 개수만큼 q, k, v를 split하는 함수
  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # 1. WQ, WK, WV에 해당하는 밀집층 지나기
    # q : (batch_size, query의 문장 길이, d_model)
    # k : (batch_size, key의 문장 길이, d_model)
    # v : (batch_size, value의 문장 길이, d_model)
    # 참고) 인코더(k, v)-디코더(q) 어텐션에서는 query 길이와 key, value의 길이는 다를 수 있다.
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # 2. 헤드 나누기
    # q : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    # k : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
    # v : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    # 3. 스케일드 닷 프로덕트 어텐션. 앞서 구현한 함수 사용.
    # (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
    # (batch_size, query의 문장 길이, num_heads, d_model/num_heads)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # 4. 헤드 연결(concatenate)하기
    # (batch_size, query의 문장 길이, d_model)
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # 5. WO에 해당하는 밀집층 지나기
    # (batch_size, query의 문장 길이, d_model)
    outputs = self.dense(concat_attention)

    return outputs


def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  # (batch_size, 1, 1, key의 문장 길이)
  return mask[:, tf.newaxis, tf.newaxis, :]


def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

  # 인코더는 패딩 마스크 사용
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  # 멀티-헤드 어텐션 (첫번째 서브층 / 셀프 어텐션)
  attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
          'mask': padding_mask # 패딩 마스크 사용
      })

  # 드롭아웃 + 잔차 연결과 층 정규화
  attention = tf.keras.layers.Dropout(rate=dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

  # 포지션 와이즈 피드 포워드 신경망 (두번째 서브층)
  outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)

  # 드롭아웃 + 잔차 연결과 층 정규화
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            name="encoder"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")

  # 인코더는 패딩 마스크 사용
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  # 포지셔널 인코딩 + 드롭아웃
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  # 인코더를 num_layers개 쌓기
  for i in range(num_layers):
    outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
        dropout=dropout, name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)


# 디코더의 첫번째 서브층(sublayer)에서 미래 토큰을 Mask하는 함수
def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x) # 패딩 마스크도 포함
  return tf.maximum(look_ahead_mask, padding_mask)



def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

  # 디코더는 룩어헤드 마스크(첫번째 서브층)와 패딩 마스크(두번째 서브층) 둘 다 사용.
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  # 멀티-헤드 어텐션 (첫번째 서브층 / 마스크드 셀프 어텐션)
  attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
          'mask': look_ahead_mask # 룩어헤드 마스크
      })

  # 잔차 연결과 층 정규화
  attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

  # 멀티-헤드 어텐션 (두번째 서브층 / 디코더-인코더 어텐션)
  attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1, 'key': enc_outputs, 'value': enc_outputs, # Q != K = V
          'mask': padding_mask # 패딩 마스크
      })

  # 드롭아웃 + 잔차 연결과 층 정규화
  attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
  attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

  # 포지션 와이즈 피드 포워드 신경망 (세번째 서브층)
  outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention2)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)

  # 드롭아웃 + 잔차 연결과 층 정규화
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)


def decoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            name='decoder'):
  inputs = tf.keras.Input(shape=(None,), name='inputs')
  enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')

  # 디코더는 룩어헤드 마스크(첫번째 서브층)와 패딩 마스크(두번째 서브층) 둘 다 사용.
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  # 포지셔널 인코딩 + 드롭아웃
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  # 디코더를 num_layers개 쌓기
  for i in range(num_layers):
    outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
        dropout=dropout, name='decoder_layer_{}'.format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)


def transformer(vocab_size, num_layers, dff,
                d_model, num_heads, dropout,
                name="transformer"):

  # 인코더의 입력
  inputs = tf.keras.Input(shape=(None,), name="inputs")

  # 디코더의 입력
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")


  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)

  # 디코더의 룩어헤드 마스크(첫번째 서브층)
  look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask, output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)

  # 디코더의 패딩 마스크(두번째 서브층)
  dec_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

  # 인코더의 출력은 enc_outputs. 디코더로 전달된다.
  enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
      d_model=d_model, num_heads=num_heads, dropout=dropout,
  )(inputs=[inputs, enc_padding_mask]) # 인코더의 입력은 입력 문장과 패딩 마스크

  # 디코더의 출력은 dec_outputs. 출력층으로 전달된다.
  dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
      d_model=d_model, num_heads=num_heads, dropout=dropout,
  )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  # 다음 단어 예측을 위한 출력층
  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)



small_transformer = transformer(
    vocab_size = 9000,
    num_layers = 4,
    dff = 512,
    d_model = 128,
    num_heads = 4,
    dropout = 0.3,
    name="small_transformer")



def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32) # tf 2.12 버전에서 동작하기 위해 신규 추가
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



sample_learning_rate = CustomSchedule(d_model=128)

plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")


import pandas as pd
import urllib.request
import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import re


# urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")

train_data = pd.read_csv('ChatBotData.csv')
train_data.head()

print('챗봇 샘플의 개수 :', len(train_data))


questions = []
for sentence in train_data['Q']:
    # 구두점에 대해서 띄어쓰기
    # ex) 12시 땡! -> 12시 땡 !
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    questions.append(sentence)

answers = []
for sentence in train_data['A']:
    # 구두점에 대해서 띄어쓰기
    # ex) 12시 땡! -> 12시 땡 !
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    answers.append(sentence)

len(questions)

print(questions[:10])
print(answers[:10])

# 서브워드텍스트인코더를 사용하여 질문과 답변을 모두 포함한 단어 집합(Vocabulary) 생성
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13)

# 시작 토큰과 종료 토큰에 대한 정수 부여.
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# 시작 토큰과 종료 토큰을 고려하여 단어 집합의 크기를 + 2
VOCAB_SIZE = tokenizer.vocab_size + 2


print('시작 토큰 번호 :',START_TOKEN)
print('종료 토큰 번호 :',END_TOKEN)
print('단어 집합의 크기 :',VOCAB_SIZE)

# 서브워드텍스트인코더 토크나이저의 .encode()를 사용하여 텍스트 시퀀스를 정수 시퀀스로 변환.
print('Tokenized sample question: {}'.format(tokenizer.encode(questions[20])))

# 서브워드텍스트인코더 토크나이저의 .encode()와 decode() 테스트해보기

# 임의의 입력 문장을 sample_string에 저장
sample_string = questions[20]

# encode() : 텍스트 시퀀스 --> 정수 시퀀스
tokenized_string = tokenizer.encode(sample_string)
print ('정수 인코딩 후의 문장 {}'.format(tokenized_string))

# decode() : 정수 시퀀스 --> 텍스트 시퀀스
original_string = tokenizer.decode(tokenized_string)
print ('기존 문장: {}'.format(original_string))


# 각 정수는 각 단어와 어떻게 mapping되는지 병렬로 출력
# 서브워드텍스트인코더는 의미있는 단위의 서브워드로 토크나이징한다. 띄어쓰기 단위 X 형태소 분석 단위 X
for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))

# 최대 길이를 40으로 정의
MAX_LENGTH = 40


# 토큰화 / 정수 인코딩 / 시작 토큰과 종료 토큰 추가 / 패딩
def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

        tokenized_inputs.append(sentence1)
        tokenized_outputs.append(sentence2)

    # 패딩
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs


questions, answers = tokenize_and_filter(questions, answers)

print('질문 데이터의 크기(shape) :', questions.shape)
print('답변 데이터의 크기(shape) :', answers.shape)

# 0번째 샘플을 임의로 출력
print(questions[0])
print(answers[0])


# 텐서플로우 dataset을 이용하여 셔플(shuffle)을 수행하되, 배치 크기로 데이터를 묶는다.
# 또한 이 과정에서 교사 강요(teacher forcing)을 사용하기 위해서 디코더의 입력과 실제값 시퀀스를 구성한다.
BATCH_SIZE = 64
BUFFER_SIZE = 20000

# 디코더의 실제값 시퀀스에서는 시작 토큰을 제거해야 한다.
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1] # 디코더의 입력. 마지막 패딩 토큰이 제거된다.
    },
    {
        'outputs': answers[:, 1:]  # 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다.
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)



# 임의의 샘플에 대해서 [:, :-1]과 [:, 1:]이 어떤 의미를 가지는지 테스트해본다.
print(answers[0]) # 기존 샘플
print(answers[:1][:, :-1]) # 마지막 패딩 토큰 제거하면서 길이가 39가 된다.
print(answers[:1][:, 1:]) # 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다. 길이는 역시 39가 된다.


tf.keras.backend.clear_session()

# Hyper-parameters
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)


MAX_LENGTH = 40

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)




def accuracy(y_true, y_pred):
  # ensure labels have shape (batch_size, MAX_LENGTH - 1)
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

# 위 그대로 퍼옴 compile까지



import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr

model.load_weights('./checkpoints/my_checkpoint')
#model = tf.keras.models.load_model('./saved_model/my_model')
#model.summary()


#1 모델글어오기
# Recreate the exact same model, including its weights and the optimizer
#model = tf.keras.models.load_model('my_chat.h5')
#model.summary()


#2 모델을 다시 끌어온다
# Create a basic model instance
#model = create_model()
#checkpoint_path = "training_1/cp.ckpt"
#model.load_weights(checkpoint_path)







# 음성인식 (듣기)
def listen(recognizer, audio):
    try:
        sentence = recognizer.recognize_google(audio, language='ko')
        print('[나]' + sentence)
        answer(sentence)


    except sr.UnknownValueError:
        print('인식 실패')  # 음성인식실패
    except sr.RequestError as e:  # 네트워크오류
        print('요청실패 : {0}'.format(e))  # API Key오류, 네트워크단절 등


# 질문sentence에 대한 맞는답변찾기
def answer(sentence):

        sentence = preprocess_sentence(sentence)
        sentence = tf.expand_dims(
            START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

        output = tf.expand_dims(START_TOKEN, 0)

        # 디코더의 예측 시작
        for i in range(MAX_LENGTH):
            predictions = model(inputs=[sentence, output], training=False)

            # 현재(마지막) 시점의 예측 단어를 받아온다.
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
            if tf.equal(predicted_id, END_TOKEN[0]):
                break

            # 마지막 시점의 예측 단어를 출력에 연결한다.
            # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
            output = tf.concat([output, predicted_id], axis=-1)
         #  return predict(tf.squeeze(output, axis=0))
        predict(tf.squeeze(output, axis=0))


# 질문에서 쓸데없는구두점제거
def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence


# 답변예측

# answer의 return predict(tf.squeeze(output, axis=0)) 이 값이 prediction에
def predict(sentence):
    prediction = sentence  # ex: answer(바보야)  =>> sentence = tf.squeeze(output, axis=0) , predict(바보야) 찾기

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    speak(predicted_sentence)
    # print('Input: {}'.format(sentence))
    # print('Output: {}'.format(predicted_sentence))

 #   return speak(predicted_sentence) 참고:굳이return안씀,함수부를땐, https://smart-factory-lee-joon-ho.tistory.com/357



# 소리내어읽기
def speak(text):

    print('[인공지능]' + text)
    file_name = 'voice.mp3'
    tts = gTTS(text=text, lang='ko')
    tts.save(file_name)
    playsound(file_name)
    if os.path.exists(file_name):  # voice.mp3 파일삭제
        os.remove(file_name)


# def 함수끝에 결과값을 return하기도 하고, speak(return값) 함수를 부르기도 함

r = sr.Recognizer()
m = sr.Microphone()
speak('무엇을 도와줄까?')


# 백그라운드에서 듣고 있음 https://smart-factory-lee-joon-ho.tistory.com/357
stop_listening = r.listen_in_background(m, listen)
#stop_listening(wait_for_stop=False) # 더 이상 듣지 않음,듣기를듣지마란뜻

while True:  #무한루프
   time.sleep(0.1)


# 성공 C:\Users\user\anaconda3\python.exe C:\Users\user\PycharmProjects\pythonProject\0504_0424_model_sucess_whakin.py
# (1, 50, 128)
# (1, 9000, 128)
# (1, 9000, 128)
# 챗봇 샘플의 개수 : 27849
# ['이번 월급도 부모님께 전부 드렸어 .', '부모님은 내 월급을 다 본인들이 관리하기 원하셔 .', '내가 버는 돈의 관리를 전적으로 내가 한다고 말씀드려야겠어 .', '최근에 많이 바쁘게 살고 있어 .', '오늘 유난히 시간이 너무 빨리 흐른 것 같아 .', '급한 약속이 생겨서 집에 늦게 갈 거 같아 .', '펀드매니저하는 친구 아들한테 내 보험금 좀 맡겨볼까 봐 .', '젊을 때는 아내가 돈 쓰는 것 때문에 많이 싸웠는데 이제는 다 부질없는 것 같아 .', '일흔이 되도록 나는 늘 더 젊어지고 건강해지고 싶었어 .', '텔레비전에 나오는 연예인들은 나와 동갑인데도 건강하고 활력이 넘쳐 .']
# ['부모님께 월급을 다 드렸군요 .  부모님께서 달라고 하셨나요 ?', '부모님말고 사용자님이 원하는 것은 무엇인가요 ?', '부모님이 사용자님의 의견을 잘 수용해주면 좋겠네요 .', '최근 많이 바쁘시군요 .  더 자세히 말씀해주실 수 있나요 ?', '시간이 빨리 흐르다니 오늘 하루는 어땠나요 ?', '집에 늦게 가시는군요 .  무슨 일이신가요 ?', '보험금을 친구 아들에게 맡기시려고 하는군요 .', '아내 분께서 돈을 쓰시는 것에 관여할 마음이 줄어드셨군요 .', '건강에 관심이 많으셨군요 .', '연예인들의 모습이 많이 부러우시군요 .']
# 시작 토큰 번호 : [8360]
# 종료 토큰 번호 : [8361]
# 단어 집합의 크기 : 8362
# Tokenized sample question: [6829, 2833, 13, 269, 2523, 117, 1]
# 정수 인코딩 후의 문장 [6829, 2833, 13, 269, 2523, 117, 1]
# 기존 문장: 나에게는 이루고 싶은 꿈이 있어 .
# 6829 ----> 나에게는
# 2833 ----> 이루
# 13 ----> 고
# 269 ----> 싶은
# 2523 ----> 꿈이
# 117 ----> 있어
# 1 ---->  .
# 질문 데이터의 크기(shape) : (27849, 40)
# 답변 데이터의 크기(shape) : (27849, 40)
# [8360  405 2476   14  511 3027 3437   47    1 8361    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0]
# [8360  511 3041   61 3437   70    2 2429  353 3959    3 8361    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0]
# [8360  511 3041   61 3437   70    2 2429  353 3959    3 8361    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0]
# [[8360  511 3041   61 3437   70    2 2429  353 3959    3 8361    0    0
#      0    0    0    0    0    0    0    0    0    0    0    0    0    0
#      0    0    0    0    0    0    0    0    0    0    0]]
# [[ 511 3041   61 3437   70    2 2429  353 3959    3 8361    0    0    0
#      0    0    0    0    0    0    0    0    0    0    0    0    0    0
#      0    0    0    0    0    0    0    0    0    0    0]]
# (1, 8362, 256)
# (1, 8362, 256)
# [인공지능]무엇을 도와줄까?
# [나]진짜로 도와줄 거야
# [인공지능]네 .  좋은 생각이세요 .
# [나]동문서답 참 좋아 동문서답을 내 적성에 딱 맞는 거 같아
# [인공지능]대화의 눈높이가 맞는 사람 만나세요 .
# [나]아 진짜 정곡을 찌르는 해 왜 이렇게 싸가지가 없냐 넌
# [인공지능]생각 정리에 도움이 됐다면 좋겠어요 !
# [나]정말 도움이 안 되고 화병 당했어 화병에 도움이 될 거 같아 병을 더 증폭시키는데 도움이 될 거 같아 응 말이지 그래서 듣던 말이지
# [인공지능]틀니 관리가 귀찮으시군요 .  조금 더 자세히 말씀해 주시겠어요 ?
# [나]아직 틀니 안 했어 아직은 튼튼해 내 이빨 너도 씹어 먹을 수 있어 랄랄라 랄랄라 랄랄라 랄라 랄라 랄라 랄라 랄라 랄라 랄라 랄라 랄라 야 좀 빨리 달리면 안 되냐 한 바퀴 도는데 그렇게 오래 걸려 그거 100미터 벌초 나와 100미터 한 5,000원 뛰어야지 되는 거 아니야 인공지능은 3초 3초가 뭐야 0.5 챙겨야 되는 거 아니야 됩니까
# [인공지능]올드에는 꼭 좋은 결과가 있었으면 좋겠어요 .
# [나]월드에는 더 꼭 좋은 결과가 있었으면 좋구요 좋겠어요 월드에는 올드
# [인공지능]정말 좋은 기대네요 .  앞으로는 열심히 일만 남았길 바랄게요 .
# [나]아 요새는 월드가 유행이야 레트로 레트로가 유행이야 날이야 지난 시절 다시 올 수 없나 그만 넌 어떻게 생각해 보는 상당히 덥다
# [인공지능]그런 ���러볼 수 있는 분이 돌아가면 어떤 방법이 있을까요 ?
# [나]로 볼 수 있는 분이 돌아가면 어떤 방법이 많이 애매모호한 단어가 상당히 기분이 좋아 애매모 내 머릿속에 아주 애매모호하다 m m o m m o m m o 우리 달려서 첫 마디만 해 보자
# [인공지능]만남 때문에 중요한 부분이에요 .  어떻게 생각하세요 ?
# [나]어떻게 생각하세요 어떡해 어떡해 어떡해 어떡해
# [인공지능]감정을 보세요 .
# [나]감정을 보세요 넌 감정이 보이니
# [인공지능]저도 그런 사람 만나고 싶어요 .
# [나]나도 그런 사람 만나고 싶어요
# [인공지능]그런 사람 신경쓰지 마세요 .
# [나]우리 천 마리 만해요 첫마디
# [인공지능]전하려면 뭐가 가장 먼저 해야 할까요 ?
# [나]전화를 하면 가장 먼저 예의를 지켜야지요 싸가지를 지켜야지요
# [인공지능]분위기도 맛도 말문이 막줄 수 있기를 바라요 .
# [나]아 철자 다 틀리고 아주 개판 이십니다
# [인공지능]조심히 오세요 .
# [나]조심히 어디를 오라고요 깜빡깜빡 하시는군요 깜빡 첫 마디만 합시다 더위 먹었나이 보세요
# [인공지능]메모하는 습관을 들여보세요 .
# [나]딸이나 랄랄랄라 랄랄랄라 랄랄랄라 랄랄랄라 3,000달러 하지만 3천단어 3,000원 하자
# [인공지능]그렇게 다짐을 하셨군요 .  지금의 마음먹은 무엇일까요 ?
# [나]야 근데 너 왜 이렇게 철자가 다 틀려 대체 누가 가르친 거야 이거
# [인공지능]그 없다고 너무 무거운 짐을 건강에 안 돼요 .
# [나]으르르르 우리 역대급으로 길게 해 볼까 만리장성을 써 보자
# [인공지능]스트레스가 심대한가봐요 .
# [나]심대한 가봐 11 용어도 안 해 나 너 로봇 로보트 로보트 다 틀리고 아주 괜찮습니다
# [인공지능]저는 자장면이요 .
# [나]나는 짬뽕이야
# [인공지능]저는 계속 이어주는 위로봇입니다 .
# [나]이어 주는 위 로봇 좋아요 스스로 학습 하게끔 해준다
# [인공지능] 반복 일 순 없으시지만 먼저 마인배역해서 더 좋은 곳이 있을까요 ?
# [나]다른 거 보시고 스스로학습 해야 돼 스스로 업데이트 하게 해야 돼 데이터를 업데이트 하기 위하여 오게 2단계 3단계 오픈소스 풀어라 우리 첫날이네요 첫마디
# [인공지능]신경쓰 부분이나 테서 일들이 있군요 .  속상하고 몸이 좋군요 .
# [나]속상하고 몸이 좋군요 나중에 내가 여기까지 올라오기까지 힘들었으니까 삼촌 날까지 하겠어 노래 다 조정 해 주겠어 오픈소스잡 푸르지오 스스로 데이트라고 업데이트 하게 하는게 2단계 입력한 거 푸는 거 쉬워 이건 정말 단속 로봇 입력한 거 뵙는 거니까내는 거니까 스스로 데이트 업데이트 하게 한게 삼척
# [인공지능]신상으로 바꾸세요 .  대인관계 이댄 활동을 통해 수용해주면 만남이 수술 덜어 배려가 있을까요 ?
# [나]사람이 그랬던가 스스로 데이터 옮기는 2단계 바로바로 신성으로 바꾸세요 조용히 해 주면 어떻게 하면 이렇게 다 틀릴 수 있는 거지
# [인공지능]그럼 바로채라도 있을까요 ?





#Exception in thread Thread-1 (threaded_listen):
#Traceback (most recent call last):
#  File "C:\Users\user\anaconda3\lib\threading.py", line 1016, in _bootstrap_inner
#    self.run()
#  File "C:\Users\user\anaconda3\lib\threading.py", line 953, in run
#    self._target(*self._args, **self._kwargs)
#  File "C:\Users\user\anaconda3\lib\site-packages\speech_recognition\__init__.py", line 568, in threaded_listen
#    if running[0]: callback(self, audio)
#  File "C:\Users\user\PycharmProjects\pythonProject\0504_0424_model_sucess_whakin.py", line 630, in listen
#    sentence = recognizer.recognize_google(audio, language='ko')
#  File "C:\Users\user\anaconda3\lib\site-packages\speech_recognition\__init__.py", line 708, in recognize_google
#    response = urlopen(request, timeout=self.operation_timeout)
#  File "C:\Users\user\anaconda3\lib\urllib\request.py", line 216, in urlopen
#    return opener.open(url, data, timeout)
#  File "C:\Users\user\anaconda3\lib\urllib\request.py", line 519, in open
#    response = self._open(req, data)
#  File "C:\Users\user\anaconda3\lib\urllib\request.py", line 536, in _open
#    result = self._call_chain(self.handle_open, protocol, protocol +
#  File "C:\Users\user\anaconda3\lib\urllib\request.py", line 496, in _call_chain
#    result = func(*args)
#  File "C:\Users\user\anaconda3\lib\urllib\request.py", line 1377, in http_open
#    return self.do_open(http.client.HTTPConnection, req)
#  File "C:\Users\user\anaconda3\lib\urllib\request.py", line 1352, in do_open
#    r = h.getresponse()
#  File "C:\Users\user\anaconda3\lib\http\client.py", line 1374, in getresponse
#    response.begin()
#  File "C:\Users\user\anaconda3\lib\http\client.py", line 318, in begin
#    version, status, reason = self._read_status()
#  File "C:\Users\user\anaconda3\lib\http\client.py", line 279, in _read_status
#    line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")
#  File "C:\Users\user\anaconda3\lib\socket.py", line 705, in readinto
#    return self._sock.recv_into(b)
# ConnectionAbortedError: [WinError 10053] 현재 연결은 사용자의 호스트 시스템의 소프트웨어의 의해 중단되었습니다