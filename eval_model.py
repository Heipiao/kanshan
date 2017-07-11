import numpy  as np
import pandas as pd
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MultiLabelBinarizer

from tqdm import tqdm

BASE_PATH = "../data/ieee_zhihu_cup/"
EMBEDDING_DIM = 256
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 30
VALIDATION_SPLIT = 0.2

question = pd.read_csv(BASE_PATH+'question_train_set.txt',sep='\t',header=-1)
question_topic = pd.read_csv(BASE_PATH+'question_topic_train_set.txt',sep='\t',header=-1)
topic = pd.read_csv(BASE_PATH+'topic_info.txt',sep='\t',header=-1)

test_question = pd.read_csv(BASE_PATH+'question_eval_set.txt',sep='\t',header=-1)

question = question.astype(str)
topic = topic.astype(str)
test_question = test_question.astype(str)

labels = []
question_h = []
question_d = []
topic_h = []
topic_d = []
test_quesstion_h = []
test_quesstion_d = []

for l in question_topic[1].tolist():
    labels.append((l.split(",")))
for q1 in question[1].tolist():
    question_h.append((" ").join(q1.split(",")))
for q1 in question[3].tolist():
    question_d.append((" ").join(q1.split(",")))
for q1 in topic[2].tolist():
    topic_h.append((" ").join(q1.split(",")))
for q1 in topic[4].tolist():
    topic_d.append((" ").join(q1.split(",")))
    
for q1 in test_question[1].tolist():
    test_quesstion_h.append((" ").join(q1.split(",")))
for q1 in test_question[3].tolist():
    test_quesstion_d.append((" ").join(q1.split(",")))
    
# question_h = question_h[:2000]
# question_d = question_d[:2000]
# topic_h = topic_h[:2000]
# topic_d = topic_d[:2000]
# labels = labels[:2000]

label = np.array(labels)
mlp = MultiLabelBinarizer()
l = mlp.fit_transform(label)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(question_h+question_d+topic_h+topic_d)
sequences_question_h = tokenizer.texts_to_sequences(question_h)
sequences_question_d = tokenizer.texts_to_sequences(question_d)
sequences_topic_h = tokenizer.texts_to_sequences(topic_h)
sequences_topic_d = tokenizer.texts_to_sequences(topic_d)

sequences_test_quesstion_h = tokenizer.texts_to_sequences(test_quesstion_h)
sequences_test_quesstion_d = tokenizer.texts_to_sequences(test_quesstion_d)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_question_h = pad_sequences(sequences_question_h, maxlen=MAX_SEQUENCE_LENGTH)
data_question_d = pad_sequences(sequences_question_d, maxlen=MAX_SEQUENCE_LENGTH)
data_topic_h = pad_sequences(sequences_topic_h, maxlen=MAX_SEQUENCE_LENGTH)
data_topic_d = pad_sequences(sequences_topic_d, maxlen=MAX_SEQUENCE_LENGTH)

data_test_question_h = pad_sequences(sequences_test_quesstion_h, maxlen=MAX_SEQUENCE_LENGTH)
data_test_question_d = pad_sequences(sequences_test_quesstion_d, maxlen=MAX_SEQUENCE_LENGTH)

print(data_question_h.shape)
print(data_question_d.shape)
print(data_topic_h.shape)
print(data_topic_d.shape)

print(data_test_question_h.shape)
print(data_test_question_d.shape)
print(l.shape)

print('Preparing embedding matrix')

embeddings_index = {}
i = 0 
f = open(BASE_PATH+'char_embedding.txt',encoding="utf-8")
count = 0
for line in f:
    if(i>0):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    i = i + 1
f.close()

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## sample train/validation data
########################################
#np.random.seed(1234)
perm = np.random.permutation(len(data_question_h))
idx_train = perm[:int(len(data_question_h)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_question_h)*(1-VALIDATION_SPLIT)):]

data_question_h_train = data_question_h[idx_train]
data_question_h_val = data_question_h[idx_val]

data_question_d_train = data_question_d[idx_train]
data_question_d_val = data_question_d[idx_val]

label_train = l[idx_train]
label_val = l[idx_val]

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(3000, 4000)
rate_drop_lstm = 0.35 + np.random.rand() * 0.25
rate_drop_dense = 0.35 + np.random.rand() * 0.25

act = 'relu'
STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)

########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

merged = concatenate([x1, y1])
merged = BatchNormalization()(merged)
merged = Dropout(rate_drop_dense)(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = BatchNormalization()(merged)
merged = Dropout(rate_drop_dense)(merged)

preds = Dense(1999, activation='softmax')(merged)

model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
model.compile(loss='categorical_crossentropy',
        optimizer='nadam',
        metrics=['acc'])
#model.summary()
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_question_h_train,data_question_d_train], label_train, 
                 validation_data=([data_question_h_val, data_question_d_val], label_val), 
        epochs=5, batch_size=2048, shuffle=True, 
         callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

print('Start making the submission before fine-tuning')
preds = model.predict([data_test_question_h, data_test_question_d, batch_size=8192, verbose=1)

print(preds.shape)
p = np.argpartition(preds,[-5,-4,-3,-2,-1])[:,:-6:-1]
submission = pd.DataFrame(mlp.classes_[p])
submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False,header=False)