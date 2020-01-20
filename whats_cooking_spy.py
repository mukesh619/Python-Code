import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
 #%% 
 for dire,_,filenames in os.walk('whats-cooking'):
     for filename in filenames:
         print(os.path.join(dire,filename))
#%%

df_train=pd.read_json("whats-cooking/train.json")
df_test=pd.read_json("whats-cooking/test.json")
df_sub=pd.read_csv("whats-cooking/sample_submission.csv")
#%%

df_train.isna().sum()
df_train.info()
df_test.isna().sum()

df_train.cuisine.value_counts()
df_train.ingredients.value_counts()
#%%
lemmatizer = WordNetLemmatizer()
def preprocess(ingredients):
    ingredients_text = ' '.join(ingredients)
    ingredients_text = ingredients_text.lower()
    ingredients_text = ingredients_text.replace('-', ' ')
    words = []
    for word in ingredients_text.split():
        if re.findall('[0-9]', word): continue
        if len(word) <= 2: continue
        if '’' in word: continue
        word = lemmatizer.lemmatize(word)
        if len(word) > 0: words.append(word)
    return ' '.join(words)

df_train['cleaned_text']=df_train['ingredients'].apply(lambda x:preprocess(x))
df_test['cleaned_text']=df_test['ingredients'].apply(lambda x:preprocess(x))
#%%
X=df_train.drop(['ingredients','id','cuisine'],axis=1)
y=df_train['cuisine']
#%%
from sklearn.preprocessing import LabelEncoder

lab=LabelEncoder()
y=lab.fit_transform(y)
#%%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2)
#%%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence,text
#%%
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.20
BATCH_SIZE = 256
EPOCHS = 20

token=Tokenizer(num_words=1000)
'''
token.fit_on_texts(df_train['cleaned_text'].values)
word_index=token.word_index
len(word_index)

X=token.texts_to_sequences(df_train['cleaned_text'].values)
X=pad_sequences(X,maxlen=MAX_SEQUENCE_LENGTH)
print(X.shape)
'''
token.fit_on_texts(X_train['cleaned_text'].values)
vocab_size=len(token.word_index)+1 # Adding 1 because of reserved 0 index
X_train1=token.texts_to_sequences(X_train['cleaned_text'].values)
X_val=token.texts_to_sequences(X_test['cleaned_text'].values)

X_test1=token.texts_to_sequences(df_test['cleaned_text'].values)

X_train1=sequence.pad_sequences(X_train1,maxlen=MAX_SEQUENCE_LENGTH,padding="post")
X_val=sequence.pad_sequences(X_val,maxlen=MAX_SEQUENCE_LENGTH,padding="post")

X_test1=sequence.pad_sequences(X_test1,maxlen=MAX_SEQUENCE_LENGTH,padding="post")

#%%
from tensorflow.keras.activations import relu,softmax
from tensorflow.keras.layers import Conv1D,Dense,Dropout,Embedding,Flatten,GlobalMaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
#%%
model=Sequential()
model.add(Embedding(vocab_size,100,input_length=200))
model.add(Dropout(.2))

model.add(Conv1D(64,kernel_size=3,activation='relu',strides=1))
model.add(GlobalMaxPool1D())

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(.2))

model.add(Dense(128,activation='relu'))
model.add(Dropout(.2))

model.add(Dense(20,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.summary()
#%%
history=model.fit(X_train1,y_train,validation_data=(X_val,y_test),batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=1)
#%%

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show()

pred=np.argmax(model.predict(X_test1),axis=1)
y_pred=lab.inverse_transform(pred)

test_id = [doc for doc in df_test['id']]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('wats_cooking.csv', index=False)
