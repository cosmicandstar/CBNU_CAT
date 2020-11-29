#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
from tensorflow.keras.layers import Dense, Concatenate, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adamax


# In[2]:


import pandas as pd
import numpy as np 


# In[3]:


file_path= "Data/"
ratings_row = pd.read_csv(file_path + "ratings_good.csv")
ratings = pd.DataFrame(ratings_row, columns=["id","lecture","rating"])

mbti_row = pd.read_csv(file_path + "user_mbti.csv")
mbti = pd.DataFrame(mbti_row, columns=["id","mbti"])


# In[4]:


def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))    


# In[5]:


def cat_NeuralMF_begin(ratings, mbti, lec_num):
    ratings = pd.merge(ratings, mbti, on='id')
    
    L = len(mbti.mbti.unique())
    K = 20
    mu = ratings.rating.mean()
    M = 10001+lec_num
    N = 10001+lec_num
    
    user = Input(shape=(1,))
    item = Input(shape=(1,))
    P_embedding = Embedding(M, K, embeddings_regularizer=l2())(user)
    Q_embedding = Embedding(N, K, embeddings_regularizer=l2())(item)
    user_bias = Embedding(M, 1, embeddings_regularizer=l2())(user)
    item_bias = Embedding(N, 1, embeddings_regularizer=l2())(item)    
    
    R = layers.dot([P_embedding, Q_embedding], axes=2)
    R = layers.add([R, user_bias, item_bias])
    R = Flatten()(R)

    P_embedding = Flatten()(P_embedding)
    Q_embedding = Flatten()(Q_embedding)
    user_bias = Flatten()(user_bias)
    item_bias = Flatten()(item_bias)
    mbti = Input(shape=(1,))
    mbti_embedding = Embedding(L, 5, embeddings_regularizer=l2())(mbti)
    mbti_layer = Flatten()(mbti_embedding)

    R = Concatenate()([P_embedding, Q_embedding, user_bias, item_bias, mbti_layer])
    R = Dense(2048)(R)
    R = Activation('linear')(R)
    R = Dense(256)(R)
    R = Activation('linear')(R)
    R = Dense(1)(R)

    model = Model(inputs=[user, item,mbti], outputs=R)
    model.compile(
        loss=RMSE,
        optimizer=SGD(),
        metrics=[RMSE]
    )
    model.summary()

    result = model.fit(
        x=[ratings.id.values, ratings.lecture.values, ratings.mbti.values],
        y= ratings.rating.values - mu,
        epochs=80,
        batch_size=1
    )
    return model


# In[6]:




def predict(model, user_id, lec_num, mbti_num, mu):
    user_ids = np.full((lec_num),1)
    mbti_ids = np.full((lec_num),mbti_num)
    item_ids = np.arange(10001,10001+lec_num)
    #print(item_ids)
    
    predictions = a.predict([user_ids, item_ids,mbti_ids])+mu
    temp = pd.DataFrame(item_ids,columns=['lec'])
    temp['score'] = predictions
    temp = temp.sort_values(by=['score'],axis=0,ascending=False)
    print(temp)
    return list(temp['lec'])

def more_learning(model, ratings, mbti):
    ratings = pd.merge(ratings, mbti, on='id')
    print(ratings.lecture.values)
    a.fit(
        x=[ratings.id.values, ratings.lecture.values, ratings.mbti.values],
        y= ratings.rating.values - mu,
        epochs=50,
        batch_size=1
    )


# In[7]:


mu = ratings.rating.mean() #학습 데이터 평균
a = cat_NeuralMF_begin(ratings,mbti,200)
predict(a,3,200,0,mu)

#----------new data do not use yet --------------#
ratings_add_row = pd.read_csv(file_path + "ratings_add.csv")
ratings_add = pd.DataFrame(ratings_add_row, columns=["id","lecture","rating"])

more_learning(a, ratings_add, mbti)


# In[21]:





# In[ ]:




