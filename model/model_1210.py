#!/usr/bin/env python
# coding: utf-8

# In[85]:


import pandas as pd
import numpy as np 
from sklearn.metrics.pairwise import euclidean_distances
import sklearn
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist, jaccard
import copy

#file_path = "C:/Users/user/Desktop/lecture/"
lecture_list = None

def total_sim(like_data,wish_data,score_data,lecture_data,alpha):
    global lecture_list
    #like_in = pd.read_csv(file_path + "ratings_good.csv")
    #wish_in = pd.read_csv(file_path + "ratings_wish.csv")
    #score_in = pd.read_csv(file_path + "ratings_score.csv")
    #lecture_in = pd.read_csv(file_path + "lecture_bin.csv", encoding='cp949')

    like_in = pd.DataFrame(like_data, columns=["id","lecture","rating"])
    wish_in = pd.DataFrame(wish_data, columns=["id","lecture","rating"])
    score_in = pd.DataFrame(score_data, columns=["id","lecture","rating"])
    lecture_in = pd.DataFrame(lecture_data, columns=["id","keyword","value"])

    #user_keyword = pd.DataFrame(a, columns=["keyword","key"])
    
    lecture_wide = pre_lec(lecture_in)
    lecture_list = lecture_wide.sort_values(axis=0,ascending=True,by ="id").index
    #혹시 몰라서 정렬함.
    
    wish_in["rating"] = 1
    
    like_wide = pre(like_in)
    score_wide = pre(score_in)
    wish_wide = pre(wish_in)
    
    like_sim = sim(like_wide)
    score_sim = sim(score_wide)
    wish_sim = sim_bin(wish_wide)
    lecture_sim = sim_lecture(lecture_wide)

    total_sim = like_sim * alpha[1] + score_sim * alpha[2] + wish_sim * alpha[3] + lecture_sim * alpha[4]  
    
    """like_sim.to_csv("like_sim.csv")
    wish_sim.to_csv("wish_sim.csv")
    lecture_sim.to_csv("lecture_sim.csv")
    score_sim.to_csv("score_sim.csv")"""
    
    return total_sim
    
def pre(data):
    data_wide = data.pivot_table(index=["id"] , 
                    columns='lecture', 
                    values='rating')
    data_wide = data_wide.fillna(0)
    
    return data_wide

def pre_lec(data):
    data_wide = data.pivot_table(index=["id"] , 
                    columns='keyword', 
                    values='value')
    data_wide = data_wide.fillna(0)
    
    return data_wide

def sim(data_wide):
    #data_sim = sklearn.metrics.pairwise.cosine_similarity(data_wide.T, dense_output=True)
    #data_sim = pd.DataFrame(data_sim,index=list(data_wide),columns=list(data_wide))
    #data_sim = data_sim - np.eye(len(data_sim))
    
    data_sim = sklearn.metrics.pairwise_distances(data_wide.T, metric='cosine')
    data_sim = pd.DataFrame(data_sim,index=list(data_wide),columns=list(data_wide))
    
    data_sim = 1 - data_sim - np.eye(len(data_sim))
    data_sim = pd.DataFrame(data_sim,index = lecture_list, columns = lecture_list)
    data_sim = data_sim.fillna(0)
    
    #데이터의 수가 희소하므로 표준화는 배제한다. 
    #표준화가 필요하다면 0을 제외하고 진행할 것. 4/9 와 9 가 있을 때, 4/9의 9가 더 작아질 위험이 있음.

    # euclidean_distance & 정규화
    """data_sim = euclidean_distances(data_wide.T,data_wide.T)
    data_sim = data_sim / data_sim.max(axis=0).max()
    data_sim = (data_sim - data_sim.min(axis=0).min()) / (data_sim.max(axis=0).max() - data_sim.min(axis=0).min())
    data_sim = 1 - data_sim - np.eye(len(data_sim))"""

    return data_sim


def sim_bin(data_wide):
    data_sim = 1 - pdist(data_wide.T,'jaccard')
    data_sim = squareform(data_sim)
    
    data_sim = pd.DataFrame(data_sim, index=list(data_wide), columns=list(data_wide))
    data_sim = pd.DataFrame(data_sim,index = lecture_list, columns = lecture_list)
    data_sim = data_sim.fillna(0)    
    
    return data_sim
    
def sim_lecture(lecture_wide):  

    lecture_wide["sum"] = lecture_wide.sum(axis = 1)
    lecture_wide_filtered = lecture_wide[lecture_wide['sum']!= 0]
    lecture_wide_filtered = lecture_wide_filtered.drop(['sum'], axis=1)
    #후에 행 합계가 0인 경우 제거하는 함수 발견하면 수정할 것.

    data_sim = 1- pdist(lecture_wide_filtered,'jaccard')
    data_sim = squareform(data_sim)
    
    data_sim = pd.DataFrame(data_sim, index=list(lecture_wide_filtered.T), columns=list(lecture_wide_filtered.T))
    data_sim = pd.DataFrame(data_sim, index=list(lecture_list), columns=list(lecture_list))
    data_sim = data_sim.fillna(0)  
    return data_sim 

def user_input(data):
    data_df = pd.DataFrame(data, columns=["id","rating"]) 
    data_wide = data_df.pivot_table( 
                    columns='id', 
                    values='rating')
    data_wide_all = pd.DataFrame(data_wide, columns=list(lecture_list))
    data_wide_all = data_wide_all.fillna(0)
    
    return data_wide_all.T

#####################################################################################################

def keyword_sort(user_keyword,lecture_bin):
    
    keyword_index = user_keyword["keyword"]

    user_keyword = user_keyword.set_index("keyword")
    user_keyword_list = list(user_keyword.iloc[:, 0])

    check= lecture_bin * user_keyword_list
    result_list = check.sum(axis=1).sort_values(axis=0,ascending=False).index 
    
    return result_list

def recommend(sim,input_rating):
    
    series = np.dot(sim,input_rating)
    result = pd.DataFrame(series,index=list(sim))
    result = result.sort_values(by=0, axis=0,ascending=False)
    
    return result.index

#######################################################################################################

total_sim = total_sim("C:/Users/user/Desktop/lecture/",[0.7,0.4,0,2,0.1])
user = user_input("여기에 유저 평점 넣으슈")
print(recommend(total_sim,user))


# In[68]:


"""
lecture_wide_row = pd.read_csv(file_path + "lecture_anal_row.csv", encoding='cp949')
lecture_data_long = pd.wide_to_long(lecture_wide_row,stubnames='value', i=['id'], j='keyword',
                    sep='_', suffix='\w+')
lecture_data_long.to_csv("lecture_data_long.csv")

lecture_in = pd.read_csv(file_path + "lecture_bin.csv", encoding='cp949')
"""

