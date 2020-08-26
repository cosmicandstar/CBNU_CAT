#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

lecture_long = pd.DataFrame(a, columns=["id","keyword","value"])
user_keyword = pd.DataFrame(a, columns=["keyword","key"])

def pre_lec(data):
    data_wide = data.pivot_table(index=["id"] , 
                    columns='keyword', 
                    values='value')
    data_wide = data_wide.fillna(0)
    
    return data_wide

def keyword_sort(user_keyword,lecture_bin):
    
    keyword_index = user_keyword["keyword"]

    user_keyword = user_keyword.set_index("keyword")
    user_keyword_list = list(user_keyword.iloc[:, 0])

    check= lecture_bin * user_keyword_list
    result_list = check.sum(axis=1).sort_values(axis=0,ascending=False).index 
    
    return result_list

lecture_wide = pre_lec(lecture_long)
keyword_sort(user_keyword, lecture_wide)

