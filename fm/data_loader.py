import pandas as pd
from itertools import count
from collections import defaultdict
from scipy.sparse import csr
import numpy as np
from sklearn.feature_extraction import DictVectorizer

def vectorize_dic(dic,ix=None,p=None):

    if(ix==None):
        # 创建迭代器,生成从n开始的连续整数
        d = count(0)
        ix = defaultdict(lambda :next(d))
    # ix:set(user+item)
    n = len(list(dic.values())[0])  # 样本数
    g = len(list(dic.keys()))  #['users','items'] 每个样本特征数
    nz = n*g  # 总特征个数

    i = 0
    col_ix = np.empty(nz,dtype=int)

    # 将item和user 从0索引
    # col_ix: user1,item1,user2,item2..  2*n
    for k,lis in dic.items():
        col_ix[i::g] = [ix[str(el)+str(k)] for el in lis]
        i += 1

    # n:样本数  0,0,1,1,2,2
    row_ix = np.repeat(np.arange(0,n),g)  # 重复g次
    data = np.ones(nz)

    # p:特征的集合数
    if(p==None):
        p = len(ix)  #set(user+item)

    # col_ix: user_id, item_id, user_id, item_id ....
    # ixx = col_ix
    # 输出True的坐标
    ixx = np.where(col_ix<p)

    return csr.csr_matrix((data[ixx],(row_ix[ixx],col_ix[ixx])),shape=(n,p)), ix












