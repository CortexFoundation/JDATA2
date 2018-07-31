import os
import time
import pickle
import hashlib
import numpy as np
import pandas as pd
from tool import *
import scipy.stats as scs


cache_path = '../cache/'
data_path = '../data/'
output_path = '../output/'
inplace = False


all_data_path = cache_path + 'all_data3.pkl'
if os.path.exists(all_data_path):
    user_action,user_info,user_comment,user_order,sku_info,all_action,all_user_order = pickle.load(open(all_data_path,'+rb'))
else:
    user_action = pd.read_csv(data_path + 'jdata_user_action.csv',dtype={
        'user_id': np.int32,
        'sku_id': np.int32,
        'a_date': str,
        'a_num': np.int16,
        'a_type': np.int8}).rename(columns={'a_date':'time'})
    user_info = pd.read_csv(data_path + 'jdata_user_basic_info.csv',dtype={
        'user_id': np.int32,
        'age': np.int8,
        'sex': np.int8,
        'user_lv_cd': np.int8})
    user_comment = pd.read_csv(data_path + 'jdata_user_comment_score.csv',dtype={
        'user_id': np.int32,
        'comment_create_tm': str,
        'o_id': np.int32,
        'score_level': np.int8}).rename(columns={'comment_create_tm':'time'})
    user_order = pd.read_csv(data_path + 'jdata_user_order.csv',dtype={
        'user_id': np.int32,
        'sku_id': np.int32,
        'o_id': np.int32,
        'o_area': np.int16,
        'o_sku_num':np.int16}).rename(columns={'o_date':'time'})
    sku_info = pd.read_csv(data_path + 'jdata_sku_basic_info.csv',dtype={
        'sku_id': np.int32,
        'price': np.float64,
        'cate': np.int8,
        'para_1': np.float64,
        'para_2':np.int8,
        'para_3':np.int8})
    # 对用户属性onehot
    user_age = pd.get_dummies(user_info['age']).astype(bool)
    user_age.columns = ['age{}'.format(c) for c in user_age.columns]
    user_info = pd.concat([user_info,user_age],axis=1)
    user_sex = pd.get_dummies(user_info['sex']).astype(bool)
    user_sex.columns = ['sex{}'.format(c) for c in user_sex.columns]
    user_info = pd.concat([user_info, user_sex], axis=1)
    user_lv = pd.get_dummies(user_info['user_lv_cd']).astype(bool)
    user_lv.columns = ['user_lv_cd{}'.format(c) for c in user_lv.columns]
    user_info = pd.concat([user_info, user_lv], axis=1)
    user_info.drop(['age','sex','user_lv_cd'],axis=1,inplace=True)

    date_day_map = {str(date)[:10]: diff_of_days(str(date)[:10], '2016-05-01') for date in
                    pd.date_range('2016-05-01', '2017-11-01')}
    user_action['diff_of_days'] = user_action['time'].map(date_day_map).astype(np.int16)
    user_action.sort_values('diff_of_days',inplace=True)
    user_action = user_action.merge(sku_info, on='sku_id', how='left')
    user_action = user_action[user_action['cate'].isin([30,101])]

    user_order['diff_of_days'] = user_order['time'].map(date_day_map).astype(np.int16)
    user_order.sort_values('diff_of_days',inplace=True)
    user_order = user_order.merge(sku_info,on='sku_id',how='left')
    all_user_order = user_order.copy()
    user_order = user_order[user_order['cate'].isin([30,101])]

    user_comment['diff_of_days'] = user_comment['time'].str[:10].map(date_day_map).astype(np.int16)
    user_comment.sort_values('time',inplace=True)
    user_comment = user_comment.merge(user_order[['o_id','sku_id','price', 'cate', 'para_1', 'para_2', 'para_3']],on='o_id',how='left')
    user_comment = user_comment[user_comment['cate'].isin([30,101])]


    all_action1 = user_action[['user_id','sku_id','time','a_num','a_type','cate','diff_of_days']].rename(columns={'a_num':'num','a_type':'type'})
    # all_action1['type'] = 1
    all_action2 = user_order[['user_id','sku_id','time','o_sku_num','cate','diff_of_days']].rename(columns={'o_sku_num':'num'})
    all_action2['type'] = 3
    all_action3 = user_comment[['user_id','sku_id','time','cate','diff_of_days']]
    all_action3['type'] = 4
    all_action = pd.concat([all_action1, all_action2, all_action3],axis=0)
    all_action['num'].fillna(1,inplace=True)
    all_action.sort_values(['diff_of_days','type'],inplace=True)
    pickle.dump((user_action,user_info,user_comment,user_order,sku_info,all_action,all_user_order),open(all_data_path,'+wb'))

# sku_id = sku_info[sku_info['cate'].isin([30,101])]['sku_id'].values

############################### 工具函数 ###########################
# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result


def pre_treatment(data_key):
    result_path = cache_path + 'data_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & 1:
        data = pd.read_hdf(result_path, 'w')
    else:
        if (data_key <= '2017-02-01'):
            data = user_order[(user_order['time'] < data_key) & (user_order['time'] >= date_add_days(data_key,-90))]
        # elif (data_key < '2017-02-01'):
        #     data = user_order[(user_order['time'] < data_key) & (user_order['time'] >= date_add_days(data_key,-90))]
        else:
            data = user_order[(user_order['time'] < data_key) & ((user_order['time'] >= '2017-02-01'))]
        data = data[['user_id']].drop_duplicates()
        data['end_date'] = data_key
        data.reset_index(drop=True,inplace=True)
        data.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return data

# 用户action时间特征
def get_user_actioin_time_feat(data, data_key):
    result_path = cache_path + 'user_action_time_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        start_day = diff_of_days(data_key,'2016-05-01')
        user_action_temp = user_action[(user_action['time']<data_key)].copy()
        user_action_temp['diff_of_days'] = (start_day - user_action_temp['diff_of_days']).astype(np.int16)
        data_temp['user_action_last0_time'] = get_last_values(data_temp, user_action_temp, 'user_id', 'time', 'diff_of_days', shift=0, sort=None)
        for i in ['a_type','price','para_1','para_2','para_3']:
            for j in [0,1]:
                data_temp['user_action_last{}_{}'.format(i,j)] = get_last_values(data_temp, user_action_temp, 'user_id', 'time', i, shift=j)
        for i in [2,3,4]:
            data_temp['user_action_last{}_time'.format(i)] = get_last_values(data_temp, user_action_temp, 'user_id', 'time', 'diff_of_days', shift=i)
        for i in [0,1,2]:
            data_temp['user_action_last{}_type2'.format(i)] = get_last_values(data_temp, user_action_temp[
                user_action_temp['a_type'] == 2], 'user_id', 'time', 'diff_of_days', shift=i)
        for i in [30,101]:
            data_temp['user_action_cate{}_last0_time'.format(i)] = get_last_values(data_temp, user_action_temp[
                user_action_temp['cate'] == i], 'user_id', 'time', 'diff_of_days', shift=0)
            data_temp['user_action_cate{}_last0_type2'.format(i)] = get_last_values(data_temp, user_action_temp[(user_action_temp['a_type'] == 2) &
            (user_action_temp['cate'] == i)], 'user_id', 'time', 'diff_of_days', shift=0)
        user_action_temp = user_action_temp[['user_id', 'diff_of_days']].drop_duplicates()
        for i in [1, 2, 3, 4, 5, 6, 7]:  # 去重待预测品类
            data_temp['user_action_last{}_time_drop_duplicate'.format(i)] = get_last_values(data_temp, user_action_temp, 'user_id', 'time', 'diff_of_days', shift=i)
        feat = data_temp
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 用户order时间特征
def get_user_order_time_feat(data, data_key):
    result_path = cache_path + 'user_order_time_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        start_day = diff_of_days(data_key,'2016-05-01')
        user_order_temp = user_order[(user_order['time']<data_key)].copy()
        user_order_temp['diff_of_days'] = (start_day - user_order_temp['diff_of_days']).astype(np.int16)
        for i in [0,1,2,3,4,5,6,7]:                                 # 待预测品类
            data_temp['user_order_last{}_time'.format(i)] = get_last_values(data_temp, user_order_temp, 'user_id', 'time', 'diff_of_days', shift=i)
            data_temp['user_order_last{}_o_sku_num'.format(i)] = get_last_values( data_temp, user_order_temp, 'user_id', 'time', 'o_sku_num', shift=i)

        data_temp['user_order_last{}_area'.format(i)] = get_last_values(data_temp, user_order_temp, 'user_id', 'time', 'o_area', shift=i) # 最后一次地址
        for i in [30,101]:                               # 各个品类
            data_temp['user_order_cate{}_last0_time'.format(i)] = get_last_values(
                data_temp, user_order_temp[user_order_temp['cate'] == i], 'user_id', 'time', 'diff_of_days', shift=0)
            data_temp['user_order_cate{}_last0_num'.format(i)] = get_last_values( data_temp, user_order_temp, 'user_id', 'time', 'o_sku_num', shift=0)  # 最后一次购买个数
        user_order_temp = user_order_temp.groupby(['user_id', 'diff_of_days'],as_index=False)['o_sku_num'].agg({'o_sku_num':'sum'})
        for i in [1,2,3,4,5,6,7]:                                       # 去重待预测品类
            data_temp['user_order_select_last{}_time_drop_duplicate'.format(i)] = get_last_values(data_temp, user_order_temp, 'user_id', 'time', 'diff_of_days', shift=i)
            data_temp['user_order_select_num_last{}_time_drop_duplicate'.format(i)] = get_last_values(data_temp, user_order_temp, 'user_id', 'time', 'o_sku_num', shift=i)
        user_order_temp = all_user_order[(all_user_order['time'] < data_key) & (all_user_order['cate'].isin([71, 1, 83, 46]))].copy()
        user_order_temp['diff_of_days'] = (start_day - user_order_temp['diff_of_days']).astype(np.int16)
        data_temp['user_order_other_last0_time'] = get_last_values(data_temp, user_order_temp, 'user_id', 'time', 'diff_of_days', shift=0)
        data_temp['user_order_other_last0_time_diff_days'] = data_temp['user_order_last0_time'] - data_temp['user_order_other_last0_time']
        feat = data_temp
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat


# 用户order时间特征[71,1,83,46]
def get_user_order_other_time_feat(data, data_key):
    result_path = cache_path + 'user_order_other_time_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        start_day = diff_of_days(data_key,'2016-05-01')
        user_order_temp = all_user_order[(all_user_order['time']<data_key) & (all_user_order['cate'].isin([71, 1, 83, 46]))].copy()
        user_order_temp['diff_of_days'] = (start_day - user_order_temp['diff_of_days']).astype(np.int16)
        for i in [0,1,2,3,4,5,6,7]:                                 # 待预测品类
            data_temp['user_order_last{}_time'.format(i)] = get_last_values(data_temp, user_order_temp, 'user_id', 'time', 'diff_of_days', shift=i)
            data_temp['user_order_last{}_o_sku_num'.format(i)] = get_last_values( data_temp, user_order_temp, 'user_id', 'time', 'o_sku_num', shift=i)

        data_temp['user_order_last{}_area'.format(i)] = get_last_values(data_temp, user_order_temp, 'user_id', 'time', 'o_area', shift=i) # 最后一次地址
        for i in [30,101]:                               # 各个品类
            data_temp['user_order_cate{}_last0_time'.format(i)] = get_last_values(
                data_temp, user_order_temp[user_order_temp['cate'] == i], 'user_id', 'time', 'diff_of_days', shift=0)
            data_temp['user_order_cate{}_last0_num'.format(i)] = get_last_values( data_temp, user_order_temp, 'user_id', 'time', 'o_sku_num', shift=0)  # 最后一次购买个数
        user_order_temp = user_order_temp.groupby(['user_id', 'diff_of_days'],as_index=False)['o_sku_num'].agg({'o_sku_num':'sum'})
        for i in [1,2,3,4,5,6,7]:                                       # 去重待预测品类
            data_temp['user_order_select_last{}_time_drop_duplicate'.format(i)] = get_last_values(data_temp, user_order_temp, 'user_id', 'time', 'diff_of_days', shift=i)
            data_temp['user_order_select_num_last{}_time_drop_duplicate'.format(i)] = get_last_values(data_temp, user_order_temp, 'user_id', 'time', 'o_sku_num', shift=i)
        feat = data_temp
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat
# 用户action-order时间特征
def get_user_action_order_time_feat(data, data_key):
    result_path = cache_path + 'user_action_order_time_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        start_day = diff_of_days(data_key,'2016-05-01')
        user_order_temp = user_order[(user_order['time']<data_key)].copy()
        user_order_temp['o_diff_of_days'] = (start_day - user_order_temp['diff_of_days']).astype(np.int16)
        user_order_temp = user_order_temp[['user_id','cate','o_diff_of_days']]
        user_action_temp = user_action[(user_action['time'] < data_key)].copy()
        user_action_temp['diff_of_days'] = (start_day - user_action_temp['diff_of_days']).astype(np.int16)
        user_action_temp = user_action_temp.merge(user_order_temp,on=['user_id','cate'],how='left')
        user_action_temp = user_action_temp[user_action_temp['diff_of_days']>user_action_temp['o_diff_of_days']].copy()
        data_temp['user_action-order_max_time'] = groupby(data_temp,user_action_temp,'user_id','diff_of_days',max).fillna(90).astype(np.int16)
        data_temp['user_action-order_median_time'] = groupby(data_temp, user_action_temp, 'user_id', 'diff_of_days','median').fillna(90).astype(np.int16)
        data_temp['user_action-order_std_time'] = groupby(data_temp, user_action_temp, 'user_id', 'diff_of_days', 'std').fillna(-1).astype(np.int16)
        data_temp['user_action-order_nunique_time'] = groupby(data_temp, user_action_temp, 'user_id', 'diff_of_days', 'nunique').fillna(0).astype(np.int16)
        data_temp['user_action-order_count_time'] = groupby(data_temp, user_action_temp, 'user_id', 'diff_of_days', 'count').fillna(0).astype(np.int16)
        data_temp['user_action-order_o_num_sum'] = groupby(data_temp, user_action_temp, 'user_id', 'a_num', 'sum').fillna(0).astype(np.int16)
        data_temp['user_action-order_o_num_sum_type2'] = groupby(data_temp, user_action_temp[user_action_temp['a_type']==2], 'user_id', 'a_num', 'sum').fillna(0).astype(np.int16)
        feat = data_temp
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 用户comment时间特征
def get_user_comment_time_feat(data, data_key):
    result_path = cache_path + 'user_comment_time_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        start_day = diff_of_days(data_key,'2016-05-01')
        user_comment_temp = user_comment[(user_comment['time']<data_key)].copy()
        user_comment_temp['diff_of_days'] = (start_day - user_comment_temp['diff_of_days']).astype(np.int16)
        data_temp['user_comment_last0_time'] = get_last_values(data_temp, user_comment_temp, 'user_id', 'time', 'diff_of_days', shift=0)
        data_temp['user_comment_last0_score'] = get_last_values(data_temp, user_comment_temp, 'user_id', 'time', 'score_level', shift=0)
        data_temp['user_comment_last1_time'] = get_last_values(data_temp, user_comment_temp, 'user_id', 'time', 'diff_of_days', shift=1)
        data_temp['user_comment_last1_score'] = get_last_values(data_temp, user_comment_temp, 'user_id', 'time', 'score_level', shift=1)
        feat = data_temp
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 用户action时间特征
def get_user_action_time_feat2(data, data_key, days):
    result_path = cache_path + 'user_action_time_feat2{}_{}days.hdf'.format(data_key,days)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        start_day = diff_of_days(data_key,'2016-05-01')
        user_action_temp = user_action[(user_action['time']<data_key) & (user_action['time']>=date_add_days(data_key,-days))].copy()
        user_action_temp['diff_of_days'] = (start_day - user_action_temp['diff_of_days']).astype(np.int16)
        data_temp['user_action_first_time_{}days'.format(days)] = groupby(data_temp, user_action_temp, 'user_id', 'diff_of_days', 'max').fillna(-1).astype(np.int16)
        data_temp['user_action_mean_time_{}days'.format(days)] = groupby(data_temp, user_action_temp, 'user_id', 'diff_of_days', 'mean').fillna(-1).astype(np.int16)
        data_temp['user_action_std_time_{}days'.format(days)] = groupby(data_temp, user_action_temp, 'user_id', 'diff_of_days', 'std').fillna(-1).astype( np.int16)
        user_action_temp['diff_of_days2'] = user_action_temp['diff_of_days'] - user_action_temp.groupby('user_id')['diff_of_days'].shift(1)
        data_temp['user_action_max_diff_time_{}days'.format(days)] = groupby(data_temp, user_action_temp, 'user_id', 'diff_of_days2', 'max').fillna(-1).astype(np.int16)
        data_temp['user_action_min_diff_time_{}days'.format(days)] = groupby(data_temp, user_action_temp, 'user_id', 'diff_of_days2', 'min').fillna(-1).astype(np.int16)
        data_temp['user_action_mean_diff_time_{}days'.format(days)] = groupby(data_temp, user_action_temp, 'user_id', 'diff_of_days2', 'mean').fillna(-1).astype( np.int16)
        data_temp['user_action_std_diff_time_{}days'.format(days)] = groupby(data_temp, user_action_temp, 'user_id', 'diff_of_days2', 'std').fillna(-1).astype(np.int16)
        user_action_temp.drop_duplicates(['user_id','diff_of_days'],inplace=True)
        data_temp['user_action_mean_time_drop_duplicate_{}days'.format(days)] = groupby(data_temp, user_action_temp, 'user_id', 'diff_of_days', 'mean').fillna(-1).astype(np.int16)
        data_temp['user_action_std_time_drop_duplicate_{}days'.format(days)] = groupby(data_temp, user_action_temp, 'user_id','diff_of_days', 'std').fillna(-1).astype(np.int16)
        user_action_temp['diff_of_days2'] = user_action_temp['diff_of_days'] - user_action_temp.groupby('user_id')[ 'diff_of_days'].shift(1)
        data_temp['user_action_mean_diff_time_drop_duplicate_{}days'.format(days)] = groupby(data_temp, user_action_temp, 'user_id', 'diff_of_days2', 'mean').fillna(-1).astype(np.int16)
        data_temp['user_action_std_diff_time_drop_duplicate_{}days'.format(days)] = groupby(data_temp, user_action_temp, 'user_id', 'diff_of_days2', 'std').fillna(-1).astype(np.int16)
        feat = data_temp
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 用户order时间特征
def get_user_order_time_feat2(data, data_key, days):
    result_path = cache_path + 'user_order_time_feat2{}_{}days.hdf'.format(data_key,days)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        start_day = diff_of_days(data_key,'2016-05-01')
        user_order_temp = user_order[(user_order['time']<data_key) & (user_order['time']>=date_add_days(data_key,-days))].copy()
        user_order_temp['diff_of_days'] = (start_day - user_order_temp['diff_of_days']).astype(np.int16)
        data_temp['user_order_first_time_{}days'.format(days)] = groupby(data_temp, user_order_temp, 'user_id', 'diff_of_days', 'max').fillna(-1).astype(np.int16)
        data_temp['user_order_mean_time_{}days'.format(days)] = groupby(data_temp, user_order_temp, 'user_id', 'diff_of_days', 'mean').fillna(-1).astype(np.int16)
        data_temp['user_order_std_time_{}days'.format(days)] = groupby(data_temp, user_order_temp, 'user_id', 'diff_of_days', 'std').fillna(-1).astype( np.int16)
        user_order_temp['diff_of_days2'] = user_order_temp['diff_of_days'] - user_order_temp.groupby('user_id')['diff_of_days'].shift(1)
        data_temp['user_order_max_diff_time_{}days'.format(days)] = groupby(data_temp, user_order_temp, 'user_id', 'diff_of_days2', 'max').fillna(-1).astype(np.int16)
        data_temp['user_order_min_diff_time_{}days'.format(days)] = groupby(data_temp, user_order_temp, 'user_id', 'diff_of_days2', 'min').fillna(-1).astype(np.int16)
        data_temp['user_order_mean_diff_time_{}days'.format(days)] = groupby(data_temp, user_order_temp, 'user_id', 'diff_of_days2', 'mean').fillna(-1).astype( np.int16)
        data_temp['user_order_std_diff_time_{}days'.format(days)] = groupby(data_temp, user_order_temp, 'user_id', 'diff_of_days2', 'std').fillna(-1).astype(np.int16)
        data_temp['user_order_last_diff_time_{}days'.format(days)] = groupby(data_temp, user_order_temp, 'user_id', 'diff_of_days2', 'last').fillna(-1).astype(np.int16)
        user_order_temp.drop_duplicates(['user_id','diff_of_days'],inplace=True)
        data_temp['user_order_mean_time_drop_duplicate_{}days'.format(days)] = groupby(data_temp, user_order_temp, 'user_id', 'diff_of_days', 'mean').fillna(-1).astype(np.int16)
        data_temp['user_order_std_time_drop_duplicate_{}days'.format(days)] = groupby(data_temp, user_order_temp, 'user_id','diff_of_days', 'std').fillna(-1).astype(np.int16)
        user_order_temp['diff_of_days2'] = user_order_temp['diff_of_days'] - user_order_temp.groupby('user_id')[ 'diff_of_days'].shift(1)
        data_temp['user_order_mean_diff_time_drop_duplicate_{}days'.format(days)] = groupby(data_temp, user_order_temp, 'user_id', 'diff_of_days2', 'mean').fillna(-1).astype(np.int16)
        data_temp['user_order_std_diff_time_drop_duplicate_{}days'.format(days)] = groupby(data_temp, user_order_temp, 'user_id', 'diff_of_days2', 'std').fillna(-1).astype(np.int16)
        feat = data_temp
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 用户活跃的次数
def get_user_active_period_count_feat(data, data_key, days):
    result_path = cache_path + 'user_active_period_count_feat_{}_{}days.hdf'.format(data_key, days)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        end_date = date_add_days(data_key, -days)
        action_temp = user_action[(user_action['time'] < data_key) & (user_action['time'] >= end_date)].copy()
        order_temp = user_order[(user_order['time'] < data_key) & (user_order['time'] >= end_date)].copy()
        comment_temp = user_comment[(user_comment['time'] < data_key) & (user_comment['time'] >= end_date)].copy()
        all_action_temp = all_action[(all_action['time'] < data_key) & (all_action['time'] >= end_date)].copy()
        action_doy = action_temp['diff_of_days'].copy()
        order_doy = order_temp['diff_of_days'].copy()
        comment_doy = comment_temp['diff_of_days'].copy()
        all_action_doy = all_action_temp['diff_of_days'].copy()
        for i in [2, 3, 5, 7, 15, 30]:
            action_temp['diff_of_days'] = action_doy // i
            order_temp['diff_of_days'] = order_doy // i
            comment_temp['diff_of_days'] = comment_doy // i
            all_action_temp['diff_of_days'] = all_action_doy // i
            data_temp['user_action_select_n_period_{}days_type{}'.format(days, i)] = groupby(data_temp, action_temp, 'user_id', 'diff_of_days','nunique').fillna(0).astype('int16')
            data_temp['user_order_select_n_period_{}days_type{}'.format(days, i)] = groupby(data_temp, order_temp, 'user_id', 'diff_of_days','nunique').fillna(0).astype('int16')
            data_temp['user_comment_select_n_period_{}days_type{}'.format(days, i)] = groupby(data_temp, comment_temp, 'user_id', 'diff_of_days', 'nunique').fillna(0).astype('int16')
            data_temp['user_all_action_select_n_period_{}days_type{}'.format(days, i)] = groupby(data_temp, all_action_temp, 'user_id', 'diff_of_days', 'nunique').fillna(0).astype('int16')
        feat = data_temp
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 用户单位时间内action的统计特征
def get_user_action_stat_feat(data, data_key, days):
    result_path = cache_path + 'user_action_stat_feat_{}_{}days.hdf'.format(data_key,days)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        end_date = date_add_days(data_key, -days)
        action_temp = user_action[(user_action['time'] < data_key) & (user_action['time'] >= end_date)].copy()
        data_temp['user_action_n_items_{}days'.format(days)] = groupby(data_temp,action_temp,'user_id','sku_id','nunique').fillna(0).astype('int16')
        data_temp['user_action_n_cate_{}days'.format(days)] = groupby(data_temp, action_temp, 'user_id', 'cate','nunique').fillna(0).astype('int16')
        data_temp['user_action_count_{}days'.format(days)] = groupby(data_temp, action_temp, 'user_id', 'a_num', 'count').fillna(0).astype('int16')
        data_temp['user_action_item_count_{}days'.format(days)] = groupby(data_temp, action_temp, 'user_id', 'a_num', sum).fillna(0).astype('int16')
        data_temp['user_action_n_days_{}days'.format(days)] = groupby(data_temp, action_temp, 'user_id', 'time', 'nunique').fillna(0).astype('int16')
        data_temp['user_action_n_item_type2_{}days'.format(days)] = groupby(data_temp, action_temp[action_temp['a_type'] == 2], 'user_id', 'sku_id', 'nunique').fillna(0).astype('int16')
        data_temp['user_action_item_count_type2_{}days'.format(days)] = groupby(data_temp, action_temp[action_temp['a_type'] == 2],  'user_id', 'a_num', 'sum').fillna(0).astype('int16')
        data_temp['user_action_n_days_type2_{}days'.format(days)] = groupby(data_temp, action_temp[action_temp['a_type'] == 2], 'user_id', 'time', 'nunique').fillna(0).astype('int16')
        data_temp['user_action_count_type2_{}days'.format(days)] = groupby(data_temp, action_temp[action_temp['a_type'] == 2], 'user_id', 'time', 'count').fillna(0).astype('int16')
        action_temp = user_action[(user_action['time'] < data_key) & (user_action['time'] >= end_date)].copy()
        data_temp['user_action_count_{}days'.format(days)] = groupby(data_temp, action_temp, 'user_id', 'a_num', 'count').fillna(0).astype('int16')
        data_temp['user_action_count_type2_{}days'.format(days)] = groupby(data_temp, action_temp[action_temp['a_type'] == 2], 'user_id', 'time', 'count').fillna(0).astype('int16')

        feat = data_temp
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 用户单位时间内order的统计特征
def get_user_order_stat_feat(data, data_key, days):
    result_path = cache_path + 'user_order_stat_feat_{}_{}days.hdf'.format(data_key,days)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        end_date = date_add_days(data_key, -days)
        order_temp = user_order[(user_order['time'] < data_key) & (user_order['time'] >= end_date)].copy()
        data_temp['user_order_count_{}days'.format(days)] = groupby(data_temp, order_temp, 'user_id', 'sku_id', 'count').fillna(0).astype('int16')
        data_temp['user_order_n_items_{}days'.format(days)] = groupby(data_temp,order_temp,'user_id','sku_id','nunique').fillna(0).astype('int16')
        data_temp['user_order_n_cates_{}days'.format(days)] = groupby(data_temp, order_temp, 'user_id', 'cate', 'nunique').fillna(0).astype('int16')
        data_temp['user_order_n_area_{}days'.format(days)] = groupby(data_temp, order_temp, 'user_id', 'o_area', 'nunique').fillna(0).astype('int16')
        data_temp['user_order_n_oid_{}days'.format(days)] = groupby(data_temp, order_temp, 'user_id', 'o_id','nunique').fillna(0).astype('int16')
        data_temp['user_order_item_count_{}days'.format(days)] = groupby(data_temp, order_temp, 'user_id', 'o_sku_num', sum).fillna(0).astype('int16')
        data_temp['user_order_item_max_{}days'.format(days)] = groupby(data_temp, order_temp, 'user_id', 'o_sku_num',max).fillna(0).astype('int16')
        data_temp['user_order_item_mean_{}days'.format(days)] = groupby(data_temp, order_temp, 'user_id', 'o_sku_num','mean').fillna(0)
        data_temp['user_order_n_user_days_{}days'.format(days)] = groupby(data_temp, order_temp, 'user_id', 'time', 'nunique').fillna(0).astype('int16')
        feat = data_temp
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# action全局变量
def get_user_action_feat(data, data_key, days):
    result_path = cache_path + 'user_action_feat_{}_{}days.hdf'.format(data_key,days)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        end_date = date_add_days(data_key, -days)
        stat = user_action[(user_action['time'] < data_key) & (user_action['time'] >= end_date) & (user_action['a_type']==2)].copy()
        for j in ['price', 'para_1', 'para_2', 'para_3']:
            data_temp['user_action_mean_{}_{}days'.format(j,days)] = groupby(data_temp, stat, 'user_id', j, np.mean)
            data_temp['user_action_median_{}_{}days'.format(j,days)] = groupby(data_temp, stat, 'user_id', j, np.median)
            data_temp['user_action_max_{}_{}days'.format(j,days)] = groupby(data_temp, stat, 'user_id', j, np.max)
            data_temp['user_action_min_{}_{}days'.format(j,days)] = groupby(data_temp, stat, 'user_id', j, np.min)
            data_temp['user_action_std_{}_{}days'.format(j,days)] = groupby(data_temp, stat, 'user_id', j, np.std)
        feat = data_temp
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# order全局变量
def get_user_order_feat(data, data_key, days):
    result_path = cache_path + 'user_order_feat_{}_{}days.hdf'.format(data_key,days)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        end_date = date_add_days(data_key, -days)
        stat = user_order[(user_order['time'] < data_key) & (user_order['time'] >= end_date)].copy()
        # for i in [101, 30]:
        #     for j in ['price','para_1','para_2','para_3']:
        #         stat_temp = stat[stat['cate']==i].copy()
        #         data_temp['user_order_cate{}_mean_{}_{}days'.format(i,j,days)] = groupby(data_temp, stat_temp, 'user_id', j, np.mean)
        #         data_temp['user_order_cate{}_median_{}_{}days'.format(i, j,days)] = groupby(data_temp, stat_temp, 'user_id', j, np.median)
        #         data_temp['user_order_cate{}_max_{}_{}days'.format(i, j,days)] = groupby(data_temp, stat_temp, 'user_id', j, np.max)
        #         data_temp['user_order_cate{}_min_{}_{}days'.format(i, j,days)] = groupby(data_temp, stat_temp, 'user_id', j, np.min)
        #         data_temp['user_order_cate{}_std_{}_{}days'.format(i, j,days)] = groupby(data_temp, stat_temp, 'user_id', j, np.std)
        for j in ['price', 'para_1', 'para_2', 'para_3']:
            data_temp['user_order_mean_{}_{}days'.format(j,days)] = groupby(data_temp, stat, 'user_id', j, np.mean)
            data_temp['user_order_median_{}_{}days'.format(j,days)] = groupby(data_temp, stat, 'user_id', j, np.median)
            data_temp['user_order_max_{}_{}days'.format(j,days)] = groupby(data_temp, stat, 'user_id', j, np.max)
            data_temp['user_order_min_{}_{}days'.format(j,days)] = groupby(data_temp, stat, 'user_id', j, np.min)
            data_temp['user_order_std_{}_{}days'.format(j,days)] = groupby(data_temp, stat, 'user_id', j, np.std)
        feat = data_temp
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# order全局变量
def get_user_order_feat2(data, data_key, days):
    result_path = cache_path + 'user_order_feat2_{}_{}days.hdf'.format(data_key,days)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        end_date = date_add_days(data_key, -days)
        stat = user_order[(user_order['time'] < data_key) & (user_order['time'] >= end_date)].copy()
        sku_ids = [11206,  16197,  31517,  75756,  64874,  55837,  51600,  31841,
             84163,  93317,  90025,  64960,  38949,  90936,  49365,  33833,
             90894, 104140,   5325,  58700]
        stat = stat[stat['sku_id'].isin(sku_ids)]
        stat = stat.groupby(['user_id','sku_id'])['user_id'].agg({'count':'count'}).unstack()
        stat.columns = [str(c) + '_count_{}days'.format(days) for c in stat.columns.levels[1]]
        stat.reset_index(inplace=True)
        feat = data_temp.merge(stat,on='user_id',how='left')
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 构造候选集和label
def pre_treatment2(data_key):
    result_path = cache_path + 'data2_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & 1:
        data = pd.read_hdf(result_path, 'w')
    else:
        data = user_order[(user_order['time'] < data_key) & (user_order['time'] >= date_add_days(data_key,-30))]
        data = data.drop_duplicates(['user_id','sku_id'])
        data['end_date'] = data_key
        label = user_order[(user_order['time'] >= data_key) & (user_order['time'] < date_add_days(data_key,28))].copy()
        data['label'] = (data['user_id'].isin(set(label['user_id'].values))).astype(int)
        data.reset_index(drop=True,inplace=True)
        data.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return data

# 转化率特征
def get_user_rate_feat2(data, data_key, days):
    result_path = cache_path + 'user_rate_feat2_{}_{}days.hdf'.format(data_key, days)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        end_date = date_add_days(data_key, -days)
        stat = user_order[(user_order['time'] < data_key) & (user_order['time'] >= end_date)].copy()
        stat2 = pd.DataFrame()
        for i in range(100):
            date = date_add_days(data_key,-28-7*i)
            if date<'2016-08-01':
                break
            stat2 = stat2.append(pre_treatment2(date))
        stat['sku_rate'] = stat['sku_id'].map(stat2.groupby('sku_id')['label'].mean())
        data_temp['user_rate_{}day'.format(days)] = data_temp['user_id'].map(stat2.drop_duplicates(['user_id','end_date']).groupby('user_id')['label'].mean())
        data_temp['sku_rate_mean_{}day'.format(days)] = data_temp['user_id'].map(stat.groupby('user_id')['sku_rate'].mean())
        data_temp['sku_rate_max_{}day'.format(days)] = data_temp['user_id'].map( stat.groupby('user_id')['sku_rate'].max())
        data_temp['sku_rate_min_{}day'.format(days)] = data_temp['user_id'].map(stat.groupby('user_id')['sku_rate'].min())
        feat = data_temp
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 构造候选集和label
def pre_treatment3(data_key):
    result_path = cache_path + 'data3_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & 1:
        data = pd.read_hdf(result_path, 'w')
    else:
        data = user_action[(user_action['time'] < data_key) & (user_action['time'] >= date_add_days(data_key,-30))]
        data = data.drop_duplicates(['user_id','sku_id'])
        data['end_date'] = data_key
        label = user_order[(user_order['time'] >= data_key) & (user_order['time'] < date_add_days(data_key,28))].copy()
        data['label'] = (data['user_id'].isin(set(label['user_id'].values))).astype(int)
        data.reset_index(drop=True,inplace=True)
        data.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return data

# 转化率特征
def get_user_rate_feat3(data, data_key, days):
    result_path = cache_path + 'user_rate_feat3_{}_{}days.hdf'.format(data_key, days)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        end_date = date_add_days(data_key, -days)
        stat = user_action[(user_action['time'] < data_key) & (user_action['time'] >= end_date)].copy()
        stat2 = pd.DataFrame()
        for i in range(100):
            date = date_add_days(data_key,-28-7*i)
            if date<'2016-08-01':
                break
            stat2 = stat2.append(pre_treatment3(date))
        stat['sku_rate'] = stat['sku_id'].map(stat2.groupby('sku_id')['label'].mean())
        data_temp['user_action_rate_{}day'.format(days)] = data_temp['user_id'].map(stat2.drop_duplicates(['user_id','end_date']).groupby('user_id')['label'].mean())
        data_temp['sku_action_rate_mean_{}day'.format(days)] = data_temp['user_id'].map(stat.groupby('user_id')['sku_rate'].mean())
        data_temp['sku_action_rate_max_{}day'.format(days)] = data_temp['user_id'].map( stat.groupby('user_id')['sku_rate'].max())
        data_temp['sku_action_rate_min_{}day'.format(days)] = data_temp['user_id'].map(stat.groupby('user_id')['sku_rate'].min())
        feat = data_temp
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 获取label
def get_label(result,data_key,days = None):
    start_day = diff_of_days(data_key, '2016-05-01')
    if days is None:
        end_date = date_add_days(data_key,28)
    else:
        end_date = date_add_days(data_key, 28)
    label = user_order[(user_order['time']>=data_key) & (user_order['time']<end_date)].copy()
    label['diff_of_days'] = label['diff_of_days'] - start_day
    label = label.drop_duplicates('user_id',keep='first')
    result = result.merge(label[['user_id','diff_of_days']],on='user_id',how='left')
    result['label'] = (~result['diff_of_days'].isnull()).astype(int)
    return result


############################### 预处理函数 ###########################
# 用户基础特征
def get_user_base_feat(data,data_key):
    result_path = cache_path + 'user_time_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        data_temp = data_temp.merge(user_info,on='user_id',how='left')
        for i in range(1, 19):
            data_temp = data_temp.merge(pd.read_hdf(cache_path + "p%s.hdf" % i, 'w'), 'left', 'user_id')
        feat = data_temp
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat


def make_feat(data_key):
    t0 = time.time()
    print('数据key为：{}'.format(data_key))
    result_path = cache_path + 'feat_set_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & 0:
        result = pd.read_hdf(result_path, 'w')
    else:
        data = pre_treatment(data_key)

        # print('开始构造特征...')
        result = [data]
        result.append(get_user_base_feat(data,data_key))                # 用户基础特征
        # if (data_key <= '2017-01-01') | (data_key == '2017-02-01') | (data_key == '2017-05-01'):
        result.append(get_user_actioin_time_feat(data, data_key))       # 用户action最后一次时间
        result.append(get_user_order_time_feat(data, data_key))         # 用户order最后一次时间
        result.append(get_user_action_order_time_feat(data, data_key))  # 用户action最后一次时间
        result.append(get_user_comment_time_feat(data, data_key))       # 用户comment最后一次时间
        # if data_key > '2017-02-01':
        for days in [90,180,300]:
            # result.append(get_user_action_time_feat2(data, data_key, days))  # 用户action最后一次时间
            result.append(get_user_order_time_feat2(data, data_key, days))  # 用户order最后一次时间
                # result.append(get_user_all_action_time_feat(data, data_key))       # 用户所有行为最后一次时间
        for days in [30, 90, 150, 300]:
            result.append(get_user_active_period_count_feat(data, data_key, days))  # 活跃的次数（合并天）
            result.append(get_user_action_stat_feat(data, data_key, days))      # 用户单位时间内的action统计特征
            result.append(get_user_order_stat_feat(data, data_key, days))  # 用户单位时间内的order统计特征
        for days in [30,  90, 300]:
            # result.append(get_user_action_feat(data, data_key, days))              # action全局变量
            result.append(get_user_order_feat(data, data_key, days))            # order全局变量'price', 'para_1', 'para_2', 'para_3'
        for days in [30]:
            # result.append(get_user_actioin_feat2(data, data_key))                 # action全局变量
            result.append(get_user_order_feat2(data, data_key, days))            # order全局变量
            result.append(get_user_rate_feat2(data, data_key, days))               # 转化率特征
            result.append(get_user_rate_feat3(data, data_key, days))                # 转化率特征

        # print('开始合并特征...')
        result = concat(result)

        # result = second_feat(result)
        # print('添加label')
        result = get_label(result,data_key)
        # print('存储数据...')
        # result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result



import time
import pickle
import datetime
from tqdm import tqdm
import lightgbm as lgb
# from jdata2.feat_user3 import *

train_feat = []
start_date = '2017-08-04'
days = 2
for i in tqdm(range(days)):
    train_feat.append(make_feat(date_add_days(start_date, i*(-28))[:10]))
train_feat = pd.concat(train_feat,axis=0)

eval_feat = make_feat('2017-09-01')

predictors = [c for c in train_feat.columns if c not in ['end_date','diff_of_days','label','pred']]

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 8,
    'num_leaves': 63,
    'learning_rate': 0.02,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 66,
}
lgb_train = lgb.Dataset(train_feat[predictors], train_feat.label)

gbm = lgb.train(params,lgb_train,465)
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(output_path + 'piupiu_s1_feat_imp.csv')
preds = gbm.predict(eval_feat[predictors])
eval_feat['piupiu_s1'] = preds

submission = eval_feat[['user_id', 'piupiu_s1']].sort_values('piupiu_s1', ascending=False)
submission[['user_id','piupiu_s1']].to_csv(r'../output/piupiu_s1.csv',index=False)

