import pandas as pd
import numpy as np
import lightgbm as lgb

from datetime import datetime, timedelta
from functools import partial
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF, PCA, TruncatedSVD
import gc
from joblib import dump, load, Parallel, delayed
from tool import *


data_path = '../data/'
cache_path = '../cache/'

sku_basic_info  = pd.read_csv(data_path + "jdata_sku_basic_info.csv")
user_basic_info = pd.read_csv(data_path + "jdata_user_basic_info.csv")
user_action     = pd.read_csv(data_path + "jdata_user_action.csv", parse_dates= ['a_date'])
user_order      = pd.read_csv(data_path + "jdata_user_order.csv",  parse_dates= ['o_date'])
user_comment_score = pd.read_csv(data_path + "jdata_user_comment_score.csv", parse_dates= ['comment_create_tm'])



aim_cates = [30, 101]

print (sku_basic_info.shape)
sku_basic_info = sku_basic_info[sku_basic_info.cate.isin(aim_cates)]
print (sku_basic_info.shape)
print ()

print (user_action.shape)
user_action    = user_action[user_action.sku_id.isin(sku_basic_info.sku_id)].merge(sku_basic_info, 'left', 'sku_id')
print (user_action.shape)
print ()

print (user_order.shape)
user_order     = user_order[user_order.sku_id.isin(sku_basic_info.sku_id)].merge(sku_basic_info, 'left', 'sku_id')
print (user_order.shape)
print ()

print (user_comment_score.shape)
user_comment_score = user_comment_score[user_comment_score.o_id.isin(user_order.o_id)].merge(user_order[['o_id', 'price', 'cate', 'para_1', 'para_2', 'para_3']], 'left', 'o_id')
print (user_comment_score.shape)


user_action['month'] = (user_action.a_date.dt.year-2016) * 12 + user_action.a_date.dt.month
user_order['month']  = (user_order.o_date.dt.year-2016)  * 12 + user_order.o_date.dt.month
user_comment_score['month'] = (user_comment_score.comment_create_tm.dt.year-2016) * 12 + user_comment_score.comment_create_tm.dt.month


user_order_sample_source = user_order[user_order.month.isin([14, 15, 16])]

# user_action_ = user_action[user_action.month < 11]

############################################# 降维 ######################################################
# user_action_ = user_action[user_action['time']<'2017-04-01']
user_action_ = user_action[user_action['a_date']<'2017-04-01']

mapping = {}
for sample in user_action_[['user_id', 'sku_id']].values:
    mapping.setdefault(sample[0], []).append(str(sample[1]))
cate1s = list(mapping.keys())
print(len(cate1s))
cate2_as_sentence = [' '.join(mapping[cate_]) for cate_ in cate1s]
cate2_as_matrix = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=2).fit_transform(cate2_as_sentence)

lda = LDA(n_components=5,
          learning_method='online',
          batch_size=1000,
          n_jobs=40,
          random_state=520)
topics_of_cate1 = lda.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_lda_action" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p1.hdf', 'w')

nmf = NMF(n_components=5,
          random_state=520,
          beta_loss='kullback-leibler',
          solver='mu',
          max_iter=1000,
          alpha=.1,
          l1_ratio=.5)
topics_of_cate1 = nmf.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_nmf_action" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p2.hdf', 'w')

pca = TruncatedSVD(5)
topics_of_cate1 = pca.fit_transform(cate2_as_matrix)
del cate2_as_matrix;
gc.collect()
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_svd_action" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p3.hdf', 'w')

mapping = {}
for sample in user_action_[['user_id', 'para_2']].values:
    mapping.setdefault(sample[0], []).append(str(sample[1]))
cate1s = list(mapping.keys())
print(len(cate1s))
cate2_as_sentence = [' '.join(mapping[cate_]) for cate_ in cate1s]
cate2_as_matrix = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=2).fit_transform(cate2_as_sentence)

lda = LDA(n_components=5,
          learning_method='online',
          batch_size=1000,
          n_jobs=40,
          random_state=520)
topics_of_cate1 = lda.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_lda_action_param2" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p4.hdf', 'w')

nmf = NMF(n_components=5,
          random_state=520,
          beta_loss='kullback-leibler',
          solver='mu',
          max_iter=1000,
          alpha=.1,
          l1_ratio=.5)
topics_of_cate1 = nmf.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_nmf_action_param2" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p5.hdf', 'w')

pca = TruncatedSVD(5)
topics_of_cate1 = pca.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_svd_action_param2" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p6.hdf', 'w')

mapping = {}
for sample in user_action_[['user_id', 'para_3']].values:
    mapping.setdefault(sample[0], []).append(str(sample[1]))
cate1s = list(mapping.keys())
print(len(cate1s))
cate2_as_sentence = [' '.join(mapping[cate_]) for cate_ in cate1s]
cate2_as_matrix = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=2).fit_transform(cate2_as_sentence)

lda = LDA(n_components=5,
          learning_method='online',
          batch_size=1000,
          n_jobs=40,
          random_state=520)
topics_of_cate1 = lda.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_lda_action_param3" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p7.hdf', 'w')

nmf = NMF(n_components=5,
          random_state=520,
          beta_loss='kullback-leibler',
          solver='mu',
          max_iter=1000,
          alpha=.1,
          l1_ratio=.5)
topics_of_cate1 = nmf.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_nmf_action_param3" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p8.hdf', 'w')

pca = TruncatedSVD(5)
topics_of_cate1 = pca.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_svd_action_param3" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p9.hdf', 'w')

# user_order_ = user_order[user_order['time']<'2017-04-01']
user_order_ = user_order[user_order['o_date']<'2017-04-01']

mapping = {}
for sample in user_order_[['user_id', 'sku_id']].values:
    mapping.setdefault(sample[0], []).append(str(sample[1]))
cate1s = list(mapping.keys())
print(len(cate1s))
cate2_as_sentence = [' '.join(mapping[cate_]) for cate_ in cate1s]
cate2_as_matrix = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=2).fit_transform(cate2_as_sentence)

lda = LDA(n_components=5,
          learning_method='online',
          batch_size=1000,
          n_jobs=40,
          random_state=520)
topics_of_cate1 = lda.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1, columns=["%s_%s_lda_order" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p10.hdf', 'w')

nmf = NMF(n_components=5,
          random_state=520,
          beta_loss='kullback-leibler',
          solver='mu',
          max_iter=1000,
          alpha=.1,
          l1_ratio=.5)
topics_of_cate1 = nmf.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1, columns=["%s_%s_nmf_order" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p11.hdf', 'w')

pca = TruncatedSVD(5)
topics_of_cate1 = pca.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1, columns=["%s_%s_svd_order" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p12.hdf', 'w')

mapping = {}
for sample in user_order_[['user_id', 'para_2']].values:
    mapping.setdefault(sample[0], []).append(str(sample[1]))
cate1s = list(mapping.keys())
print(len(cate1s))
cate2_as_sentence = [' '.join(mapping[cate_]) for cate_ in cate1s]
cate2_as_matrix = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=2).fit_transform(cate2_as_sentence)

lda = LDA(n_components=5,
          learning_method='online',
          batch_size=1000,
          n_jobs=40,
          random_state=520)
topics_of_cate1 = lda.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_lda_order_param2" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p13.hdf', 'w')

nmf = NMF(n_components=5,
          random_state=520,
          beta_loss='kullback-leibler',
          solver='mu',
          max_iter=1000,
          alpha=.1,
          l1_ratio=.5)
topics_of_cate1 = nmf.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_nmf_order_param2" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p14.hdf', 'w')

pca = TruncatedSVD(5)
topics_of_cate1 = pca.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_svd_order_param2" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p15.hdf', 'w')

mapping = {}
for sample in user_order_[['user_id', 'para_3']].values:
    mapping.setdefault(sample[0], []).append(str(sample[1]))
cate1s = list(mapping.keys())
print(len(cate1s))
cate2_as_sentence = [' '.join(mapping[cate_]) for cate_ in cate1s]
cate2_as_matrix = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=2).fit_transform(cate2_as_sentence)

lda = LDA(n_components=5,
          learning_method='online',
          batch_size=1000,
          n_jobs=40,
          random_state=520)
topics_of_cate1 = lda.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_lda_order_param3" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p16.hdf', 'w')

nmf = NMF(n_components=5,
          random_state=520,
          beta_loss='kullback-leibler',
          solver='mu',
          max_iter=1000,
          alpha=.1,
          l1_ratio=.5)
topics_of_cate1 = nmf.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_nmf_order_param3" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p17.hdf', 'w')

pca = TruncatedSVD(5)
topics_of_cate1 = pca.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_svd_order_param3" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(cache_path + 'p18.hdf', 'w')
