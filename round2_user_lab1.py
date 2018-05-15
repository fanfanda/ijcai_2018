import pandas as pd
import datetime
import xgboost as xgb
import numpy as np
from round2_utility import *
import sys
from sklearn.metrics import log_loss
import gc
# train_data = pd.read_table("../training_data/round2_train.txt",delim_whitespace=True)
# #复赛没有重复数据，不用去除重复
# # train_data = train_data.drop_duplicates().reset_index(drop = True)
# test_data_a = pd.read_table("../training_data/round2_ijcai_18_test_a_20180425.txt", delim_whitespace = True)

# all_data = train_data.append(test_data_a, ignore_index = True)

# validate_index = pd.read_table("../training_data/validate_index.csv",sep=',')
user_id_data = "all_data_userid1_b"
all_data = pd.read_csv("../training_data/round2_temp_data/" + user_id_data + ".csv", sep = ',')

all_data['show_time'] = all_data.context_timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
all_data['weekday'] = all_data.show_time.apply(lambda x:x.weekday())
all_data['hour'] = all_data.show_time.apply(lambda x:x.hour)
all_data['day'] = all_data.show_time.apply(lambda x:x.day)



# all_data['is_weekend'] = all_data.weekday.apply(lambda x:1 if x == 0 or x == 6 else 0)


# list_item_category = []
item_category_list = all_data.item_category_list.apply(lambda x: x.split(';'))
# for i in item_category_list:
#     list_item_category += i[1:]
# list_item_category = set(list_item_category)


# print("get_dummies.....")
all_data['first_category'] = item_category_list.apply(lambda x:x[1])
# all_data['second_category'] = item_category_list.apply(lambda x:x[2] if len(x)>2 else np.nan)

all_features = set(all_data.columns)
print('user_sametime_content_counts.....')
all_data['user_same_time_firstcategory_counts'] = all_data.groupby(['user_id','context_timestamp']).item_category_list.transform('nunique')
all_data['user_same_time_shop_counts'] = all_data.groupby(['user_id','context_timestamp']).shop_id.transform('nunique')
all_data['user_same_time_item_counts'] = all_data.groupby(['user_id','context_timestamp']).item_id.transform('nunique')
all_data['user_same_time_brand_counts'] = all_data.groupby(['user_id','context_timestamp']).item_brand_id.transform('nunique')
all_data['sametime_click_times'] = all_data.groupby(['user_id','context_timestamp']).instance_id.transform('nunique')
all_data['context_timestamp'] = all_data['context_timestamp'].astype('str')
print("drop context_timestamp/predict_category_property/item_category_list/item_property_list/context_id")

drop_stage1_set = set(['predict_category_property','item_property_list','context_id'])
all_data.drop(list(drop_stage1_set), axis = 1, inplace = True)
# del item_category_list



all_data_user_id = list(set(all_data.user_id))

unpack_list = []
size = 20
for i in range(size):
    temp_dataset = all_data[all_data.user_id.isin(all_data_user_id[i::size])].reset_index(drop = True)
    unpack_list.append(temp_dataset)
unpack_list = add_features(genes_feature,unpack_list)
unpack_list = add_features(fast_pre_back_click_time,unpack_list)
unpack_list = add_features(fast_pre_back_category_fine_click_time,unpack_list)
unpack_list = add_features(fast_pre_back_item_click_time,unpack_list)
unpack_list = add_features(fast_pre_back_shop_click_time,unpack_list)
unpack_list = add_features(fast_pre_back_brand_click_time,unpack_list)

unpack_list = add_features(fast_statics_of_before_back_click,unpack_list)
unpack_list = add_features(fast_statics_of_item_before_back_click,unpack_list)
unpack_list = add_features(fast_statics_of_kind_item_fine_before_back_click, unpack_list)
unpack_list = add_features(fast_statics_of_brand_before_back_click, unpack_list)
unpack_list = add_features(fast_statics_of_shop_before_back_click, unpack_list)

unpack_list = add_features(calculate_sliding_click_times, unpack_list)
unpack_list = add_features(calculate_sliding_click_shop_times, unpack_list)
unpack_list = add_features(calculate_sliding_click_first_category_fine_times, unpack_list)
unpack_list = add_features(calculate_sliding_click_item_id_times, unpack_list)
unpack_list = add_features(calculate_sliding_click_brand_id_times, unpack_list)


unpack_list = add_features(calculate_before_click_times, unpack_list, args = 3600) #单位是秒
unpack_list = add_features(calculate_before_click_shop_times, unpack_list, args = 3600) #单位是秒
unpack_list = add_features(calculate_before_click_first_category_times, unpack_list, args = 3600) #单位是秒
unpack_list = add_features(calculate_before_click_item_id_times, unpack_list, args = 3600) #单位是秒
unpack_list = add_features(calculate_before_click_brand_id_times, unpack_list, args = 3600) #单位是秒

unpack_list = add_features(calculate_before_click_times, unpack_list, args = 10800) #单位是秒
unpack_list = add_features(calculate_before_click_shop_times, unpack_list, args = 10800) #单位是秒
unpack_list = add_features(calculate_before_click_first_category_times, unpack_list, args = 10800) #单位是秒
unpack_list = add_features(calculate_before_click_item_id_times, unpack_list, args = 10800) #单位是秒
unpack_list = add_features(calculate_before_click_brand_id_times, unpack_list, args = 10800) #单位是秒


# unpack_list = add_features(statics_of_item_before_back_click,unpack_list)
# unpack_list = add_features(statics_of_kind_item_before_back_click, unpack_list)

# dataset1,dataset2,dataset3,dataset4,dataset5,dataset6,dataset7,dataset8 = unpack_list
dataset1 = unpack_list[0]
for i in range(1,size):
    dataset1 = dataset1.append(unpack_list[i],ignore_index = True)
all_data = dataset1

all_data_int = all_data.select_dtypes(include = ['int'])
all_data_convert_int = all_data_int.apply(pd.to_numeric, downcast = 'unsigned')
all_data[all_data_convert_int.columns] = all_data_convert_int
del all_data_int
del all_data_convert_int
gc.collect()

#打乱一下训练顺序
all_data = all_data.sample(frac=1,random_state = 12).reset_index(drop=True)

drop_stage3_set = all_features - (set(['instance_id'])|drop_stage1_set)
all_data.drop(list(drop_stage3_set), axis = 1, inplace = True)
# all_data.drop('first_category', axis = 1, inplace = True)

all_data.to_csv("../training_data/round2_temp_data/genes_rank_feature_" + user_id_data + ".csv", index = False)
