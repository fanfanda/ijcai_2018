import pandas as pd
import datetime
import xgboost as xgb
import numpy as np
from round2_utility import *
import sys
from sklearn.metrics import log_loss
import gc
#复赛没有重复数据，不用去除重复
train_data = pd.read_table("../training_data/round2_train.txt",delim_whitespace=True)
test_data_a = pd.read_table("../training_data/round2_ijcai_18_test_a_20180425.txt", delim_whitespace = True)
test_data_b = pd.read_table("../training_data/round2_ijcai_18_test_b_20180510.txt", delim_whitespace = True)
test_data = test_data_a.append(test_data_b, ignore_index = True)

all_data = train_data.append(test_data, ignore_index = True)

print('消除当天莫名其妙的-1')
all_data['item_sales_level'] = all_data.groupby(['item_id','day']).item_sales_level.transform('max')

print("get_time......")
all_data['show_time'] = all_data.context_timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
all_data['weekday'] = all_data.show_time.apply(lambda x:x.weekday())
all_data['hour'] = all_data.show_time.apply(lambda x:x.hour)
all_data['day'] = all_data.show_time.apply(lambda x:x.day)

print("get_dummies.....")
item_category_list = all_data.item_category_list.apply(lambda x: x.split(';'))
all_data['first_category'] = item_category_list.apply(lambda x:x[1])
all_data['second_category'] = item_category_list.apply(lambda x:x[2] if len(x)>2 else np.nan)

all_features = set(all_data.columns)

print("drop context_timestamp/predict_category_property/item_category_list/item_property_list/context_id")

drop_stage1_set = set(['context_timestamp','predict_category_property','item_category_list','item_property_list','context_id'])
all_data.drop(list(drop_stage1_set), axis = 1, inplace = True)
del item_category_list
gc.collect()



drop_stage2_set = set(['item_price_level','item_sales_level','item_collected_level','item_pv_level','shop_review_num_level'])
all_data.drop(list(drop_stage2_set), axis = 1, inplace = True)

gc.collect()

# feature_first_name = ['user_id','user_star_level','user_occupation_id','user_age_level','user_gender_id']
feature_first_name = ['user_id','user_star_level','user_occupation_id','user_age_level','user_gender_id']
feature_second_name = ['shop_id','item_brand_id','item_city_id','item_id','first_category']
be_feature_name = feature_second_name + feature_first_name
#单天的点击统计
print("curday statics.....")
for fe in be_feature_name:
	all_data = be_cur_one_click(all_data, fe_1 = fe)

for fe_1 in feature_first_name:
    for fe_2 in feature_second_name:
        all_data = cur_one_to_one_click(all_data, fe_1 = fe_1, fe_2 = fe_2)

gc.collect()
#整体的点击统计
all_data = curday_all_click(all_data, args = None)

feature_first_name_all = ['user_id','user_star_level','user_occupation_id','user_age_level','user_gender_id']
feature_second_name_all = ['shop_id','item_brand_id','item_city_id','item_id','first_category']
be_feature_name_all = feature_first_name_all + feature_second_name_all

# all click times
print("allday statics.....")
for i in be_feature_name_all:
	all_data = be_one_all_click(all_data, fe_1 = i)

for i in feature_first_name_all:
    for j in feature_second_name_all:
        all_data = one_to_one_all_click(all_data, fe_1 = i, fe_2 = j)

# portition
#当天的占比统计
print("portition statics.....")     
be_feature_name_portition = ['shop_id','item_brand_id','item_id','first_category']
for i in be_feature_name_portition:
	all_data = be_one_portition(all_data, fe_1 = i)

drop_stage3_set = all_features - (set(['instance_id'])|drop_stage1_set|drop_stage2_set)
all_data.drop(list(drop_stage3_set), axis = 1, inplace = True)
all_data.to_csv("../training_data/round2_temp_data/whole_all_data_b.csv", index = False)
