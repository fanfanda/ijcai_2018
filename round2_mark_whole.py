import pandas as pd
import datetime
import xgboost as xgb
import numpy as np
from round2_utility import *
from sliding_features import *
import sys
from sklearn.metrics import log_loss
import gc
#复赛没有重复数据，不用去除重复
train_data = pd.read_table("../training_data/round2_train.txt",delim_whitespace=True)
test_data_a = pd.read_table("../training_data/round2_ijcai_18_test_a_20180425.txt", delim_whitespace = True)
test_data_b = pd.read_table("../training_data/round2_ijcai_18_test_b_20180510.txt", delim_whitespace = True)
test_data = test_data_a.append(test_data_b, ignore_index = True)

all_data = train_data.append(test_data, ignore_index = True)
item_category_list = all_data.item_category_list.apply(lambda x: x.split(';'))



print("get_dummies.....")
all_data['first_category'] = item_category_list.apply(lambda x:x[1])

all_data['show_time'] = all_data.context_timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
all_data['weekday'] = all_data.show_time.apply(lambda x:x.weekday())
all_data['hour'] = all_data.show_time.apply(lambda x:x.hour)
all_data['day'] = all_data.show_time.apply(lambda x:x.day)
print('消除当天莫名其妙的-1')
all_data['item_sales_level'] = all_data.groupby(['item_id','day']).item_sales_level.transform('max')
all_features = set(all_data.columns)

# print("add convert.......")
# be_feature_name_convert = ['shop_id','item_brand_id','user_star_level','user_gender_id','user_occupation_id','item_id','user_age_level']
# for i in be_feature_name_convert:
#     sliding_convert_data = online_sliding_be_one_click_buy_convert(all_data,i)
#     all_data = add_sliding_be_one_click_buy_convert(all_data, sliding_convert_data, i)

# print("shop_countU_category")
# all_data['shop_countU_category'] = all_data.groupby(['shop_id']).item_category_list.transform('nunique')
# all_data['shop_countU_item'] = all_data.groupby(['shop_id']).item_id.transform('nunique')
# all_data['good_review_num']= all_data.item_sales_level * all_data.shop_review_positive_rate



# print("user_sliding_shop_score......")
# for i in ['shop_score_service','shop_score_delivery','shop_score_description']:
# 	all_data = user_shop_some_statics(all_data,fe_1=i)
# #4.店铺商品的价格的最大值/最小值/平均值：item_shop_item_price_max/min/avg
# all_data['shop_price_max'] = all_data.groupby('shop_id').item_price_level.transform('max')
# all_data['shop_price_min'] = all_data.groupby('shop_id').item_price_level.transform('min')
# all_data['shop_price_avg'] = all_data.groupby('shop_id').item_price_level.transform('mean')
# all_data['shop_clickuser_avgstar'] = all_data.groupby('shop_id').user_star_level.transform('mean')

# some_features_portition_list = ['item_price_level','item_sales_level','item_collected_level','item_pv_level','shop_review_num_level','shop_star_level']
# for i in some_features_portition_list:
# 	all_data = some_features_portition(all_data, fe_1 = i)
# print("other statics.....")
# all_data['user_cate_rank'] = all_data.groupby(['user_id','item_category_list'])['context_timestamp'].rank(pct=True)
# all_data['user_cate_period'] = all_data.groupby(['user_id','item_category_list']).context_timestamp.transform(lambda x: x.max()-x.min())/60

# all_data['price_sale_rate'] = all_data.item_price_level / all_data.item_sales_level
# all_data['sale_star_rate'] = all_data.item_sales_level / all_data.shop_star_level
# all_data.loc[all_data.item_sales_level == -1,'price_sale_rate'] = -1
# all_data.loc[all_data.item_sales_level == -1,'sale_star_rate'] = -1

drop_stage3_set = all_features - set(['instance_id'])
all_data.drop(list(drop_stage3_set), axis = 1, inplace = True)
all_data.to_csv("../training_data/round2_temp_data/whole_convert_online.csv", index = False)
