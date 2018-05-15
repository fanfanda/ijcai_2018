import pandas as pd
import datetime
import xgboost as xgb
import numpy as np
from round2_utility import *
import sys
from sklearn.metrics import log_loss
from sliding_features import *
import gc
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

train_data = pd.read_table("../training_data/round2_train.txt",delim_whitespace=True)
test_data_a = pd.read_table("../training_data/round2_ijcai_18_test_a_20180425.txt", delim_whitespace = True)
test_data_b = pd.read_table("../training_data/round2_ijcai_18_test_b_20180510.txt", delim_whitespace = True)
test_data = test_data_a.append(test_data_b, ignore_index = True)

all_data = train_data.append(test_data, ignore_index = True)
print("get_time......")
all_data['show_time'] = all_data.context_timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
all_data['weekday'] = all_data.show_time.apply(lambda x:x.weekday())
all_data['hour'] = all_data.show_time.apply(lambda x:x.hour)
all_data['day'] = all_data.show_time.apply(lambda x:x.day)
print('消除当天莫名其妙的-1')
all_data['item_sales_level'] = all_data.groupby(['item_id','day']).item_sales_level.transform('max')

all_data['is_eleven'] = all_data.day.apply(lambda x:1 if x == 7 else 0)

print("add_new.....")
item_category_list = all_data.item_category_list.apply(lambda x: x.split(';'))
all_data['first_category'] = item_category_list.apply(lambda x:x[1])
all_data['second_category'] = item_category_list.apply(lambda x:x[2] if len(x)>2 else np.nan)


all_data['item_sales_devide_item_pv_level'] = all_data.item_sales_level / (all_data.item_pv_level + 1)


print("drop context_timestamp.....")
drop_1 = set(['context_timestamp','predict_category_property','context_id','item_category_list'])
all_data.drop(drop_1, axis = 1, inplace = True)
del train_data
del item_category_list
gc.collect()

#打乱一下训练顺序
all_data = all_data.sample(frac=1,random_state = 12).reset_index(drop=True)


print("add convert.......")
be_feature_name_convert = ['shop_id','item_brand_id','user_star_level','user_gender_id','user_occupation_id','item_id','user_age_level','item_collected_level','item_pv_level','item_price_level','item_sales_level']
for i in be_feature_name_convert:
    sliding_convert_data = online_sliding_be_one_click_buy_convert(all_data,i)
    all_data = add_sliding_be_one_click_buy_convert(all_data, sliding_convert_data, i)

gc.collect()


print("reduce memory......")
all_data_int = all_data.select_dtypes(include = ['int'])
all_data_convert_int = all_data_int.apply(pd.to_numeric, downcast = 'unsigned')
all_data[all_data_convert_int.columns] = all_data_convert_int

del all_data_int
del all_data_convert_int
gc.collect()

print("load wholedata......")
whole_data = pd.read_csv("../training_data/round2_temp_data/whole_all_data_b.csv", sep = ',')
feature_first_name = ['user_id','user_star_level','user_occupation_id','user_age_level','user_gender_id']
feature_second_name = ['shop_id','item_brand_id','item_city_id','item_id','first_category']
be_feature_name = feature_second_name + feature_first_name


whole_drop = []
for i in be_feature_name:
    whole_drop.append('be_curday_' + i + '_click')
# whole_drop.append('be_curday_item_city_id_click')
for i in feature_first_name:
    for j in feature_second_name:
        whole_drop.append('curday_' + i + '_click_' + j + '_times')
# ['first_category','second_category']

whole_data.drop(whole_drop, axis = 1, inplace = True)
print("reduce memory......")
whole_data_int = whole_data.select_dtypes(include = ['int'])
whole_data_convert_int = whole_data_int.apply(pd.to_numeric, downcast = 'unsigned')
whole_data[whole_data_convert_int.columns] = whole_data_convert_int
del whole_data_int
del whole_data_convert_int
gc.collect()
print("merge wholedata......")
all_data = pd.merge(all_data, whole_data, how = 'left', on = ['instance_id'])
del whole_data
gc.collect()

print("load wholedata2......")
whole_data_2 = pd.read_csv("../training_data/round2_temp_data/whole_all_data_b_v2.csv", sep = ',')
print("merge wholedata2......")
all_data = pd.merge(all_data, whole_data_2, how = 'left', on = ['instance_id'])
del whole_data_2
gc.collect()

print("load pre_back_timeandcounts_all_data_userid1_b.....")
user_data1 = pd.read_csv("../training_data/round2_temp_data/pre_back_timeandcountsall_data_userid1_b.csv", sep = ',')
user_data2 = pd.read_csv("../training_data/round2_temp_data/pre_back_timeandcountsall_data_userid2_b.csv", sep = ',')
user_data = user_data1.append(user_data2, ignore_index = True)
# user_data.drop(['first_category','second_category'], axis = 1, inplace = True)

del user_data1
del user_data2
gc.collect()
print("merge fast_pre_back_click_time_all_data_userid")
all_data = pd.merge(all_data, user_data, how = 'left', on = ['instance_id'])

print("load more_before_all_data_userid.....")
user_data1 = pd.read_csv("../training_data/round2_temp_data/sliding_more_all_data_userid1_b.csv", sep = ',')
user_data2 = pd.read_csv("../training_data/round2_temp_data/sliding_more_all_data_userid2_b.csv", sep = ',')
user_data = user_data1.append(user_data2, ignore_index = True)


del user_data1
del user_data2
gc.collect()
print("merge more_before_all_data_userid")
all_data = pd.merge(all_data, user_data, how = 'left', on = ['instance_id'])



all_data['item_property_list']=all_data.item_property_list.apply(lambda x: ' '.join(x.split(';')))

cv = CountVectorizer(max_features=30)
data_a = cv.fit_transform(all_data['item_property_list'])
data_a = pd.DataFrame(data_a.todense(), columns=['p_'+str(i) for i in range(data_a.shape[1])])
all_data = pd.concat([all_data,data_a], axis=1)

first_category_dumies = pd.get_dummies(all_data.first_category,prefix='kind_item')
second_category_dumies = pd.get_dummies(all_data.second_category,prefix='kind_item')
# first_category_dumies[first_category_dumies == 0] = np.nan
# second_category_dumies[second_category_dumies == 0] = np.nan
# city_dummies = pd.get_dummies(all_data.item_city_id,prefix='city')
# city_dummies[city_dummies == 0] = np.nan
# all_data.drop('item_city_id', axis = 1, inplace = True)

# all_data = all_data.join([city_dummies,first_category_dumies,second_category_dumies])
all_data = all_data.join([first_category_dumies,second_category_dumies])
all_data_int = all_data.select_dtypes(include = ['int'])
all_data_convert_int = all_data_int.apply(pd.to_numeric, downcast = 'unsigned')
all_data[all_data_convert_int.columns] = all_data_convert_int
del all_data_int
del all_data_convert_int
del first_category_dumies
del second_category_dumies
gc.collect()


#开始拆分线下训练集，验证集
print("unpack and restore.....")

new_train_data = all_data[(all_data.is_trade.notnull())&(all_data.day != 6)].reset_index(drop = True)
new_test_data = all_data[all_data.is_trade.isnull()].reset_index(drop = True)


del all_data
gc.collect()
# validata_train_data.to_csv("../training_data/round2_4_25_lab/validata_train_data.csv", index = False)
# validata_test_data.to_csv("../training_data/round2_4_25_lab/validata_test_data.csv", index = False)
# new_train_data.to_csv("../training_data/round2_4_25_lab/all_train_data.csv", index = False)
# new_test_data.to_csv("../training_data/round2_4_25_lab/test_data_a.csv", index = False)

# new_all_data = dataset1.append([dataset2,dataset3,dataset4],ignore_index=True)
print("generate Dmatrix.....")

drop_list = set(['instance_id','is_trade','item_id','user_id','context_id','item_category_list','item_property_list','context_timestamp','predict_category_property','show_time','weekday','day','item_brand_id','first_category','second_category','shop_id','be_curday_all_click'])
drop_list = list(drop_list - drop_1)
# for i, fe_1 in enumerate(be_feature_name_convert):
#     drop_list.append('sliding_2d_be_'+fe_1+'_click')
#     drop_list.append('sliding_2d_be_'+fe_1+'_buy')

# for i in ['user_star_level','user_occupation_id','user_age_level','user_gender_id','item_city_id']:
#     drop_list.append('be_curday_' + i + '_click')
feature_first_name_all = ['user_star_level','user_occupation_id','user_age_level','user_gender_id']
feature_second_name_all = ['shop_id','item_brand_id','item_city_id','item_id','first_category']
be_feature_name_all = feature_first_name_all + feature_second_name_all
# for i in be_feature_name_all:
#     drop_list.append('all_' + i + '_click_times')
# for i in feature_first_name_all:
#     for j in feature_second_name_all:
#         drop_list.append('curday_' + i + '_click_' + j + '_times')

drop_list.append('all_user_age_level_item_city_id_click_times') #rm
drop_list.append('all_user_gender_id_item_city_id_click_times') #rm


era_params={'booster':'gbtree',
	    'objective': 'binary:logistic',
	    'eval_metric':'logloss',
	    # 'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.03,
	    'tree_method':'hist',
	    'seed':0,
        'silent':1,
	    'nthread':16
	    }



best_iteration = 3000

# del model
# del Dtrain_data
# del Dvalidate_data
# gc.collect()
new_train_data_x = new_train_data.drop(drop_list,axis = 1)
new_test_data_x = new_test_data.drop(drop_list,axis = 1)

new_train_data['weight'] = new_train_data.day.apply(lambda x:1 if x == 7 else 0.8)

Dtrain_data = xgb.DMatrix(new_train_data_x,label=new_train_data.is_trade, weight = new_train_data['weight'])
Dtest_data = xgb.DMatrix(new_test_data_x,label=new_test_data.is_trade)

watchlist = [(Dtrain_data,'train')]
model = xgb.train(era_params,Dtrain_data,num_boost_round=best_iteration,evals=watchlist)

result = list(model.predict(Dtest_data))
new_test_data['predicted_score'] = result

# test_data_a_instance =  test_data_a[['instance_id']].reset_index(drop = True)

test_data_b = pd.merge(test_data_b,new_test_data, how = 'left', on = ['instance_id'])

test_data_b[['instance_id','predicted_score']].to_csv("round2_xxx.txt", index = False, sep = ' ')
