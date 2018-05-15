import multiprocessing
import datetime
import numpy as np
import xgboost as xgb
import pandas as pd
import numpy as np
import sys
from bayes_smooth import *
import threading
import queue

def zhoucheng(data, args = None):
    # 0 pre_category 用户产生这条记录前，是否产生过相同category的浏览记录
    # 1 pre_shopid 用户产生这条记录前，是否产生过相同shop的浏览记录
    # 2 pre_itemid 用户产生这条记录前，是否产生过相同item的浏览记录
    data['pre_category_clicked']= data.groupby(['user_id','item_category_list'])['context_timestamp'].rank().apply(lambda x: 1 if x>1 else 0)
    data['pre_shopid_clicked']= data.groupby(['user_id','shop_id'])['context_timestamp'].rank().apply(lambda x: 1 if x>1 else 0)
    data['pre_itemid_clicked']= data.groupby(['user_id','item_id'])['context_timestamp'].rank().apply(lambda x: 1 if x>1 else 0)

    
    return data

def genes_feature(data, args = None):
    #1.--------------------------------------------------------------------------------    
    print('user_time_rank')
    data['user_time_rank'] = data.groupby(['user_id'])['context_timestamp'].rank(pct=True)

    #2.--------------------------------------------------------------------------------       
    print('user_item_time_rank')
    data['user_item_time_rank'] = data.groupby(['user_id','item_id'])['context_timestamp'].rank(pct=True)
    #3.--------------------------------------------------------------------------------    
    print('user_shop_time_rank')
    data['user_shop_time_rank'] = data.groupby(['user_id','shop_id'])['context_timestamp'].rank(pct=True)
    
    #4.--------------------------------------------------------------------------------    
    print('user_catelist_time_rank')
    data['user_catelist_time_rank'] = data.groupby(['user_id','item_category_list'])['context_timestamp'].rank(pct=True)  
    
    #5.--------------------------------------------------------------------------------    
    print('user_cate1_time_rank')
    data['user_cate1_time_rank'] = data.groupby(['user_id','first_category'])['context_timestamp'].rank(pct=True)
    
    print('user_time_rank_today')
    data['user_time_rank_today'] = data.groupby(['user_id','day'])['context_timestamp'].rank(pct=True)        

    print('user_item_time_rank_today')
    data['user_item_time_rank_today'] = data.groupby(['user_id','item_id','day'])['context_timestamp'].rank(pct=True)  

    print('user_shop_time_rank_today')
    data['user_shop_time_rank_today'] = data.groupby(['user_id','shop_id','day'])['context_timestamp'].rank(pct=True)


    print('user_catelist_time_rank_today')
    data['user_catelist_time_rank_today'] = data.groupby(['user_id','item_category_list','day'])['context_timestamp'].rank(pct=True)  


    print('user_cate1_time_rank_today')
    data['user_cate1_time_rank_today'] = data.groupby(['user_id','first_category','day'])['context_timestamp'].rank(pct=True)
    
    return data

def shop_cut(x):
    if x == 1:
        return 10
    elif x >= 0.995:
        return 9
    elif x >= 0.99:
        return 8
    elif x >= 0.985:
        return 7
    elif x >= 0.98:
        return 6
    elif x >= 0.975:
        return 5
    elif x >= 0.97:
        return 4
    elif x >= 0.965:
        return 3
    elif x >= 0.96:
        return 2
    elif x >= 0.955:
        return 1
    elif x >= 0.95:
        return 0
    elif x == -1:
        return -1
# all_data['user_cate_rank'] = all_data.groupby(['user_id','item_category_list'])['context_timestamp'].rank(pct=True)
# print("zhoucheng user_cate_period")
# all_data['user_cate_period'] = all_data.groupby(['user_id','item_category_list']).context_timestamp.transform(lambda x: x.max()-x.min())/60
def user_cate_rank(pd_data, args = None):
    pd_data['user_cate_rank'] = pd_data.groupby(['user_id','item_category_list'])['context_timestamp'].rank(pct=True)
    return pd_data

def user_cate_period(pd_data, args = None):
    pd_data['user_cate_period'] = pd_data.groupby(['user_id','item_category_list']).context_timestamp.transform(lambda x: x.max()-x.min())/60
    return pd_data

def price_sale_rate(pd_data, args = None):
    pd_data['price_sale_rate'] = pd_data.apply(lambda x:-1 if x.item_sales_level == -1 else x.item_price_level/x.item_sales_level, axis = 1)
    return pd_data

def sale_star_rate(pd_data, args = None):
    pd_data['sale_star_rate'] = pd_data.apply(lambda x:-1 if x.item_sales_level == -1 else x.item_sales_level / x.shop_star_level, axis = 1)
    return pd_data
    
def same_time_click_times(pd_data,args = None):
    group_list = ['user_id','show_time']
    temp_data = pd_data[group_list].reset_index(drop = True)
    temp_data['sametime_click_times'] = 1
    temp_data = temp_data.groupby(group_list).agg('sum').reset_index()
    pd_data = pd.merge(pd_data, temp_data, on = group_list, how='left')
    return pd_data
def same_time_shop_click_times(pd_data,args = None):
    group_list = ['user_id','show_time','shop_id']
    temp_data = pd_data[group_list].reset_index(drop = True)
    temp_data['sametime_shop_click_times'] = 1
    temp_data = temp_data.groupby(group_list).agg('sum').reset_index()
    pd_data = pd.merge(pd_data, temp_data, on = group_list, how='left')
    return pd_data

def same_time_kind_click_times(pd_data,args = None):
    group_list = ['user_id','show_time','first_category']
    temp_data = pd_data[group_list].reset_index(drop = True)
    temp_data['sametime_firstcategory_click_times'] = 1
    temp_data = temp_data.groupby(group_list).agg('sum').reset_index()
    pd_data = pd.merge(pd_data, temp_data, on = group_list, how='left')
    return pd_data

def user_shop_some_statics(pd_data, fe_1 = None):
    group_list = [fe_1,'user_id']
    temp_data = pd_data[group_list].reset_index(drop = True)
    # temp_data['sametime_click_times'] = 1
    temp_data_max = temp_data.groupby(['user_id']).agg('max').reset_index()
    temp_data_mean = temp_data.groupby(['user_id']).agg('mean').reset_index()
    temp_data_avg = temp_data.groupby(['user_id']).agg('min').reset_index()
    temp_data_max.rename(columns={fe_1:fe_1+'_max'},inplace=True)
    temp_data_avg.rename(columns={fe_1:fe_1+'_avg'},inplace=True)
    temp_data_mean.rename(columns={fe_1:fe_1+'_mean'},inplace=True)

    pd_data = pd.merge(pd_data, temp_data_max, on = ['user_id'], how='left')
    pd_data = pd.merge(pd_data, temp_data_avg, on = ['user_id'], how='left')
    pd_data = pd.merge(pd_data, temp_data_mean, on = ['user_id'], how='left')

    return pd_data
#按照种类分别来统计这些属性与该类别的平均属性的比值
# some_features_portition_list = ['item_price_level','item_sales_level','item_collected_level','item_pv_level','shop_review_num_level','shop_star_level']
def some_features_portition(pd_data, fe_1 = None):
    group_list = [fe_1,'first_category']
    temp_data = pd_data[group_list].reset_index(drop = True)
    # temp_data = pd_data[group_list]
    # temp_data['sametime_click_times'] = 1
    # temp_data_max = temp_data.groupby(['first_category']).agg('max').reset_index()
    temp_data_mean = temp_data.groupby(['first_category']).agg('mean').reset_index()
    # temp_data_avg = temp_data.groupby(['first_category']).agg('min').reset_index()
    # temp_data_max.rename(columns={fe_1:fe_1+'_max'},inplace=True)
    # temp_data_avg.rename(columns={fe_1:fe_1+'_avg'},inplace=True)
    temp_data_mean.rename(columns={fe_1:fe_1+'_mean'},inplace=True)
    # pd_data = pd.merge(pd_data, temp_data_max, on = ['user_id'], how='left')
    # pd_data = pd.merge(pd_data, temp_data_avg, on = ['user_id'], how='left')
    pd_data = pd.merge(pd_data, temp_data_mean, on = ['first_category'], how='left')
    pd_data['portition_' + fe_1] = pd_data[fe_1] / pd_data[fe_1 + '_mean']
    print("over " + fe_1)
    return pd_data

def one_hot_item_property(pd_data, item_property_dict):
    all_num = len(pd_data)
    count = 0
    item_property_list = pd_data.item_property_list.apply(lambda x:x.split(';'))
    pd_data['all_item_property_list'] = item_property_list
    for i in item_property_dict.keys():
        pd_data['property_' + str(i)] = np.nan
    for i in range(len(pd_data)):
        count += 1
        if count % 5000 == 0:
            print('one_hot:',count,'/',all_num)
        for j in pd_data.loc[i,'all_item_property_list']:
            if j in item_property_dict.keys():
                pd_data.loc[i,'property_' + str(j)] = 1
    return pd_data
def one_hot_pianxiang_item_property(pd_data, item_property_dict, fen_item_property_dict):
    all_num = len(pd_data)
    count = 0
    item_property_list = pd_data.item_property_list.apply(lambda x:x.split(';'))
    pd_data['all_item_property_list'] = item_property_list
    for i in item_property_dict.keys():
        pd_data['gender_' + i] = np.nan
        pd_data['age_' + i] = np.nan
    for i in range(all_num):
        user_gender_id = str(pd_data.loc[i,'user_gender_id'])
        user_age_level = str(pd_data.loc[i,'user_age_level'])
        count += 1
        if count % 5000 == 0:
            print('one_hot:',count,'/',all_num)
        for item_property in pd_data.loc[i,'all_item_property_list']:
            if 'property_' + str(item_property) in item_property_dict.keys():
                if user_gender_id + '_' +str(item_property) in fen_item_property_dict.keys():
                    pd_data.loc[i,'gender_property_' + str(item_property)] = fen_item_property_dict[user_gender_id + '_' +str(item_property)]
                if user_age_level + '_' +str(item_property) in fen_item_property_dict.keys():
                    pd_data.loc[i,'age_property_' + str(item_property)] = fen_item_property_dict[user_age_level + '_' +str(item_property)]
    return pd_data

def be_one_all_click(pd_data, fe_1):
    group_list = [fe_1]
    temp_data = pd_data[group_list].reset_index(drop = True)
    temp_data['all_' + fe_1 + '_click_times'] = 1
    temp_data = temp_data.groupby(group_list).agg('sum').reset_index()
    pd_data = pd.merge(pd_data, temp_data, on = group_list, how='left')
    print("over " + fe_1)
    return pd_data

def one_to_one_all_click(pd_data, fe_1, fe_2):
    group_list = [fe_1,fe_2]
    temp_data = pd_data[group_list].reset_index(drop = True)
    temp_data['all_' + fe_1 + '_' + fe_2 + '_click_times'] = 1
    temp_data = temp_data.groupby(group_list).agg('sum').reset_index()
    pd_data = pd.merge(pd_data, temp_data, on = group_list, how='left')
    print("over " + fe_1 + "_" + fe_2)
    return pd_data

def be_one_portition(pd_data, fe_1):
    pd_data['be_curday_' + fe_1 + '_portition'] = pd_data['be_curday_' + fe_1 + '_click'] / pd_data.be_curday_all_click
    return pd_data
def one_to_one_portition(pd_data, fe_1, fe_2):
    pd_data['all_' + fe_1 + '_to_' + fe_2 + '_portition'] = pd_data['all_' + fe_1 + '_' + fe_2 + '_click_times'] / pd_data['all_' + fe_2 + '_click_times']
    return pd_data
def fan_one_to_one_portition(pd_data, fe_1, fe_2):
    pd_data['all_fan_' + fe_1 + '_to_' + fe_2 + '_portition'] = pd_data['all_' + fe_1 + '_' + fe_2 + '_click_times'] / pd_data['all_' + fe_1 + '_click_times']
    return pd_data

def cur_one_to_one_click(pd_data, fe_1, fe_2):
    group_list = [fe_1,'day',fe_2]
    temp_data = pd_data[group_list].reset_index(drop = True)
    temp_data['curday_' + fe_1 + '_click_' +fe_2+ '_times'] = 1
    temp_data = temp_data.groupby(group_list).agg('sum').reset_index()
    pd_data = pd.merge(pd_data, temp_data, on = group_list, how='left')
    print("over " + fe_1 + "_" + fe_2)
    return pd_data

def add_all_be_features(func, lists, all_data , fe_1 = ""):
    result = []
    pool = multiprocessing.Pool(processes = 10)
    for i in lists:
        result.append(pool.apply_async(func=func,args=(i, all_data, fe_1)))
    pool.close()
    pool.join()
    result = list(map(lambda x:x.get(),result))
    return result
def add_all_two_features(func, lists, all_data , fe_1 = "", fe_2 = ""):
    result = []
    pool = multiprocessing.Pool(processes = 10)
    for i in lists:
        result.append(pool.apply_async(func=func,args=(i, all_data, fe_1, fe_2)))
    pool.close()
    pool.join()
    result = list(map(lambda x:x.get(),result))
    return result

def be_cur_one_click(pd_data, fe_1):
    group_list = ['day',fe_1]
    temp_data = pd_data[group_list].reset_index(drop = True)
    temp_data['be_curday_' + fe_1 + '_click'] = 1
    temp_data = temp_data.groupby(group_list).agg('sum').reset_index()
    pd_data = pd.merge(pd_data, temp_data, on = group_list, how='left')
    print("over " + fe_1)
    return pd_data
#####
def item_click_portition(pd_data,args):
    pd_data['item_click_first_category_portition'] = pd_data.be_curday_item_id_click / pd_data.be_curday_first_category_click
    return pd_data
def shop_click_portition(pd_data,args):
    pd_data['shop_click_portition'] = pd_data.be_curday_shop_id_click / pd_data.be_curday_all_click
    return pd_data

def brand_click_portition(pd_data, args):
    pd_data['brand_click_portition'] = pd_data.be_curday_item_brand_id_click / pd_data.be_curday_all_click
    return pd_data
def curday_all_click(pd_data, args):
    group_list = ['day']
    temp_data = pd_data[group_list].reset_index(drop = True)
    temp_data['be_curday_all_click'] = 1
    temp_data = temp_data.groupby(group_list).agg('sum').reset_index()
    pd_data = pd.merge(pd_data, temp_data, on = group_list, how='left')
    return pd_data

def user_sliding_click(pd_data, args):
    # pd_data = pd_data[pd_data.user_id.isin(pd_data_user_id)]
    temp_group = pd_data.groupby(['user_id'])
    pd_data['user_sliding_click_'+str(args)] = 0
    count = 0
    all_group = len(temp_group)
    for key,group in temp_group:
        for i in group.index:
            cur_time = group.loc[i,'show_time'] 
            pre_cur_time = cur_time - np.timedelta64(args,'m')
            back_cur_time = cur_time + np.timedelta64(args,'m')
            sliding_i_group = group[(group.show_time < back_cur_time)&(group.show_time > pre_cur_time)]
            pd_data.loc[i,'user_sliding_click_'+str(args)] = len(sliding_i_group)
        count += 1
        if count % 5000 == 0:
            print('user_sliding_click:',count,'/',all_group)
    return pd_data

# def multithreading_func(pd_data, args):
#     len(user_)
def user_sliding_click_multithreading(pd_data, args):
    threads = []
    pd_data_user_id = list(set(pd_data.user_id))
    q = queue.Queue()
    # for i in range(4):
    #     temp=threading.Thread(target=user_sliding_click,args=(pd_data, args=(args, pd_data_user_id[i::4]))
    #     threads.append(temp)
    for i in range(4):
        pd_data_temp = pd_data[pd_data.user_id.isin(pd_data_user_id[i::4])].reset_index(drop = True)
        temp=threading.Thread(target=user_sliding_click,args=(pd_data_temp,args,q))
        threads.append(temp)
    for i in threads:
        i.setDaemon(True)
        i.start()
    for i in threads:
        i.join()
    result = []
    while not q.empty():
        result.append(q.get())
    pd_data = result[0]
    for i in range(1,len(result)):
        pd_data = pd_data.append(i,ignore_index = True)
    return pd_data


def user_sliding_kinditem_click(pd_data, args):
    temp_group = pd_data.groupby(['user_id','first_category'])
    pd_data['user_sliding_click_kinditem_'+str(args)] = 0
    count = 0
    all_group = len(temp_group)
    for key,group in temp_group:
        for i in group.index:
            cur_time = group.loc[i,'show_time'] 
            pre_cur_time = cur_time - np.timedelta64(args,'m')
            back_cur_time = cur_time + np.timedelta64(args,'m')
            sliding_i_group = group[(group.show_time < back_cur_time)&(group.show_time > pre_cur_time)]
            pd_data.loc[i,'user_sliding_click_kinditem_'+str(args)] = len(sliding_i_group)
        count += 1
        if count % 5000 == 0:
            print('user_sliding_click_kinditem:',count,'/',all_group)
    return pd_data
def user_sliding_item_click(pd_data, args):
    temp_group = pd_data.groupby(['user_id','item_id'])
    pd_data['user_sliding_click_item_'+str(args)] = 0
    count = 0
    all_group = len(temp_group)
    for key,group in temp_group:
        for i in group.index:
            cur_time = group.loc[i,'show_time'] 
            pre_cur_time = cur_time - np.timedelta64(args,'m')
            back_cur_time = cur_time + np.timedelta64(args,'m')
            sliding_i_group = group[(group.show_time < back_cur_time)&(group.show_time > pre_cur_time)]
            pd_data.loc[i,'user_sliding_click_item_'+str(args)] = len(sliding_i_group)
        count += 1
        if count % 5000 == 0:
            print('user_sliding_click_item:',count,'/',all_group)
    return pd_data



def pre_back_click_time(pd_data, args):
    temp_group = pd_data.groupby(['user_id','day'])
    pd_data['pre_click_time'] = -1
    pd_data['back_click_time'] = -1
    count = 0
    all_group = len(temp_group)
    for key,group in temp_group:
        for i in group.index:
            cur_time = group.loc[i,'show_time']
            before_i_group = group[group.show_time < cur_time]
            after_i_group = group[group.show_time > cur_time]
            if not before_i_group.empty:
                pd_data.loc[i,'pre_click_time'] = (cur_time - max(before_i_group.show_time)).seconds
            if not after_i_group.empty:
                pd_data.loc[i,'back_click_time'] = (min(after_i_group.show_time) - cur_time).seconds
            if len(group[group.show_time == cur_time]) > 1:
                pd_data.loc[i,'pre_click_time'] = 0
                pd_data.loc[i,'back_click_time'] = 0
        count += 1
        if count % 5000 == 0:
            print('pre_back_click_time:',count,'/',all_group)
    return pd_data
#######


def pre_back_category_click_time(pd_data, args):
    temp_group = pd_data.groupby(['user_id','first_category','day'])
    pd_data['pre_category_click_time'] = -1
    pd_data['back_category_click_time'] = -1
    count = 0
    all_group = len(temp_group)
    for key,group in temp_group:
        for i in group.index:
            cur_time = group.loc[i,'show_time']
            before_i_group = group[group.show_time < cur_time]
            after_i_group = group[group.show_time > cur_time]
            if not before_i_group.empty:
                pd_data.loc[i,'pre_category_click_time'] = (cur_time - max(before_i_group.show_time)).seconds
            if not after_i_group.empty:
                pd_data.loc[i,'back_category_click_time'] = (min(after_i_group.show_time) - cur_time).seconds
            if len(group[group.show_time == cur_time]) > 1:
                pd_data.loc[i,'pre_category_click_time'] = 0
                pd_data.loc[i,'back_category_click_time'] = 0
        count += 1
        if count % 5000 == 0:
            print('pre_back_first_category_click_time:',count,'/',all_group)
    return pd_data


def pre_back_item_click_time(pd_data, args):
    temp_group = pd_data.groupby(['user_id','item_id','day'])
    pd_data['pre_item_id_click_time'] = -1
    pd_data['back_item_id_click_time'] = -1
    count = 0
    all_group = len(temp_group)
    for key,group in temp_group:
        for i in group.index:
            cur_time = group.loc[i,'show_time']
            before_i_group = group[group.show_time < cur_time]
            after_i_group = group[group.show_time > cur_time]
            if not before_i_group.empty:
                pd_data.loc[i,'pre_item_id_click_time'] = (cur_time - max(before_i_group.show_time)).seconds
            if not after_i_group.empty:
                pd_data.loc[i,'back_item_id_click_time'] = (min(after_i_group.show_time) - cur_time).seconds
            if len(group[group.show_time == cur_time]) > 1:
                pd_data.loc[i,'pre_item_id_click_time'] = 0
                pd_data.loc[i,'back_item_id_click_time'] = 0
        count += 1
        if count % 5000 == 0:
            print('pre_back_item_id_click_time:',count,'/',all_group)
    return pd_data
def pre_back_shop_click_time(pd_data, args):
    temp_group = pd_data.groupby(['user_id','shop_id','day'])
    pd_data['pre_shop_id_click_time'] = -1
    pd_data['back_shop_id_click_time'] = -1
    count = 0
    all_group = len(temp_group)
    for key,group in temp_group:
        for i in group.index:
            cur_time = group.loc[i,'show_time']
            before_i_group = group[group.show_time < cur_time]
            after_i_group = group[group.show_time > cur_time]
            if not before_i_group.empty:
                pd_data.loc[i,'pre_shop_id_click_time'] = (cur_time - max(before_i_group.show_time)).seconds
            if not after_i_group.empty:
                pd_data.loc[i,'back_shop_id_click_time'] = (min(after_i_group.show_time) - cur_time).seconds
            if len(group[group.show_time == cur_time]) > 1:
                pd_data.loc[i,'pre_shop_id_click_time'] = 0
                pd_data.loc[i,'back_shop_id_click_time'] = 0
        count += 1
        if count % 5000 == 0:
            print('pre_back_shop_id_click_time:',count,'/',all_group)
    return pd_data


def statics_of_before_back_click(pd_data,list_item_category):
    # pd_data['show_time'] = pd_data.context_timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
    temp_group = pd_data.groupby(['user_id'])

    item_category_list = pd_data.item_category_list.apply(lambda x: x.split(';'))
    
    count = 0
    all_group = len(temp_group)
    for key,group in temp_group:
        for i in group.index:
            before_i_group = group[group.show_time < group.loc[i,'show_time']]
            after_i_group = group[group.show_time > group.loc[i,'show_time']]
            pd_data.loc[i,'before_click'] = len(before_i_group)
            pd_data.loc[i,'after_click'] = len(after_i_group)
            for j in before_i_group.index:
                for k in item_category_list[j][1:]:
                    # if np.isnan(pd_data.loc[i,'before_item_'+k]):
                    #     pd_data.loc[i,'before_item_'+k] = 0
                    pd_data.loc[i,'before_item_'+k] += 1
            for j in after_i_group.index:
                for k in item_category_list[j][1:]:
                    # if np.isnan(pd_data.loc[i,'after_item_'+k]):
                    #     pd_data.loc[i,'after_item_'+k] = 0
                    pd_data.loc[i,'after_item_'+k] += 1
        count += 1
        if count % 5000 == 0:
            print('statics_of_before_back_click:',count,'/',all_group)
    return pd_data


    
def statics_of_item_before_back_click(pd_data,args):
    # pd_data['show_time'] = pd_data.context_timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
    temp_group = pd_data.groupby(['user_id','item_id'])
    pd_data['before_click_item_id'] = 0
    pd_data['after_click_item_id'] = 0
    count = 0
    all_group = len(temp_group)
    for key,group in temp_group:
        for i in group.index:
            before_i_group = group[group.show_time < group.loc[i,'show_time']]
            after_i_group = group[group.show_time > group.loc[i,'show_time']]
            pd_data.loc[i,'before_click_item_id'] = len(before_i_group)
            pd_data.loc[i,'after_click_item_id'] = len(after_i_group)
        count += 1
        if count % 5000 == 0:
            print('statics_of_item_id_before_back_click:',count,'/',all_group)
    return pd_data

    

def statics_of_kind_item_before_back_click(pd_data,args):
    # pd_data['show_time'] = pd_data.context_timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
    temp_group = pd_data.groupby(['user_id','first_category'])
    pd_data['before_click_kind_item'] = 0
    pd_data['after_click_kind_item'] = 0
    count = 0
    all_group = len(temp_group)
    for key,group in temp_group:
        for i in group.index:
            before_i_group = group[group.show_time < group.loc[i,'show_time']]
            after_i_group = group[group.show_time > group.loc[i,'show_time']]
            pd_data.loc[i,'before_click_kind_item'] = len(before_i_group)
            pd_data.loc[i,'after_click_kind_item'] = len(after_i_group)
        count += 1
        if count % 5000 == 0:
            print('statics_of_kind_item_before_back_click:',count,'/',all_group)
    return pd_data

def add_hour(pd_data,args):
    # pd_data['show_time'] = pd_data.context_timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
    pd_data['hour'] = pd_data.show_time.apply(lambda x:x.hour)
    return pd_data

def revise_show_time(pd_data,args):
    pd_data['show_time'] = pd_data.context_timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
    return pd_data

def add_weekend(pd_data,args):
    pd_data['is_weekend'] = pd_data.weekday.apply(lambda x:1 if x == 0 or x == 6 else 0)
    return pd_data

# 
def add_features(func, lists, args = None):
    result = []
    pool = multiprocessing.Pool(processes = 20)
    for i in lists:
        result.append(pool.apply_async(func=func,args=(i,args)))
    pool.close()
    pool.join()
    result = list(map(lambda x:x.get(),result))
    return result
def add_combine_features(func, lists, fe_1 = "" , fe_2 = ""):
    result = []
    pool = multiprocessing.Pool(processes = 10)
    for i in lists:
        result.append(pool.apply_async(func=func,args=(i, fe_1, fe_2)))
    pool.close()
    pool.join()
    result = list(map(lambda x:x.get(),result))
    return result
def add_three_level_feature(func, lists, fe_1 = "", fe_2 = "", fe_3 = ""):
    result = []
    pool = multiprocessing.Pool(processes = 10)
    for i in lists:
        result.append(pool.apply_async(func=func,args=(i, fe_1, fe_2, fe_3)))
    pool.close()
    pool.join()
    result = list(map(lambda x:x.get(),result))
    return result

def get_Dmartix(drop_list,train_data,validate_list=[],result = False):
    train_data_x = train_data.drop(drop_list,axis=1)
    Dtrain_data = xgb.DMatrix(train_data_x,label=train_data.is_trade)
    len_validate = len(validate_list)
    temp_list = [0] * len_validate
    for i in range(len_validate):
        if result:
            temp_list[i] = validate_list[i].drop(drop_list + ['result'],axis=1)
        else:
            temp_list[i] = validate_list[i].drop(drop_list, axis=1)
        temp_list[i] = xgb.DMatrix(temp_list[i], label=validate_list[i].is_trade)
    
    return Dtrain_data,temp_list

def restore(path = "",datalist = []):
    for i in range(len(datalist)):
        if i < 5:
            datalist[i].to_csv(path + "lab_dataset" + str(i+1) + ".csv", index = False)
        else:
            datalist[i].to_csv(path + "lab_validate_data" + str(i-4) + ".csv", index = False)
def new_restore(path = "",datalist = []):
    datalist[0].to_csv(path + "train_data.csv", index = False)
    datalist[1].to_csv(path + "validata_data.csv", index = False)


def load_data(path = ""):
    dataset1 = pd.read_csv(path + "lab_dataset1.csv",sep=',')
    dataset2 = pd.read_csv(path + "lab_dataset2.csv",sep=',')
    dataset3 = pd.read_csv(path + "lab_dataset3.csv",sep=',')
    dataset4 = pd.read_csv(path + "lab_dataset4.csv",sep=',')
    dataset5 = pd.read_csv(path + "lab_dataset5.csv",sep=',')
    validate_data1 = pd.read_csv(path + "lab_validate_data1.csv",sep=',')
    validate_data2 = pd.read_csv(path + "lab_validate_data2.csv",sep=',')
    validate_data3 = pd.read_csv(path + "lab_validate_data3.csv",sep=',')
    validate_data4 = pd.read_csv(path + "lab_validate_data4.csv",sep=',')
    validate_data5 = pd.read_csv(path + "lab_validate_data5.csv",sep=',')
    return dataset1,dataset2,dataset3,dataset4,dataset5,validate_data1,validate_data2,validate_data3,validate_data4,validate_data5

def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep = True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep = True)
    usage_mb = usage_b / 1024**2
    return "{:03.2f} MB".format(usage_mb)

params={'booster':'gbtree',
	    'objective': 'binary:logistic',
	    'eval_metric':'logloss',
	    # 'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'hist',
	    'seed':0,
        'silent':1,
	    'nthread':10
	    }

#fast statistic

def fast_pre_back_click_time(pd_data, args):
    pd_data['pre_click_time'] = pd_data.sort_values('context_timestamp').groupby(['user_id','day']).context_timestamp.agg('diff')
    pd_data['back_click_time'] = pd_data.sort_values('context_timestamp').groupby(['user_id','day']).pre_click_time.transform(lambda x: x.shift(periods = -1))
    
    pd_data['pre_click_time'].fillna(-1,inplace = True)
    pd_data['back_click_time'].fillna(-1,inplace = True)
    
    pd_data['pre_click_time'] = pd_data.apply(lambda x:0 if x.sametime_click_times != 1 else x.pre_click_time, axis = 1)
    pd_data['back_click_time'] = pd_data.apply(lambda x:0 if x.sametime_click_times != 1 else x.back_click_time, axis = 1)

    pd_data['pre_click_time'] = pd_data['pre_click_time'].astype('int')
    pd_data['back_click_time'] = pd_data['back_click_time'].astype('int')
    return pd_data
def fast_pre_back_category_click_time(pd_data, args):
    pd_data['pre_category_click_time'] = pd_data.sort_values('context_timestamp').groupby(['user_id','first_category','day']).context_timestamp.agg('diff')
    pd_data['back_category_click_time'] = pd_data.sort_values('context_timestamp').groupby(['user_id','first_category','day']).pre_category_click_time.transform(lambda x: x.shift(periods = -1))

    pd_data['pre_category_click_time'].fillna(-1,inplace = True)
    pd_data['back_category_click_time'].fillna(-1,inplace = True)

    pd_data['pre_category_click_time'] = pd_data.apply(lambda x:0 if x.user_same_time_firstcategory_counts != 1 else x.pre_category_click_time, axis = 1)
    pd_data['back_category_click_time'] = pd_data.apply(lambda x:0 if x.user_same_time_firstcategory_counts != 1 else x.back_category_click_time, axis = 1)

    pd_data['pre_category_click_time'] = pd_data['pre_category_click_time'].astype('int')
    pd_data['back_category_click_time'] = pd_data['back_category_click_time'].astype('int')
    return pd_data
def fast_pre_back_category_fine_click_time(pd_data, args):
    pd_data['pre_category_click_time'] = pd_data.sort_values('context_timestamp').groupby(['user_id','item_category_list','day']).context_timestamp.agg('diff')
    pd_data['back_category_click_time'] = pd_data.sort_values('context_timestamp').groupby(['user_id','item_category_list','day']).pre_category_click_time.transform(lambda x: x.shift(periods = -1))

    pd_data['pre_category_click_time'].fillna(-1,inplace = True)
    pd_data['back_category_click_time'].fillna(-1,inplace = True)

    pd_data['pre_category_click_time'] = pd_data.apply(lambda x:0 if x.user_same_time_firstcategory_counts != 1 else x.pre_category_click_time, axis = 1)
    pd_data['back_category_click_time'] = pd_data.apply(lambda x:0 if x.user_same_time_firstcategory_counts != 1 else x.back_category_click_time, axis = 1)

    pd_data['pre_category_click_time'] = pd_data['pre_category_click_time'].astype('int')
    pd_data['back_category_click_time'] = pd_data['back_category_click_time'].astype('int')
    return pd_data

def fast_pre_back_item_click_time(pd_data, args):
    pd_data['pre_item_id_click_time'] = pd_data.sort_values('context_timestamp').groupby(['user_id','item_id','day']).context_timestamp.agg('diff')
    pd_data['back_item_id_click_time'] = pd_data.sort_values('context_timestamp').groupby(['user_id','item_id','day']).pre_item_id_click_time.transform(lambda x: x.shift(periods = -1))

    pd_data['pre_item_id_click_time'].fillna(-1,inplace = True)
    pd_data['back_item_id_click_time'].fillna(-1,inplace = True)

    pd_data['pre_item_id_click_time'] = pd_data['pre_item_id_click_time'].astype('int')
    pd_data['back_item_id_click_time'] = pd_data['back_item_id_click_time'].astype('int')
    return pd_data
def fast_pre_back_shop_click_time(pd_data, args):
    pd_data['pre_shop_id_click_time'] = pd_data.sort_values('context_timestamp').groupby(['user_id','shop_id','day']).context_timestamp.agg('diff')
    pd_data['back_shop_id_click_time'] = pd_data.sort_values('context_timestamp').groupby(['user_id','shop_id','day']).pre_shop_id_click_time.transform(lambda x: x.shift(periods = -1))

    pd_data['pre_shop_id_click_time'].fillna(-1,inplace = True)
    pd_data['back_shop_id_click_time'].fillna(-1,inplace = True)

    pd_data['pre_shop_id_click_time'] = pd_data.apply(lambda x:0 if x.user_same_time_shop_counts != 1 else x.pre_shop_id_click_time, axis = 1)
    pd_data['back_shop_id_click_time'] = pd_data.apply(lambda x:0 if x.user_same_time_shop_counts != 1 else x.back_shop_id_click_time, axis = 1)

    pd_data['pre_shop_id_click_time'] = pd_data['pre_shop_id_click_time'].astype('int')
    pd_data['back_shop_id_click_time'] = pd_data['back_shop_id_click_time'].astype('int')
    return pd_data
def fast_pre_back_brand_click_time(pd_data, args):
    pd_data['pre_item_brand_id_click_time'] = pd_data.sort_values('context_timestamp').groupby(['user_id','item_brand_id','day']).context_timestamp.agg('diff')
    pd_data['back_item_brand_id_click_time'] = pd_data.sort_values('context_timestamp').groupby(['user_id','item_brand_id','day']).pre_item_brand_id_click_time.transform(lambda x: x.shift(periods = -1))

    pd_data['pre_item_brand_id_click_time'].fillna(-1,inplace = True)
    pd_data['back_item_brand_id_click_time'].fillna(-1,inplace = True)

    pd_data['pre_item_brand_id_click_time'] = pd_data.apply(lambda x:0 if x.user_same_time_brand_counts != 1 else x.pre_item_brand_id_click_time, axis = 1)
    pd_data['back_item_brand_id_click_time'] = pd_data.apply(lambda x:0 if x.user_same_time_brand_counts != 1 else x.back_item_brand_id_click_time, axis = 1)

    pd_data['pre_item_brand_id_click_time'] = pd_data['pre_item_brand_id_click_time'].astype('int')
    pd_data['back_item_brand_id_click_time'] = pd_data['back_item_brand_id_click_time'].astype('int')
    return pd_data
def fast_statics_of_before_back_click(pd_data, list_item_category):
    print('statics_of_before_back_click')
    pd_data['before_click'] = pd_data.groupby(['user_id','day'])['context_timestamp'].rank() - 1
    pd_data['after_click'] = pd_data.groupby(['user_id','day'])['context_timestamp'].rank(ascending=False) - 1
    
    pd_data['before_click'] = pd_data['before_click'].astype('int')
    pd_data['after_click'] = pd_data['after_click'].astype('int')
    return pd_data
def fast_statics_of_item_before_back_click(pd_data,args):
    print('statics_of_item_id_before_back_click')
    pd_data['before_click_item_id'] = pd_data.groupby(['user_id','item_id','day'])['context_timestamp'].rank() - 1
    pd_data['after_click_item_id'] = pd_data.groupby(['user_id','item_id','day'])['context_timestamp'].rank(ascending=False) - 1
    
    pd_data['before_click_item_id'] = pd_data['before_click_item_id'].astype('int')
    pd_data['after_click_item_id'] = pd_data['after_click_item_id'].astype('int')
    return pd_data

def fast_statics_of_kind_item_before_back_click(pd_data,args):
    print('statics_of_kind_item_before_back_click')
    pd_data['before_click_kind_item'] = pd_data.groupby(['user_id','first_category','day'])['context_timestamp'].rank() - 1
    pd_data['after_click_kind_item'] = pd_data.groupby(['user_id','first_category','day'])['context_timestamp'].rank(ascending=False) - 1
    
    pd_data['before_click_kind_item'] = pd_data['before_click_kind_item'].astype('int')
    pd_data['after_click_kind_item'] = pd_data['after_click_kind_item'].astype('int')
    return pd_data
def fast_statics_of_kind_item_fine_before_back_click(pd_data,args):
    print('statics_of_kind_item_before_back_click')
    pd_data['before_click_kind_item'] = pd_data.groupby(['user_id','item_category_list','day'])['context_timestamp'].rank() - 1
    pd_data['after_click_kind_item'] = pd_data.groupby(['user_id','item_category_list','day'])['context_timestamp'].rank(ascending=False) - 1
    
    pd_data['before_click_kind_item'] = pd_data['before_click_kind_item'].astype('int')
    pd_data['after_click_kind_item'] = pd_data['after_click_kind_item'].astype('int')
    return pd_data
    
def fast_statics_of_brand_before_back_click(pd_data,args):
    print('statics_of_brand_before_back_click')
    pd_data['before_click_brand_item'] = pd_data.groupby(['user_id','item_brand_id','day'])['context_timestamp'].rank() - 1
    pd_data['after_click_brand_item'] = pd_data.groupby(['user_id','item_brand_id','day'])['context_timestamp'].rank(ascending=False) - 1
    
    pd_data['before_click_brand_item'] = pd_data['before_click_brand_item'].astype('int')
    pd_data['after_click_brand_item'] = pd_data['after_click_brand_item'].astype('int')
    return pd_data
def fast_statics_of_shop_before_back_click(pd_data,args):
    print('statics_of_shop_before_back_click')
    pd_data['before_click_shop_item'] = pd_data.groupby(['user_id','shop_id','day'])['context_timestamp'].rank() - 1
    pd_data['after_click_shop_item'] = pd_data.groupby(['user_id','shop_id','day'])['context_timestamp'].rank(ascending=False) - 1
    
    pd_data['before_click_shop_item'] = pd_data['before_click_shop_item'].astype('int')
    pd_data['after_click_shop_item'] = pd_data['after_click_shop_item'].astype('int')
    return pd_data

def fast_statics_of_before_back_click_day(pd_data, args):
    print('statics_of_before_back_click')
    pd_data['user_times_last'] = pd_data.groupby(['user_id','day'])['context_timestamp'].rank() - 1
    pd_data['user_times_next'] = pd_data.groupby(['user_id','day'])['context_timestamp'].rank(ascending=False) - 1
    
    pd_data['user_times_last'] = pd_data['user_times_last'].astype('int')
    pd_data['user_times_next'] = pd_data['user_times_next'].astype('int')
    return pd_data
def fast_statics_of_item_before_back_click_day(pd_data,args):
    print('statics_of_item_id_before_back_click')
    pd_data['user_item_times_last'] = pd_data.groupby(['user_id','item_id','day'])['context_timestamp'].rank() - 1
    pd_data['user_item_times_next'] = pd_data.groupby(['user_id','item_id','day'])['context_timestamp'].rank(ascending=False) - 1
    
    pd_data['user_item_times_last'] = pd_data['user_item_times_last'].astype('int')
    pd_data['user_item_times_next'] = pd_data['user_item_times_next'].astype('int')
    return pd_data
def fast_statics_of_shop_before_back_click_day(pd_data,args):
    print('statics_of_item_id_before_back_click')
    pd_data['user_shop_times_last'] = pd_data.groupby(['user_id','shop_id','day'])['context_timestamp'].rank() - 1
    pd_data['user_shop_times_next'] = pd_data.groupby(['user_id','shop_id','day'])['context_timestamp'].rank(ascending=False) - 1
    
    pd_data['user_shop_times_last'] = pd_data['user_shop_times_last'].astype('int')
    pd_data['user_shop_times_next'] = pd_data['user_shop_times_next'].astype('int')
    return pd_data

def before_times(x):
    s = x.split(':')
    t = int(s[0])
    result = 0
    for i in range(1,len(s)):
        if 0 < (t - int(s[i])):
            if (t - int(s[i])) <= 900:
                result += 1
        else:
            return result
    return result

def after_times(x):
    s = x.split(':')
    t = int(s[0])
    result = 0
    for i in range(len(s)-1,0,-1):
        if 0 < (int(s[i]) - t):
            if (int(s[i]) - t) <= 900:
                result += 1
        else:
            return result
    return result

def calculate_sliding_click_times(data,args = None):
    # a = data[['instance_id','user_id','context_timestamp']].copy()
    data = data.sort_values(['user_id','context_timestamp'])
    data.context_timestamp = data.context_timestamp.astype('str')
    time_click = data.groupby('user_id')['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index().rename(columns = {'context_timestamp':'time_click'})
    data = pd.merge(data,time_click,'left',on = ['user_id'])
    data.time_click = data.context_timestamp + ':' + data.time_click
    data['before_15_times'] = data.time_click.apply(before_times)
    data['after_15_times'] = data.time_click.apply(after_times)
    data['sliding_15_click_times'] = data['before_15_times'] + data['after_15_times'] + data.sametime_click_times
    data.context_timestamp = data.context_timestamp.astype('int')
    # return a[['instance_id','user_id','context_timestamp','sliding_15_click_times']]
    data.drop('time_click', axis = 1, inplace = True)
    return data

def calculate_sliding_click_shop_times(data,args = None):
    # a = data[['instance_id','user_id','context_timestamp']].copy()
    data = data.sort_values(['user_id','context_timestamp'])
    data.context_timestamp = data.context_timestamp.astype('str')
    
    time_click = data.groupby(['user_id','shop_id'])['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index().rename(columns = {'context_timestamp':'time_click'})
    data = pd.merge(data,time_click,'left',on = ['user_id','shop_id'])
    data.time_click = data.context_timestamp + ':' + data.time_click
    data['shop_15_before_times'] = data.time_click.apply(before_times)
    data['shop_15_after_times'] = data.time_click.apply(after_times)
    data['shop_15_sliding_click_times'] = data['shop_15_before_times'] + data['shop_15_after_times'] + data.user_same_time_shop_counts
    # return a[['instance_id','user_id','context_timestamp','sliding_15_click_times']]
    data.context_timestamp = data.context_timestamp.astype('int')
    data.drop('time_click', axis = 1, inplace = True)
    return data
def calculate_sliding_click_first_category_times(data,args = None):
    # a = data[['instance_id','user_id','context_timestamp']].copy()
    data = data.sort_values(['user_id','context_timestamp'])
    data.context_timestamp = data.context_timestamp.astype('str')
    time_click = data.groupby(['user_id','first_category'])['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index().rename(columns = {'context_timestamp':'time_click'})
    data = pd.merge(data,time_click,'left',on = ['user_id','first_category'])
    data.time_click = data.context_timestamp + ':' + data.time_click
    data['first_category_15_before_times'] = data.time_click.apply(before_times)
    data['first_category_15_after_times'] = data.time_click.apply(after_times)
    data['first_category_15_sliding_click_times'] = data['first_category_15_before_times'] + data['first_category_15_after_times'] + data.user_same_time_firstcategory_counts
    data.context_timestamp = data.context_timestamp.astype('int')
    data.drop('time_click', axis = 1, inplace = True)
    return data

def calculate_sliding_click_first_category_fine_times(data,args = None):
    # a = data[['instance_id','user_id','context_timestamp']].copy()
    data = data.sort_values(['user_id','context_timestamp'])
    data.context_timestamp = data.context_timestamp.astype('str')
    time_click = data.groupby(['user_id','item_category_list'])['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index().rename(columns = {'context_timestamp':'time_click'})
    data = pd.merge(data,time_click,'left',on = ['user_id','item_category_list'])
    data.time_click = data.context_timestamp + ':' + data.time_click
    data['first_category_15_before_times'] = data.time_click.apply(before_times)
    data['first_category_15_after_times'] = data.time_click.apply(after_times)
    data['first_category_15_sliding_click_times'] = data['first_category_15_before_times'] + data['first_category_15_after_times'] + data.user_same_time_firstcategory_counts
    data.context_timestamp = data.context_timestamp.astype('int')
    data.drop('time_click', axis = 1, inplace = True)
    return data

def calculate_sliding_click_item_id_times(data,args = None):
    # a = data[['instance_id','user_id','context_timestamp']].copy()
    data = data.sort_values(['user_id','context_timestamp'])
    data.context_timestamp = data.context_timestamp.astype('str')
    time_click = data.groupby(['user_id','item_id'])['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index().rename(columns = {'context_timestamp':'time_click'})
    data = pd.merge(data,time_click,'left',on = ['user_id','item_id'])
    data.time_click = data.context_timestamp + ':' + data.time_click
    data['item_id_15_before_times'] = data.time_click.apply(before_times)
    data['item_id_15_after_times'] = data.time_click.apply(after_times)
    data['item_id_15_sliding_click_times'] = data['item_id_15_before_times'] + data['item_id_15_after_times'] + data.user_same_time_item_counts
    # return a[['instance_id','user_id','context_timestamp','sliding_15_click_times']]
    data.context_timestamp = data.context_timestamp.astype('int')
    data.drop('time_click', axis = 1, inplace = True)
    return data
def calculate_sliding_click_brand_id_times(data,args = None):
    # a = data[['instance_id','user_id','context_timestamp']].copy()
    data = data.sort_values(['user_id','context_timestamp'])
    data.context_timestamp = data.context_timestamp.astype('str')
    time_click = data.groupby(['user_id','item_brand_id'])['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index().rename(columns = {'context_timestamp':'time_click'})
    data = pd.merge(data,time_click,'left',on = ['user_id','item_brand_id'])
    data.time_click = data.context_timestamp + ':' + data.time_click
    data['item_brand_id_15_before_times'] = data.time_click.apply(before_times)
    data['item_brand_id_15_after_times'] = data.time_click.apply(after_times)
    data['item_brand_id_15_sliding_click_times'] = data['item_brand_id_15_before_times'] + data['item_brand_id_15_after_times'] + data.user_same_time_brand_counts
    # return a[['instance_id','user_id','context_timestamp','sliding_15_click_times']]
    data.context_timestamp = data.context_timestamp.astype('int')
    data.drop('time_click', axis = 1, inplace = True)
    return data
### add
def before_times_time_parameter(x, args):
    s = x.split(':')
    t = int(s[0])
    result = 0
    for i in range(1,len(s)):
        if 0 < (t - int(s[i])):
            if (t - int(s[i])) <= args:
                result += 1
        else:
            return result
    return result
    
def calculate_before_click_times(data,args = None):
    # a = data[['instance_id','user_id','context_timestamp']].copy()
    data = data.sort_values(['user_id','context_timestamp'])
    data.context_timestamp = data.context_timestamp.astype('str')
    time_click = data.groupby('user_id')['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index().rename(columns = {'context_timestamp':'time_click'})
    data = pd.merge(data,time_click,'left',on = ['user_id'])
    data.time_click = data.context_timestamp + ':' + data.time_click
    data['before_'+ str(args) +'_times'] = data.time_click.apply(lambda x:before_times_time_parameter(x, args))
    # data['after_15_times'] = data.time_click.apply(after_times)
    # data['sliding_15_click_times'] = data['before_15_times'] + data['after_15_times'] + data.sametime_click_times
    data.context_timestamp = data.context_timestamp.astype('int')

    data.drop('time_click', axis = 1, inplace = True)
    return data

def calculate_before_click_shop_times(data,args = None):
    # a = data[['instance_id','user_id','context_timestamp']].copy()
    data = data.sort_values(['user_id','context_timestamp'])
    data.context_timestamp = data.context_timestamp.astype('str')
    
    time_click = data.groupby(['user_id','shop_id'])['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index().rename(columns = {'context_timestamp':'time_click'})
    data = pd.merge(data,time_click,'left',on = ['user_id','shop_id'])
    data.time_click = data.context_timestamp + ':' + data.time_click
    data['shop_' + str(args) + '_before_times'] = data.time_click.apply(lambda x:before_times_time_parameter(x, args))
    # data['shop_15_after_times'] = data.time_click.apply(after_times)
    # data['shop_15_sliding_click_times'] = data['shop_15_before_times'] + data['shop_15_after_times'] + data.user_same_time_shop_counts

    data.context_timestamp = data.context_timestamp.astype('int')
    data.drop('time_click', axis = 1, inplace = True)
    return data

def calculate_before_click_first_category_times(data,args = None):
    # a = data[['instance_id','user_id','context_timestamp']].copy()
    data = data.sort_values(['user_id','context_timestamp'])
    data.context_timestamp = data.context_timestamp.astype('str')
    time_click = data.groupby(['user_id','item_category_list'])['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index().rename(columns = {'context_timestamp':'time_click'})
    data = pd.merge(data,time_click,'left',on = ['user_id','item_category_list'])
    data.time_click = data.context_timestamp + ':' + data.time_click
    data['first_category_' + str(args) + '_before_times'] = data.time_click.apply(lambda x:before_times_time_parameter(x, args))
    # data['first_category_15_after_times'] = data.time_click.apply(after_times)
    # data['first_category_15_sliding_click_times'] = data['first_category_15_before_times'] + data['first_category_15_after_times'] + data.user_same_time_firstcategory_counts
    data.context_timestamp = data.context_timestamp.astype('int')
    data.drop('time_click', axis = 1, inplace = True)
    return data

def calculate_before_click_item_id_times(data,args = None):
    # a = data[['instance_id','user_id','context_timestamp']].copy()
    data = data.sort_values(['user_id','context_timestamp'])
    data.context_timestamp = data.context_timestamp.astype('str')
    time_click = data.groupby(['user_id','item_id'])['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index().rename(columns = {'context_timestamp':'time_click'})
    data = pd.merge(data,time_click,'left',on = ['user_id','item_id'])
    data.time_click = data.context_timestamp + ':' + data.time_click
    data['item_id_' + str(args) + '_before_times'] = data.time_click.apply(lambda x:before_times_time_parameter(x, args))
    # data['item_id_15_after_times'] = data.time_click.apply(after_times)
    # data['item_id_15_sliding_click_times'] = data['item_id_15_before_times'] + data['item_id_15_after_times'] + data.user_same_time_item_counts

    data.context_timestamp = data.context_timestamp.astype('int')
    data.drop('time_click', axis = 1, inplace = True)
    return data
def calculate_before_click_brand_id_times(data,args = None):
    # a = data[['instance_id','user_id','context_timestamp']].copy()
    data = data.sort_values(['user_id','context_timestamp'])
    data.context_timestamp = data.context_timestamp.astype('str')
    time_click = data.groupby(['user_id','item_brand_id'])['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index().rename(columns = {'context_timestamp':'time_click'})
    data = pd.merge(data,time_click,'left',on = ['user_id','item_brand_id'])
    data.time_click = data.context_timestamp + ':' + data.time_click
    data['item_brand_id_' + str(args) + '_before_times'] = data.time_click.apply(lambda x:before_times_time_parameter(x, args))
    # data['item_brand_id_15_after_times'] = data.time_click.apply(after_times)
    # data['item_brand_id_15_sliding_click_times'] = data['item_brand_id_15_before_times'] + data['item_brand_id_15_after_times'] + data.user_same_time_brand_counts
    data.context_timestamp = data.context_timestamp.astype('int')
    data.drop('time_click', axis = 1, inplace = True)
    return data

def fast_statics_of_is_last_click(pd_data, list_item_category):
    print('statics_of_before_back_click')
    pd_data['before_click'] = pd_data.groupby(['user_id'])['context_timestamp'].rank() - 1
    pd_data['after_click'] = pd_data.groupby(['user_id'])['context_timestamp'].rank(ascending=False) - 1
    
    pd_data['before_click'] = pd_data['before_click'].astype('int')
    pd_data['after_click'] = pd_data['after_click'].astype('int')
    return pd_data