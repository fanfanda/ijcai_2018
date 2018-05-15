import pandas as pd
from bayes_smooth import *
from collections import defaultdict
import multiprocessing
import numpy as np
import sys

def helper_sliding_beconvert(x,y,all_x,all_y,k):
    if k < 12:
        if all_y - y == 0:
            return -1
        else:
            return (all_x - x) / (all_y - y)
    else:
        if all_y == 0 or np.isnan(all_y):
            return -1
        else:
            return all_x / all_y

def helper_sliding_beconvert1(x,y,all_x,all_y,k):
    if k < 11:
        if all_y - y == 0:
            return -1
        else:
            return (all_x - x) / (all_y - y)
    else:
        if all_y == 0 or np.isnan(all_y):
            return -1
        else:
            return all_x / all_y

def online_sliding_be_one_click_buy_convert(all_data, first_fe):
    group_list = [first_fe]
    
    iter_list = [7,6,5,4,3,2,1,31]
    # iter_list = [0,6,5,4,3]
    iter_range = list(set(iter_list))
    iter_range.sort(key = iter_list.index)
        
    for i in iter_range:
        sliding_temp = all_data[(all_data.day.isin([i]))&(all_data.hour < 12)][group_list+['is_trade']].reset_index(drop = True)
        sliding_temp['sliding_2d_be_'+first_fe+'_click'] = 1
        sliding_temp.rename(columns={'is_trade': 'sliding_2d_be_' + first_fe + '_buy'}, inplace=True)
        sliding_temp = sliding_temp.groupby(group_list).agg('sum').reset_index()
        
        sliding_temp_hour = all_data[all_data.day.isin([i])][group_list+['hour','is_trade']].reset_index(drop = True)
        sliding_temp_hour['sliding_2d_be_hour'+first_fe+'_click'] = 1
        sliding_temp_hour.rename(columns={'is_trade': 'sliding_2d_be_hour' + first_fe + '_buy'}, inplace=True)

        sliding_temp_hour = sliding_temp_hour.groupby(group_list+['hour']).agg('sum').reset_index()
        sliding_temp_hour = pd.merge(sliding_temp_hour,sliding_temp,how = 'left', on = first_fe)
        sliding_temp_hour['be_' + first_fe + '_convert'] = sliding_temp_hour.apply(lambda x:helper_sliding_beconvert(x['sliding_2d_be_hour' + first_fe + '_buy'],x['sliding_2d_be_hour'+first_fe+'_click'],x['sliding_2d_be_' + first_fe + '_buy'],x['sliding_2d_be_'+first_fe+'_click'],x.hour),axis = 1)
        
        sliding_temp_hour['day'] = i
        
        sliding_temp_hour = sliding_temp_hour[['day',first_fe,'hour','be_' + first_fe + '_convert']].reset_index(drop = True)
        if i == 7:
            new_all_data = sliding_temp_hour.reset_index(drop = True)
        else:
            new_all_data = new_all_data.append(sliding_temp_hour,ignore_index = True)
    print("over.....")
    return new_all_data


def sliding_be_one_click_buy_convert(all_data, first_fe):
    group_list = [first_fe]
    
    iter_list = [7,6,5,4,3,2,1,31]
    # iter_list = [7]
    # iter_list = [0,6,5,4,3]
    iter_range = list(set(iter_list))
    iter_range.sort(key = iter_list.index)
        
    for i in iter_range:
        if i != 7:
            sliding_temp = all_data[(all_data.day.isin([i]))&(all_data.hour < 12)][group_list+['is_trade']].reset_index(drop = True)
            sliding_temp['sliding_2d_be_'+first_fe+'_click'] = 1
            sliding_temp.rename(columns={'is_trade': 'sliding_2d_be_' + first_fe + '_buy'}, inplace=True)
            sliding_temp = sliding_temp.groupby(group_list).agg('sum').reset_index()
            
            sliding_temp_hour = all_data[all_data.day.isin([i])][group_list+['hour','is_trade']].reset_index(drop = True)
            sliding_temp_hour['sliding_2d_be_hour'+first_fe+'_click'] = 1
            sliding_temp_hour.rename(columns={'is_trade': 'sliding_2d_be_hour' + first_fe + '_buy'}, inplace=True)

            sliding_temp_hour = sliding_temp_hour.groupby(group_list+['hour']).agg('sum').reset_index()
            sliding_temp_hour = pd.merge(sliding_temp_hour,sliding_temp,how = 'left', on = first_fe)
            sliding_temp_hour['be_' + first_fe + '_convert'] = sliding_temp_hour.apply(lambda x:helper_sliding_beconvert(x['sliding_2d_be_hour' + first_fe + '_buy'],x['sliding_2d_be_hour'+first_fe+'_click'],x['sliding_2d_be_' + first_fe + '_buy'],x['sliding_2d_be_'+first_fe+'_click'],x.hour),axis = 1)
        else:
            sliding_temp = all_data[(all_data.day.isin([i]))&(all_data.hour < 11)][group_list+['is_trade']].reset_index(drop = True)
            sliding_temp['sliding_2d_be_'+first_fe+'_click'] = 1
            sliding_temp.rename(columns={'is_trade': 'sliding_2d_be_' + first_fe + '_buy'}, inplace=True)
            sliding_temp = sliding_temp.groupby(group_list).agg('sum').reset_index()
            
            sliding_temp_hour = all_data[all_data.day.isin([i])][group_list+['hour','is_trade']].reset_index(drop = True)
            sliding_temp_hour['sliding_2d_be_hour'+first_fe+'_click'] = 1
            sliding_temp_hour.rename(columns={'is_trade': 'sliding_2d_be_hour' + first_fe + '_buy'}, inplace=True)

            sliding_temp_hour = sliding_temp_hour.groupby(group_list+['hour']).agg('sum').reset_index()
            print(sliding_temp_hour.columns,'///////////')
            sliding_temp_hour = pd.merge(sliding_temp_hour,sliding_temp,how = 'left', on = first_fe)
            print(sliding_temp_hour.columns)
            sliding_temp_hour['be_' + first_fe + '_convert'] = sliding_temp_hour.apply(lambda x:helper_sliding_beconvert1(x['sliding_2d_be_hour' + first_fe + '_buy'],x['sliding_2d_be_hour'+first_fe+'_click'],x['sliding_2d_be_' + first_fe + '_buy'],x['sliding_2d_be_'+first_fe+'_click'],x.hour),axis = 1)

        # sliding_temp_hour['except_click_' + first_fe] = sliding_temp_hour['sliding_2d_be_'+first_fe+'_click'] - sliding_temp_hour['sliding_2d_be_hour'+first_fe+'_click']
        # sliding_temp_hour['except_buy_' + first_fe] = sliding_temp_hour['sliding_2d_be_' + first_fe + '_buy'] - sliding_temp_hour['sliding_2d_be_hour' + first_fe + '_buy']
        
        sliding_temp_hour['day'] = i
        
        sliding_temp_hour = sliding_temp_hour[['day',first_fe,'hour','be_' + first_fe + '_convert']].reset_index(drop = True)
        if i == 7:
            new_all_data = sliding_temp_hour.reset_index(drop = True)
        else:
            new_all_data = new_all_data.append(sliding_temp_hour,ignore_index = True)
    print("over.....")
    return new_all_data

def sliding_be_one_click_buy_convert_half(all_data, first_fe):
    group_list = [first_fe,'day','half','instance_id']
    sliding_temp = all_data[group_list + ['hour','is_trade']].reset_index(drop = True)
    # sliding_temp['sliding_2d_be_'+first_fe+'_click'] = 1

    sliding_temp.rename(columns={'is_trade': 'sliding_2d_be_' + first_fe + '_buy'}, inplace=True)
    sliding_temp = sliding_temp.groupby(group_list)[''].agg('sum').reset_index()

    sliding_temp[(sliding_temp.hour == 11)&(sliding_temp.day == 7),'sliding_2d_be_'+first_fe+'_click'] += 519888
    sliding_temp[(sliding_temp.hour == 11)&(sliding_temp.day == 7),'sliding_2d_be_' + first_fe + '_buy'] += 18674 
    sliding_temp['be_' + first_fe + '_convert'] = sliding_temp['sliding_2d_be_' + first_fe + '_buy'] / sliding_temp['sliding_2d_be_'+first_fe+'_click']
    
    sliding_temp = sliding_temp[['day',first_fe,'half','be_' + first_fe + '_convert']].reset_index(drop = True)
    print("over.....")
    return sliding_temp

def sliding_be_one_click_buy_convert_all(all_data, first_fe):
    group_list = [first_fe]
    
    iter_list = [7,6,5,4,3,2,1,31]
    # iter_list = [7]
    # iter_list = [0,6,5,4,3]
    iter_range = list(set(iter_list))
    iter_range.sort(key = iter_list.index)
        
    
    sliding_temp = all_data[(all_data.day.isin([6,5,4,3,2,1,31]))][group_list+['is_trade']].reset_index(drop = True)
    sliding_temp['sliding_2d_be_'+first_fe+'_click'] = 1
    sliding_temp.rename(columns={'is_trade': 'sliding_2d_be_' + first_fe + '_buy'}, inplace=True)
    sliding_temp = sliding_temp.groupby(group_list).agg('sum').reset_index()
    
    sliding_temp['be_' + first_fe + '_convert'] = sliding_temp['sliding_2d_be_' + first_fe + '_buy'] / sliding_temp['sliding_2d_be_'+first_fe+'_click']

        # sliding_temp_hour['except_click_' + first_fe] = sliding_temp_hour['sliding_2d_be_'+first_fe+'_click'] - sliding_temp_hour['sliding_2d_be_hour'+first_fe+'_click']
        # sliding_temp_hour['except_buy_' + first_fe] = sliding_temp_hour['sliding_2d_be_' + first_fe + '_buy'] - sliding_temp_hour['sliding_2d_be_hour' + first_fe + '_buy']
        
    sliding_temp['day'] = 7
    
    return sliding_temp
def add_sliding_be_one_click_buy_convert_all(pd_data,all_data,first_fe):
    temp_all_data = all_data.drop_duplicates().reset_index(drop = True)
    pd_data = pd.merge(pd_data, temp_all_data, on = ['day',first_fe], how = 'left')
    pd_data['be_' + first_fe + '_convert'].fillna(-1,inplace = True)
    return pd_data
def sliding_be_one_change(pd_data, first_fe):
    list_range = [7,6,5,4,3,2,1]
    for i in list_range:
        if i == 1:
            temp_data = pd_data[pd_data.day == 31][[first_fe, 'shop_id']].reset_index(drop = True)
        else:
            temp_data = pd_data[pd_data.day.isin(list(filter(lambda x: x < i,list_range)) + [31,])][[first_fe, 'shop_id']].reset_index(drop = True)
        temp_data['yesterday_max_' + first_fe] = temp_data.groupby(['shop_id'])[first_fe].transform('max')
        temp_data['day'] = i
        temp_data = temp_data[['shop_id', 'yesterday_max_' + first_fe, 'day']].reset_index(drop = True)
        if i == 7:
            new_all_data = temp_data.reset_index(drop = True)
        else:
            new_all_data = new_all_data.append(temp_data,ignore_index = True)
    return new_all_data
def add_sliding_be_one_change(all_data,pd_data,first_fe):
    temp_all_data = pd_data.drop_duplicates().reset_index(drop = True)
    all_data = pd.merge(all_data, temp_all_data, on = ['day','shop_id'], how = 'left')
    all_data['yesterday_max_' + first_fe].fillna(999,inplace = True)
    return all_data
def sliding_item_change(pd_data, first_fe):
    list_range = [7,6,5,4,3,2,1]
    for i in list_range:
        if i == 1:
            temp_data = pd_data[pd_data.day == 31][[first_fe, 'item_id']].reset_index(drop = True)
        else:
            temp_data = pd_data[pd_data.day.isin(list(filter(lambda x: x < i,list_range)) + [31,])][[first_fe, 'item_id']].reset_index(drop = True)
        temp_data['yesterday_max_' + first_fe] = temp_data.groupby(['item_id'])[first_fe].transform('max')
        temp_data['day'] = i
        temp_data = temp_data[['item_id', 'yesterday_max_' + first_fe, 'day']].reset_index(drop = True)
        if i == 7:
            new_all_data = temp_data.reset_index(drop = True)
        else:
            new_all_data = new_all_data.append(temp_data,ignore_index = True)
    return new_all_data
def add_sliding_item_id_change(all_data,pd_data,first_fe):
    temp_all_data = pd_data.drop_duplicates().reset_index(drop = True)
    all_data = pd.merge(all_data, temp_all_data, on = ['day','item_id'], how = 'left')
    all_data['yesterday_max_' + first_fe].fillna(999,inplace = True)
    return all_data
def add_sliding_be_one_click_buy_convert(pd_data,all_data,first_fe):
    temp_all_data = all_data.drop_duplicates().reset_index(drop = True)
    pd_data = pd.merge(pd_data, temp_all_data, on = ['day','hour',first_fe], how = 'left')
    pd_data['be_' + first_fe + '_convert'].fillna(-1,inplace = True)
    return pd_data
def add_sliding_be_one_click_buy_convert_half(pd_data,all_data,first_fe):
    temp_all_data = all_data.drop_duplicates().reset_index(drop = True)
    pd_data = pd.merge(pd_data, temp_all_data, on = ['day',first_fe,'half'], how = 'left')
    pd_data['be_' + first_fe + '_convert'].fillna(-1,inplace = True)
    return pd_data

def sliding_one_to_one(all_data, first_fe, second_fe, interval = 1, test = False, smoothing_dict = {}, normal = []):
    group_list = [first_fe,second_fe]
    if test:
        iter_range = [1]
    else:
        iter_list = [0,6,5,4,3,2,1]
        iter_range = list(set(iter_list))
        iter_range.sort(key = iter_list.index)
    for i in iter_range:
        # if i == 1 and not test:
        #     sliding_temp = all_data[all_data.weekday.isin([2])][group_list+['is_trade']].reset_index(drop = True)
        # else:
        #     sliding_temp = all_data[all_data.weekday.isin([(i + 7 - interval)%7,(i + 7 - 1)%7])][group_list+['is_trade']].reset_index(drop = True)
        if test:
            sliding_temp = all_data[group_list+['is_trade']].reset_index(drop = True)
        else:
            sliding_temp = all_data[~all_data.weekday.isin([i,])][group_list+['is_trade']].reset_index(drop = True)
        sliding_temp['sliding_2d_' + first_fe + '_' + second_fe + '_click'] = 1
        sliding_temp.rename(columns={'is_trade': 'sliding_2d_' + first_fe + '_' + second_fe + '_buy'}, inplace=True)
        sliding_temp = sliding_temp.groupby(group_list).agg('sum').reset_index()
        sliding_temp['weekday'] = i
        sliding_temp['sliding_2d_' + first_fe + '_' + second_fe + '_convert'] = sliding_temp['sliding_2d_' + first_fe + '_' + second_fe + '_buy'] / sliding_temp['sliding_2d_' + first_fe + '_' + second_fe + '_click']
        #add normal
        if i == 1 and not test:
            sliding_temp['sliding_2d_' + first_fe + '_' + second_fe + '_click'] = sliding_temp['sliding_2d_' + first_fe + '_' + second_fe + '_click'] / normal[2]
            sliding_temp['sliding_2d_' + first_fe + '_' + second_fe + '_buy'] = sliding_temp['sliding_2d_' + first_fe + '_' + second_fe + '_buy'] / normal[2]
        else:
            sliding_temp['sliding_2d_' + first_fe + '_' + second_fe + '_click'] = sliding_temp['sliding_2d_' + first_fe + '_' + second_fe + '_click'] / normal[(i + 7 - 1)%7]
            sliding_temp['sliding_2d_' + first_fe + '_' + second_fe + '_buy'] = sliding_temp['sliding_2d_' + first_fe + '_' + second_fe + '_buy'] / normal[(i + 7 - 1)%7]
        
        
        sliding_temp = sliding_temp.reset_index(drop = True)
        if i == 0 or (test == True):
            new_all_data = sliding_temp.reset_index(drop = True)
        else:
            new_all_data = new_all_data.append(sliding_temp,ignore_index = True)
    # if 'sliding_2d_' + first_fe + '_' + second_fe + '_convert' not in smoothing_dict.keys():
    #     print("computing " + first_fe + " " + second_fe + " smoothing ratio")
    #     alpha, beta = smoothing_fuc(series_click = new_all_data['sliding_2d_' + first_fe + '_' + second_fe + '_click'], series_buy = new_all_data['sliding_2d_' + first_fe + '_' + second_fe + '_buy'])
    # else:
    #     print("load " + first_fe + " " + second_fe + " smoothing ratio")
    #     alpha, beta = smoothing_dict['sliding_2d_' + first_fe + '_' + second_fe + '_convert']
    # new_all_data['sliding_2d_' + first_fe + '_' + second_fe + '_convert'] = new_all_data.apply(lambda x:helper_add_all_smoothing(x['sliding_2d_' + first_fe + '_' + second_fe + '_buy'],x['sliding_2d_' + first_fe + '_' + second_fe + '_click'],alpha,beta),axis = 1)
    # new_all_data['sliding_2d_' + first_fe + '_' + second_fe + '_convert'] = (new_all_data['sliding_2d_' + first_fe + '_' + second_fe + '_buy'] + alpha) / (new_all_data['sliding_2d_' + first_fe + '_' + second_fe + '_click'] + alpha + beta)
    print("over.....")
    alpha, beta = 0, 0
    return alpha, beta, new_all_data
def helper_add_all_smoothing(sliding_buy,sliding_click,alpha,beta):
    if np.isnan(sliding_click):
        return alpha / (alpha + beta)
    else:
        return (sliding_buy + alpha) / (sliding_click + alpha + beta)
def add_sliding_one_to_one_click_buy_convert(pd_data,all_data,first_fe,second_fe):
    temp_all_data = all_data.drop_duplicates().reset_index(drop = True)
    pd_data = pd.merge(pd_data, temp_all_data, on = [first_fe,'weekday',second_fe], how = 'left')
    pd_data['sliding_2d_' + first_fe + '_' + second_fe + '_convert'].fillna(-1,inplace = True)
    return pd_data

def add_sliding_features(func, lists, args = None, first_fe = "", second_fe = ""):
    result = []
    pool = multiprocessing.Pool(processes = 10)
    for i in lists:
        result.append(pool.apply_async(func=func,args=(i,args,first_fe,second_fe)))
    pool.close()
    pool.join()
    result = list(map(lambda x:x.get(),result))
    return result

def smoothing_fuc(series_click, series_buy):
    I=[]
    C=[]
    bs = BayesianSmoothing(1, 1)
    temp_series_click = series_click[~series_click.isna()]
    temp_series_buy = series_buy[~series_buy.isna()]
    for i in temp_series_click:
        I.append(i)
    for i in temp_series_buy:
        C.append(i)
    bs.update(I, C, 1000, 0.0000000001)
    print(len(I),len(C))
    if len(I)!=len(C):
        print("not match!")
        sys.exit()
    return bs.alpha,bs.beta
