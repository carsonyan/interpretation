#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 10:42:48 2018

@author: carsonyan
"""

import pandas as pd
import numpy as np
import re
import xgboost as xgb

special_vals = {-9:'No Bureau Record or No Investigation',-8:'No Usable/Valid Trades or Inquiries',-7:'Condition not Met (e.g. No Inquiries, No Delinquencies) '}
dict_MaxDelq2PublicRecLast12M = {
        0:'derogatory comment',
        1:'120+ days delinquent',
        2:'90 days delinquent',
        3:'60 days delinquent',
        4:'30 days delinquent',
        5:'unknown delinquency',
        6:'unknown delinquency',
        7:'current and never delinquent',
        8:'all other',
        9:'all other',
        -9:'No Bureau Record or No Investigation',
        -8:'No Usable/Valid Trades or Inquiries',
        -7:'Condition not Met (e.g. No Inquiries, No Delinquencies)'}
dict_MaxDelqEver = {
        2:'derogatory comment',
        3:'120+ days delinquent',
        4:'90 days delinquent',
        5:'60 days delinquent',
        6:'30 days delinquent',
        7:'unknown delinquency',
        8:'current and never delinquent',
        9:'all other',
        -9:'No Bureau Record or No Investigation',
        -8:'No Usable/Valid Trades or Inquiries',
        -7:'Condition not Met (e.g. No Inquiries, No Delinquencies)'}
dict_cat = {'MaxDelq2PublicRecLast12M':dict_MaxDelq2PublicRecLast12M, 'MaxDelqEver':dict_MaxDelqEver}

def oneHot(dat, feat_num, fit=True, feat_num_med=None):
    if fit:
        feat_num_med = {}
    else:
        assert feat_num_med!= None
    if isinstance(dat, pd.core.series.Series):
        dat = dat.to_frame().T
        for c in dat.columns:
            dat[c] = pd.to_numeric(dat[c],errors='ignore')
    for f in feat_num:
        l7 = dat[f] == -7
        l8 = dat[f] == -8    
        l9 = dat[f] == -9
        if l7.any():
            dat[f+'_7'] = False
            dat.loc[l7, f+'_7'] = True
        if l8.any():
            dat[f+'_8'] = False
            dat.loc[l8, f+'_8'] = True
        if l9.any():
            dat[f+'_9'] = False
            dat.loc[l9, f+'_9'] = True
        if fit:
            feat_num_med[f] = dat.loc[~l7&~l8&~l9, f].median()
        dat.loc[l7|l8|l9, f] = feat_num_med[f]
    return dat, feat_num_med

#Summarize the categorical vars for training data
cat_sum_train = {}



#deal with cat vars, forece monotonicity
def sumCat(dat, feat_cat, target):
    for f in feat_cat:
      cat_sum_train_this = pd.concat([dat.groupby(f).size().rename('count'), dat.groupby(f)[target].sum()], axis=1)
      cat_sum_train_this['BadPercentage'] = cat_sum_train_this['IsBad']/cat_sum_train_this['count']
      cat_sum_train[f] = cat_sum_train_this
    cat_sum_train_tm = {}
    for f in feat_cat:
      dat[f+'_c'] = dat[f].astype(str)
      #f_c = dat[f].copy()
      #dat[f+'_c'] = f_c.astype(str).copy()
      if f == 'MaxDelqEver':
          dat.loc[dat[f].isin([2,3]),f+'_c'] = '2 3' #combine derogatory comment and 120+ days delinquent
          dat.loc[dat[f].isin([4,5]),f+'_c'] = '4 5' #conmbine 90 days delinquent and 60 days delinquent
         
      elif f == 'MaxDelq2PublicRecLast12M':
          dat.loc[dat[f].isin([0,1,2,3,4]),f+'_c'] = '0 1 2 3 4' #combine all delinquencies
          dat.loc[dat[f].isin([5,6]),f+'_c'] = '5 6' #combine unknown
          dat.loc[dat[f].isin([9,-9]),f+'_c'] = '-9 9' #combine No Bureau Record or No Investigation and all other
              
      cat_sum_this = pd.concat([dat.groupby(f+'_c').size().rename('count'), dat.groupby(f+'_c')[target].sum()], axis=1)
      cat_sum_this['BadPercentage'] = cat_sum_this['IsBad']/cat_sum_this['count']
      cat_sum_train_tm[f+'_c'] = cat_sum_this
      
    #Replace categories with target means
    cat_sum_final = {}
    for f in feat_cat:
        dat = pd.merge(dat, cat_sum_train_tm[f+'_c'].reset_index()[[f+'_c','BadPercentage']], on=f+'_c', how='left')
        dat.rename(columns={'BadPercentage':f+'_tm'}, inplace=True)
        #dat.drop(columns=f+'_c', inplace=True)
        cat_sum_final[f]=dat.groupby([f,f+'_tm']).size().reset_index()[[f,f+'_tm']]
        
    return cat_sum_final

def getTargetMean(dat, feat_cat, target, cat_sum_final=None):
    if cat_sum_final==None:
        cat_sum_final = sumCat(dat, feat_cat, target)
    if isinstance(dat, pd.core.series.Series):
        dat = dat.to_frame().T
    for f in feat_cat:
        dat = pd.merge(dat, cat_sum_final[f], on=f, how='left')
    return dat, cat_sum_final

def processMeta(meta, eng, dat_train_good_med, cat_sum_final):
    meta.drop(meta.index[[len(meta)-1, len(meta)-2]], inplace=True)

    #fix catergorical feature names due to target mean replacement that was done in the data
    #for c in cat_sum_final.keys():
    #    meta.loc[meta['Variable Names'] == c,'Variable Names'] = c+'_tm'
    
    meta.rename(columns={'Variable Names':'feature', 'Monotonicity Constraint (with respect to probability of bad = 1)':'Monotonicity_Constraint'}, inplace=True)
    
    # create a AA table
    meta_proc = pd.merge(meta,dat_train_good_med.to_frame().reset_index().rename(columns={'index':'feature',0:'Median_Good'}),
                  on='feature', how='outer')
    meta_proc = meta_proc.loc[meta_proc['feature']!='IsBad',:]
    meta_proc['Monotonicity_Constraint'].fillna('No constraint', inplace=True) 
    meta_proc['Role'].fillna('predictor_flag', inplace=True) 
    
    for c in cat_sum_final.keys():
      l_cat = meta_proc['feature'] == c
      l_tm = meta_proc['feature'] == c+'_tm'
      meta_proc.loc[l_tm,'Monotonicity_Constraint'] = 'Monotonically Increasing'
      meta_proc.loc[l_tm,'Role'] = 'predictor_tm'
      meta_proc.loc[l_tm,'Description'] = meta_proc.loc[l_cat,'Description'].iloc[0]
      meta_proc.loc[l_cat,'Role'] = 'predictor_cat'

    meta_proc['Description'] = meta_proc.apply(lambda x: fillDes(x, meta), axis=1)
    
    eng_proc = pd.merge(meta_proc, eng, how='left', on='feature')
    
    eng_proc['English'] = eng_proc.apply(lambda x: fillEnglish(x, meta_proc), axis=1)
    eng_proc.loc[eng_proc.shape[0]] = ['bias']+[np.nan]*(eng_proc.shape[1]-1) #['bias', nan, nan, nan, nan, nan, nan, nan, nan]

    
    return meta_proc, eng_proc

def fillDes(row, meta):
    if pd.isna(row['Description']):
        feat_root = row['feature'][:-2]
        feat_val = -int(row['feature'][-1])
        feat_desc = meta.loc[meta['feature'] == feat_root, 'Description'].iloc[0]
        return special_vals[feat_val] + ' in ' + feat_desc
    else:
        return row['Description']    
    
 
        
def fillEnglish(row, meta_proc):
    if pd.isna(row['English']) & (row['Role']=='predictor_flag'):
        s = row['Description']
        s_list = s.split(' in ')
        return s_list[0] + ' for ' + meta_proc.loc[meta_proc['feature']==row['feature'][:-2], 'Description'].iloc[0]
    else:
        return row['English']


def translate(exp_row, aa_eng, base_val='shap'):
    row_eng = aa_eng.loc[aa_eng['feature']==exp_row['feature'],'English'].iloc[0]
    if exp_row['feature']=='bias':
        row_eng = ''
    elif exp_row['Role']=='predictor_cat':
        #print(exp_row)
        row_eng = row_eng.replace("()", dict_cat[exp_row['feature']][exp_row['val']])
        if exp_row[base_val]>0:
            word_use = 'KeyWord_Bad'
        else:
            word_use = 'KeyWord_Good'
        row_eng = row_eng.replace("{}", aa_eng.loc[aa_eng['feature']==exp_row['feature'],word_use].iloc[0])
    elif (exp_row['Role']=='predictor') & ('monotonically' in exp_row['Monotonicity_Constraint'].lower()):
        if ' too ' in exp_row['reason'].lower():
            word_use = 'KeyWord_Bad'
        else:
            word_use = 'KeyWord_Good'
        row_eng = row_eng.replace("{}", aa_eng.loc[aa_eng['feature']==exp_row['feature'],word_use].iloc[0])  
    elif exp_row['Role']=='predictor':
        #print(exp_row['feature'])
        low_high = aa_eng.loc[aa_eng['feature']==exp_row['feature'],'adj_pair'].iloc[0].split(' ')
        if ' too ' in exp_row['reason'].lower():
            low_high = ['too '+e for e in low_high]
        if 'low' in exp_row['reason']:
            row_eng = row_eng.replace("{}", low_high[0])
        elif 'high' in exp_row['reason']:
            row_eng = row_eng.replace("{}", low_high[1])
        elif exp_row[base_val]>0:
            row_eng = row_eng.replace("{}", 'Not desirable')
        else:
            row_eng = row_eng.replace("{}", 'desirable')
    else :
        pass
    #print(row_eng)
    return  row_eng.title()   #TBD
    
def getReasons(row, use_val='shap', reference_col_name=None):
    #print(row['feature'])
    if row[use_val]>0: #Unfavorable
        if row['Monotonicity_Constraint'] == 'Monotonically Decreasing':
            return row['feature'] + ' is too low'
        elif row['Monotonicity_Constraint'] == 'Monotonically Increasing':
            return row['feature'] + ' is too high'
        elif row['Monotonicity_Constraint'] == 'No constraint': #Can't compare to median because sharpley values depends on other feature values too
            return row['feature'] +' is ' +  str(row['val']) #

        elif (row['Role'] == 'predictor_cat'):
            min_max = [int(s) for s in re.findall(r'\d+', row['Monotonicity_Constraint'])]
            if ('decreasing' in row['Monotonicity_Constraint']) & (row['val']>=min_max[0]) & (row['val']<min_max[1]):
                return row['feature'] + ' is too low'
            elif ('increasing' in row['Monotonicity_Constraint']) & (row['val']>min_max[0]) & (row['val']<=min_max[1]):
                return row['feature'] + ' is too high'
            else:
                return row['feature'] + ' is ' + str(row['val'])        
        else:
            return np.nan
    
    else: #these are favorable, not reject reasons, but just to populate for each feature
        if row['Monotonicity_Constraint'] == 'Monotonically Decreasing':
            return row['feature'] + ' is high'
        elif row['Monotonicity_Constraint'] == 'Monotonically Increasing':
            return row['feature'] + ' is low'
        elif row['Monotonicity_Constraint'] == 'No constraint':
            return row['feature'] +' is ' +  str(row['val'])
            
        elif (row['Role'] == 'predictor_cat'):
            min_max = [int(s) for s in re.findall(r'\d+', row['Monotonicity_Constraint'])]
            if ('decreasing' in row['Monotonicity_Constraint']) & (row['val']>min_max[0]) & (row['val']<=min_max[1]):
                return row['feature'] + ' is high'
            elif ('increasing' in row['Monotonicity_Constraint']) & (row['val']>=min_max[0]) & (row['val']<min_max[1]):
                return row['feature'] + ' is low'
            else:
                return row['feature'] + ' is ' + str(row['val'])         
        else:
            return np.nan
        
def prepObs(new_ins, f_names, feat_cat, feat_num, target, feat_num_med, cat_sum_final):
    #Prepare data before prediction
    new_ins, _ = oneHot(new_ins, feat_num, fit=False, feat_num_med=feat_num_med)
    new_ins, _ = getTargetMean(new_ins, feat_cat, target, cat_sum_final=cat_sum_final) #dat, 
    for f in list(set(f_names)-set(new_ins.columns)):
        new_ins[f]=False
    new_ins_f = pd.to_numeric(new_ins.loc[0, f_names])
    return new_ins, new_ins_f

def logistic(x):
    return 1/(1+np.exp(-x))

def generateRejectReasons(model, f_names, new_ins, new_ins_f, meta_proc, eng_proc, print_reasons=False):

    
    #Sharpley values
    shap_pred = model.predict(xgb.DMatrix(new_ins_f, feature_names=f_names), pred_contribs=True)
    shap_pred_raw = shap_pred[:,:-1]
    shap_bias = shap_pred[0,-1]
    
    shap_pred_sum = np.sum(shap_pred)
    proba_pred = logistic(shap_pred_sum)
    
    shap_pred = pd.DataFrame(shap_pred, columns=f_names+['bias'])
    shap_pred = shap_pred.T
    shap_pred.columns = ['shap']
    shap_pred.reset_index(inplace=True)
    new_ins_f['bias'] = np.nan
    shap_pred['val'] = new_ins_f.values
    #print(shap_pred)
    shap_pred.sort_values('shap', ascending=False, inplace=True)
    shap_pred.rename(columns={'index':'feature'}, inplace=True)
    
    #Change target mean values back to original values
    l = shap_pred['feature'] == 'MaxDelq2PublicRecLast12M_tm'
    shap_pred.loc[l,'feature'] = 'MaxDelq2PublicRecLast12M'
    #print(new_ins['MaxDelq2PublicRecLast12M'])
    shap_pred.loc[l,'val'] = new_ins['MaxDelq2PublicRecLast12M'].iloc[0]

    l = shap_pred['feature'] == 'MaxDelqEver_tm'
    shap_pred.loc[l,'feature'] = 'MaxDelqEver'
    shap_pred.loc[l,'val'] = new_ins['MaxDelqEver'].iloc[0]
    

    #print(shap_pred.loc[shap_pred['feature'].isin(['MaxDelq2PublicRecLast12M_tm','MaxDelq2PublicRecLast12M']),'feature']) 
    
    #Merge data dictionary
    shap_pred = pd.merge(shap_pred, meta_proc, on='feature', how='left')

    #Get reasons
    shap_pred['reason'] = shap_pred.apply(getReasons,axis=1)
    
    #print(shap_pred.loc[shap_pred['feature'] == 'MaxDelq2PublicRecLast12M_tm','val'])    
    
    #Get translation
    shap_pred['explanation'] = shap_pred.apply(lambda x: translate(x, eng_proc), axis=1)
    
    if print_reasons:
        print('----------Reasons Analysis--------')
        #print(proba_pred)
        print('The predicted probability of going bad is {0:.3f}'.format(proba_pred))
        print('This probability is' + (' high.' if shap_pred_sum>shap_bias else ' low.'))
        print()
        for i in range(4):
            if (shap_pred.loc[i,'feature'] == 'bias') or (shap_pred.loc[i,'shap'] <= 0):#This happens when predicted prob is low.
                break
            print(shap_pred.loc[i,'reason'])
            print(shap_pred.loc[i,'explanation'])
            print()
    
    
    return shap_pred, shap_bias, shap_pred_raw