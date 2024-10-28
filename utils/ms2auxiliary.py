# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 21:35:58 2024

@author: 555
"""

import pandas as pd

def ms2_difference():
    dfA = pd.read_csv('D:/gitclone/denoise/data/predict/coupling/topfd/MS2/ms2_data_filter.csv')
    dfB = pd.read_csv("D:/gitclone/denoise/data/predict/coupling/topfd/MS2/ms2_data_filter_1020_2391.csv")
  
    diff = dfB.merge(dfA, how='outer', indicator=True)
  
    diff = diff[diff['_merge'] == 'left_only']
 
    diff = diff.drop(columns=['_merge'])

    diff.to_csv('D:/gitclone/denoise/data/predict/coupling/topfd/MS2/difference_1020_2391.csv', index=False)



def ms2_eic_merge():
    df1 = pd.read_csv('D:/gitclone/denoise/data/predict/coupling/topfd/MS2/eic_ms2_data_final.csv')
    df2 = pd.read_csv('D:/gitclone/denoise/data/predict/coupling/topfd/MS2/eic_ms2_data_difference_1020_2391.csv')
    
    df_merged = pd.concat([df1, df2], ignore_index=True)
    
    df_sorted = df_merged.sort_values(by=['Scan', 'ID'])
    
    df_sorted.to_csv('D:/gitclone/denoise/data/predict/coupling/topfd/MS2/ms2_eic_merged_sorted_1020_2391.csv', index=False)
    
ms2_eic_merge()