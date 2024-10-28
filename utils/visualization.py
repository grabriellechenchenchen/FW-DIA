# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 18:30:48 2024

@author: 555
"""

import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd
import numpy as np

def read_monoisotopic_mass_from_tsv(file_path):
    monoisotopic_mass_list = []
    
    with open(file_path, mode='r', newline='') as file:
        tsv_reader = csv.DictReader(file, delimiter='\t')
        
        for row in tsv_reader:
            mass = row.get('MonoisotopicMass')
            if mass is not None:
                monoisotopic_mass_list.append(float(mass))
    
    return monoisotopic_mass_list




def read_massed_from_topfd_feature_file(file_path):
    monoisotopic_mass_list = []
    
    with open(file_path, mode='r', newline='') as file:
        tsv_reader = csv.DictReader(file, delimiter='\t')
        
        for row in tsv_reader:
            mass = row.get('Mass')
            if mass is not None:
                monoisotopic_mass_list.append(float(mass))
    
    return monoisotopic_mass_list
 


def five_model_deconv_result():
    #五个蛋白flashdeconv结果
    file_path = "D:\\gitclone\\denoise\\data\\original_005_model\\wugedanbai1001\\wugedanbai70105_ms1_bpi005.tsv"
    data1 = read_monoisotopic_mass_from_tsv(file_path)
    
    file_path2 = "D:\\gitclone\\denoise\\data\\original_005_model\\wugedanbai1001\\wugedanbai70105_ms1_original.tsv"
    data2 = read_monoisotopic_mass_from_tsv(file_path2)
    
    file_path3 = "D:\\gitclone\\denoise\\data\\original_005_model\\wugedanbai1001\\denoised_ms_centroid.tsv"
    data3 = read_monoisotopic_mass_from_tsv(file_path3)
    
    file_path = "D:\\gitclone\\denoise\\data\\original_005_model\\wugedanbai1001\\wugedanbai70105_ms1_bpi005_ms1.feature"
    data4 = read_massed_from_topfd_feature_file(file_path)
    
    file_path = "D:\\gitclone\\denoise\\data\\original_005_model\\wugedanbai1001\\wugedanbai70105_ms1_original_ms1.feature"
    data5 = read_massed_from_topfd_feature_file(file_path)
    
    file_path = "D:\\gitclone\\denoise\\data\\original_005_model\\wugedanbai1001\\denoised_ms_centroid_ms1.feature"
    data6 = read_massed_from_topfd_feature_file(file_path)
    
    
    data = [data1, data2, data3,data4, data5, data6]
    df = pd.DataFrame({
        'values': data1 + data2 + data3 + data4 + data5 + data6,
        'category': ['BPI>0.005'] * len(data1) + 
                    ['Original'] * len(data2) + 
                    ['CNN Model'] * len(data3) + 
                    ['BPI>0.0051'] * len(data4) + 
                    ['Original1'] * len(data5) + 
                    ['CNN Model1'] * len(data6)
    })
    

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
    sns.set(style="whitegrid")
    
    fragment_num = [len(x) for x in data]
    print(fragment_num)
    
    fragment_categories = ['BPI>0.005', 'Original', 'CNN Model', 'BPI>0.0051', 'Original1', 'CNN Model1']
    fragment_df = pd.DataFrame({
        'category': fragment_categories,
        'fragment_num': fragment_num
    })
    
    sns.barplot(x='category', y='fragment_num', data=fragment_df, palette='muted', ax=ax1)
    

    sns.stripplot(x='category', y='values', data=df, alpha=0.6, jitter=True, ax=ax2)
    ax2.set_title('Values by Category')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Values')

    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.show()




def five_model_deconv_result2():
    #五个蛋白flashdeconv结果
    file_path = "D:\\gitclone\\denoise\\data\\original_005_model\\wugedanbai1001\\wugedanbai70105_ms1_bpi005.tsv"
    data1 = read_monoisotopic_mass_from_tsv(file_path)
    
    file_path2 = "D:\\gitclone\\denoise\\data\\original_005_model\\wugedanbai1001\\wugedanbai70105_ms1_original.tsv"
    data2 = read_monoisotopic_mass_from_tsv(file_path2)
    
    file_path3 = "D:\\gitclone\\denoise\\data\\original_005_model\\wugedanbai1001\\denoised_ms_centroid.tsv"
    data3 = read_monoisotopic_mass_from_tsv(file_path3)
    
    file_path = "D:\\gitclone\\denoise\\data\\original_005_model\\hetngti1001\\hetangti2070_ms1_bpi005.tsv"
    data4 = read_monoisotopic_mass_from_tsv(file_path)

    file_path2 = "D:\\gitclone\\denoise\\data\\original_005_model\\hetngti1001\\hetangti2070_ms1_original.tsv"
    data5 = read_monoisotopic_mass_from_tsv(file_path2)

    file_path3 = "D:\\gitclone\\denoise\\data\\original_005_model\\hetngti1001\\denoised_ms1_centroid.tsv"
    data6 = read_monoisotopic_mass_from_tsv(file_path3)
    
    #wugedanbai
    data = [data1, data2, data3]
    df = pd.DataFrame({
        'values': data1 + data2 + data3 ,
        'category': ['BPI>0.005'] * len(data1) + 
                    ['Original'] * len(data2) + 
                    ['CNN Model'] * len(data3) 
                    
    })
    
    fragment_num = [len(x) for x in data]
    print(fragment_num)
    
    fragment_categories = ['BPI>0.005', 'Original', 'CNN Model']
    fragment_df = pd.DataFrame({
        'category': fragment_categories,
        'fragment_num': fragment_num
    })
    
    
    #hetangti
    data1 = [data4, data5, data6]
    df1 = pd.DataFrame({
        'values1':  data4 + data5 + data6,
        'category1': ['BPI>0.005'] * len(data4) + 
                    ['Original'] * len(data5) + 
                    ['CNN Model'] * len(data6) 
                    
    })
    
    
    fragment_num1 = [len(x) for x in data1]
    print(fragment_num1)
    
    fragment_categories1 = ['BPI>0.005', 'Original', 'CNN Model']
    fragment_df1 = pd.DataFrame({
        'category2': fragment_categories1,
        'fragment_num1': fragment_num1
    })
    
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    sns.set(style="whitegrid") 
    sns.barplot(x='category', y='fragment_num', data=fragment_df, palette='Set2', ax=ax1) 
    ax1.set_ylim(top=700)
    
    sns.stripplot(x='category', y='values', data=df, alpha=0.7, jitter=True, ax=ax3, palette='Set2')
    ax2.set_ylim(top=1200)
    

    sns.barplot(x='category2', y='fragment_num1', data=fragment_df1, palette='Set2', ax=ax2)

    
    sns.stripplot(x='category1', y='values1', data=df1, alpha=0.7, jitter=True, ax=ax4, palette='Set2')


    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.show()


#five_model_deconv_result2()


def ms2_model_deconv_result2():
    file_path = "D:\\gitclone\\denoise\\data\\original_005_model\\wugedanbai1002_ms2\\wugedanbai70105_ms2_centroid_bpi005.tsv"
    data1 = read_monoisotopic_mass_from_tsv(file_path)
    
    file_path2 = "D:\\gitclone\\denoise\\data\\original_005_model\\wugedanbai1002_ms2\\wugedanbai70105_ms2_centroid_original.tsv"
    data2 = read_monoisotopic_mass_from_tsv(file_path2)
    
    file_path3 = "D:\\gitclone\\denoise\\data\\original_005_model\\wugedanbai1002_ms2\\denoised_ms_edit_centroid.tsv"
    data3 = read_monoisotopic_mass_from_tsv(file_path3)
    
    file_path = "D:\\gitclone\\denoise\\data\\original_005_model\\hetangti1002_ms2\\hetangti2070_ms2_centroid_bpi005_2.tsv"
    data4 = read_monoisotopic_mass_from_tsv(file_path)

    file_path2 = "D:\\gitclone\\denoise\\data\\original_005_model\\hetangti1002_ms2\\hetangti2070_ms2_centroid_original.tsv"
    data5 = read_monoisotopic_mass_from_tsv(file_path2)

    file_path3 = "D:\\gitclone\\denoise\\data\\original_005_model\\hetangti1002_ms2\\denoised_ms_edit_try2_centroid.tsv"
    data6 = read_monoisotopic_mass_from_tsv(file_path3)
    
    #wugedanbai
    data = [data1, data2, data3]
    df = pd.DataFrame({
        'values': data1 + data2 + data3 ,
        'category': ['BPI>0.005'] * len(data1) + 
                    ['Original'] * len(data2) + 
                    ['CNN Model'] * len(data3) 
                    
    })
    
    fragment_num = [len(x) for x in data]
    print(fragment_num)
    
    fragment_categories = ['BPI>0.005', 'Original', 'CNN Model']
    fragment_df = pd.DataFrame({
        'category': fragment_categories,
        'fragment_num': fragment_num
    })
    
    
    #hetangti
    data1 = [data4, data5, data6]
    df1 = pd.DataFrame({
        'values1':  data4 + data5 + data6,
        'category1': ['BPI>0.005'] * len(data4) + 
                    ['Original'] * len(data5) + 
                    ['CNN Model'] * len(data6) 
                    
    })
    
    
    fragment_num1 = [len(x) for x in data1]
    print(fragment_num1)
    
    fragment_categories1 = ['BPI>0.005', 'Original', 'CNN Model']
    fragment_df1 = pd.DataFrame({
        'category2': fragment_categories1,
        'fragment_num1': fragment_num1
    })
    
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    sns.set(style="whitegrid") 
    sns.barplot(x='category', y='fragment_num', data=fragment_df, palette='Set2', ax=ax1) 
    ax1.set_ylim(top=600)
    
    sns.stripplot(x='category', y='values', data=df, alpha=0.7, jitter=True, ax=ax3, palette='Set2')
    ax2.set_ylim(top=800)
    

    sns.barplot(x='category2', y='fragment_num1', data=fragment_df1, palette='Set2', ax=ax2)

    
    sns.stripplot(x='category1', y='values1', data=df1, alpha=0.7, jitter=True, ax=ax4, palette='Set2')


    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.show()

ms2_model_deconv_result2()

def hetagnti_model_deconv_result():
    #核糖体flashdeconv结果

    file_path = "D:\\gitclone\\denoise\\data\\original_005_model\\hetngti1001\\hetangti2070_ms1_bpi005.tsv"
    data1 = read_monoisotopic_mass_from_tsv(file_path)

    file_path2 = "D:\\gitclone\\denoise\\data\\original_005_model\\hetngti1001\\hetangti2070_ms1_original.tsv"
    data2 = read_monoisotopic_mass_from_tsv(file_path2)

    file_path3 = "D:\\gitclone\\denoise\\data\\original_005_model\\hetngti1001\\denoised_ms1_centroid.tsv"
    data3 = read_monoisotopic_mass_from_tsv(file_path3)

    
    file_path = "D:\\gitclone\\denoise\\data\\original_005_model\\hetngti1001\\hetangti2070_ms1_bpi005_ms1.feature"
    data4 = read_massed_from_topfd_feature_file(file_path)
    
    file_path = "D:\\gitclone\\denoise\\data\\original_005_model\\hetngti1001\\hetangti2070_ms1_original_ms1.feature"
    data5 = read_massed_from_topfd_feature_file(file_path)
    
    file_path = "D:\\gitclone\\denoise\\data\\original_005_model\\hetngti1001\\hetangti2070_ms1_original_ms1.feature"
    data6 = read_massed_from_topfd_feature_file(file_path)
    
    
    data = [data1, data2, data3,data4, data5, data6]
    df = pd.DataFrame({
        'values': data1 + data2 + data3 + data4 + data5 + data6,
        'category': ['BPI>0.005'] * len(data1) + 
                    ['Original'] * len(data2) + 
                    ['CNN Model'] * len(data3) + 
                    ['BPI>0.0051'] * len(data4) + 
                    ['Original1'] * len(data5) + 
                    ['CNN Model1'] * len(data6)
    })
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
    sns.set(style="whitegrid")
    
    fragment_num = [len(x) for x in data]
    print(fragment_num)
    
    sns.histplot(y=fragment_num, x='category', bins=range(1, max(fragment_num)+2), ax=ax1, kde=False, color='skyblue', edgecolor='black')
    

    sns.stripplot(x='category', y='values', data=df, alpha=0.6, jitter=True, ax=ax2)
    ax2.set_title('Values by Category')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Values')

    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.show()

