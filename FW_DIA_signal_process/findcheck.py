# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:28:49 2024

@author: 555
"""

import os
import json
import re
import pymzml
import pyopenms
from interval import Interval
import rpy2.robjects as robjects
from pyteomics import mass,mzml
import numpy as np
import csv
from Bio import SeqIO
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import fnmatch
import time
import pandas as pd
from ast import literal_eval


'''
/***************************************************************************************
*    The code pertaining to the isotope fitting component included in this file is partially derived from the Panda-UV software. 
*    Below is the citation for the source code:
*    
*    Title: Panda-UV source code
*    Author: Yinlong Zhu
*    Date: 2024
*    Code version: 1.0
*    Availability: https://github.com/PHOENIXcenter/Panda-UV
*
***************************************************************************************/
'''

class Protein(object):
    
    def __init__(self,seq):
        self.seq = seq
    
    @property
    def Mass(self):
        _mass = mass.fast_mass2(sequence=self.seq, charge = 0)
        return _mass
    
    @property
    def SEQLEN(self):
        return len(self.seq)
    
    @property
    def FORMULA(self):
        seq_dic = mass.Composition(sequence = self.seq)
        formula = self.compute_formula(seq_dic)
        
        return formula
    
    @staticmethod
    def compute_formula(seq_dic):
        tmp_s = ""
        for k,v in seq_dic.items():
            tmp_s += str(k)+str(v)
        return tmp_s



    
def find_folders_with_suffix(root_dir, suffix):
    html_path = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name)) and name.endswith(suffix)]
    html_path = html_path[0].replace("\\","/")
    
    return html_path


def process_js_file(root_directory,js_file_path, pearson_result_df):
    
    with open(js_file_path,'r') as uf:
        data = uf.read()
        json_str = data.split('=',1)[1].strip()
        protein_data = json.loads(json_str)
        protein_data = protein_data["protein"]
        protein_compatible_proteoform_num = float(protein_data['compatible_proteoform_number'])
        protein_sequence_name = protein_data["sequence_name"]
        protein_sequence_description = protein_data["sequence_description"]
        
        with open(root_directory+'/findandcheck/DB_sequence.json','r') as uf:
            json_data = uf.read()
            fastadb = json.loads(json_data) 
            complete_seq = fastadb[protein_sequence_name]
            
        protein_data_list = []
        if protein_compatible_proteoform_num == 1:  
            protein_data_list.append(protein_data['compatible_proteoform'])
            
        else: 
            protein_data_list = protein_data['compatible_proteoform']
            
         
        filtered_df = pd.DataFrame()    
        for proteoform in protein_data_list:
            prsm = proteoform['prsm']
            prsm_num = float(proteoform['prsm_number'])
            prsm_list = []
            if prsm_num ==1:
                prsm_list.append(prsm)
            for j in prsm_list:  
                exist_mass_shift = False
                exist_ptm = False

                matched_fragment_number = j['matched_fragment_number']
                matched_peak_number = j['matched_peak_number']
                
                ms = j['ms']
                ms_header = ms['ms_header']
                ms1_mono_mass = float(ms_header["precursor_mono_mass"])
                ms1_p_max_charge = float(ms_header["precursor_charge"])
                
                pearson_target_df = pearson_result_df[(pearson_result_df['precursor mono mass']>(ms1_mono_mass-1)) & (pearson_result_df['precursor mono mass']<(ms1_mono_mass+1)) & (pearson_result_df['precursor charge'] ==ms1_p_max_charge )]
                if pearson_target_df.shape[0] == 0:
                    print("ALERT: 没有在csv中找到对应scan")
                    print(f'ms1_mono_mass: {ms1_mono_mass}')
                    print(f'ms1_p_max_charge: {ms1_p_max_charge}')
                annotated_protein = j['annotated_protein']
                sequence_name = annotated_protein['sequence_name']
                proteoform_mass = annotated_protein['proteoform_mass']
                unexpected_shift_number = annotated_protein['unexpected_shift_number']
                annotation = annotated_protein['annotation']
                annotated_seq= annotation['annotated_seq']  
                
                annotationnn_seq_remove_bracket = re.sub(r'\[[^]]*\]', '', annotated_seq)
                annotationnn_seq_remove_bracket = annotationnn_seq_remove_bracket.replace('-', '')
                first_dot_index = annotationnn_seq_remove_bracket.find('.')
                second_dot_index = annotationnn_seq_remove_bracket.find('.', first_dot_index + 1)
                substring_between_dots = annotationnn_seq_remove_bracket[first_dot_index + 1:second_dot_index]
                new_str = ''.join(re.findall(r'[a-zA-Z]+', substring_between_dots))
                
                str_1_start_index = complete_seq.index(new_str) 
               
                if 'ptm' in annotation.keys():
                    exist_ptm = True
                    ptm = annotation['ptm']
                    ptm_info = ptm["ptm"]
                    ptm_abbrv = ptm_info["abbreviation"]
                    ptm_unimod = int(ptm_info["unimod"])
                    ptm_mono_mass = float(ptm_info["mono_mass"])
                    
                    ptm_occurence = ptm["occurence"]
                    ptm_occurence_left_pos = int(ptm_occurence["left_pos"])
                    ptm_occurence_right_pos = int(ptm_occurence["right_pos"])
                    ptm_occurence_anno = ptm_occurence["anno"]
                    
                    
                if 'mass_shift' in annotation.keys():
                    
                    exist_mass_shift = True
                    mass_shift =  annotation['mass_shift']
                    if type(mass_shift) is dict:
                        mass_shift_left_position = int(mass_shift["left_position"])
                        mass_shift_right_position =  int(mass_shift["right_position"])
                        mass_shift_weight = float(mass_shift["shift"])
                        mass_shift_type = mass_shift["shift_type"]
                        if abs(mass_shift_weight) < 3:
                            exist_mass_shift = False
                    else: # more than 1 mass shift
                        new_mass_shift_list = []
                        for i in range(0,len(mass_shift)):
                            shift_item = mass_shift[i]
                            if abs(float(shift_item["shift"])) > 3:  
                                new_mass_shift_list.append(shift_item)
                        
                        if len(new_mass_shift_list) ==1:
                            mass_shift = new_mass_shift_list[0]
                            mass_shift_left_position = int(mass_shift["left_position"])
                            mass_shift_right_position =  int(mass_shift["right_position"])
                            mass_shift_weight = float(mass_shift["shift"])
                            mass_shift_type = mass_shift["shift_type"]
                        elif len(new_mass_shift_list) > 1:
                            mass_shift_left_position_list = []
                            mass_shift_right_position_list = []
                            for i in range(0,len(new_mass_shift_list)):
                                mass_shift_item = new_mass_shift_list[i]
                                mass_shift_left_position_list.append(int(mass_shift_item["left_position"]))
                                mass_shift_right_position_list.append(int(mass_shift_item["right_position"]))
                            mass_shift_left_position = min(mass_shift_left_position_list)
                            mass_shift_right_position = max(mass_shift_right_position_list)
                            mass_shift_weight = float("1000")
                            mass_shift_type = "unexpected"
                        else:
                            exist_mass_shift = False
                            
                            
                            
                        
                    
                cleavage = annotation['cleavage']
                for cleavage_position in cleavage:
                    if cleavage_position['matched_peaks'] is not None:  
                        matched_peaks = cleavage_position['matched_peaks']
                        matched_peak = matched_peaks["matched_peak"]
                        matched_peaks_list = []
                        if type(matched_peak) is list:
                            matched_peaks_list = matched_peak
                        else:
                            matched_peaks_list.append(matched_peak)
                        for matched_peak in matched_peaks_list:
                            store_or_not = False
                            peak_ion_type = matched_peak['ion_type']
                            peak_ion_position = int(matched_peak['ion_position'])
                            peak_peak_id = int(matched_peak['peak_id'])  
                            peak_peak_charge = int(matched_peak['peak_charge'])  
                            ion_display_position = int(matched_peak['ion_display_position'])
                            
                            peak_ion_position_update = peak_ion_position + str_1_start_index
                            #print(f'{peak_ion_position} {peak_ion_type} {peak_peak_charge}')
                            #print(f'peak_ion_position_update :{peak_ion_position_update }')
                            if (not exist_mass_shift) and (not exist_ptm): 
                                store_or_not = True
                            
                            if (not exist_mass_shift) and exist_ptm:
                                if ( peak_ion_position_update <= ptm_occurence_left_pos ) and peak_ion_type == 'B': #cun
                                    store_or_not = True
                                elif (peak_ion_position_update >= ptm_occurence_right_pos) and peak_ion_type == 'Y': #cun
                                    store_or_not = True
                                    
                                    
                            if exist_mass_shift and (not exist_ptm):
                                if (peak_ion_position_update <= mass_shift_left_position) and peak_ion_type == 'B': #cun
                                    store_or_not = True
                                elif (peak_ion_position_update >= mass_shift_right_position) and peak_ion_type == 'Y': #cun
                                    store_or_not = True
                            
                            if exist_mass_shift and exist_ptm:
                                if (peak_ion_position_update <= mass_shift_left_position) and peak_ion_type == 'B': #cun
                                    store_or_not = True
                                    
                                elif (peak_ion_position_update >= mass_shift_right_position) and peak_ion_type == 'Y': #cun
                                    store_or_not = True

                            if store_or_not:
                                '''
                                print(f'peak_ion_type:{peak_ion_type}')
                                print(f'peak_ion_position:{peak_ion_position}')
                                print(f'peak_peak_id: {peak_peak_id}')
                                
                                print("  ")
                                '''
                                row_i = pearson_target_df.iloc[peak_peak_id].copy()
                                row_i['ion_type'] = peak_ion_type
                                row_i['ion_position'] = peak_ion_position
                                row_i['ion_display_position'] = ion_display_position
                                row_i['peak_charge'] =peak_peak_charge
                                row_i['sequence_name'] = protein_sequence_name
                                row_i["sequence_description"] = protein_sequence_description
                                row_i['annotated_seq'] =annotated_seq
                                row_i['matched_fragment_number'] = matched_fragment_number
                                row_i['matched_peak_number'] =  matched_peak_number
                                
                                
                                if exist_mass_shift:
                                    row_i['mass_shift_type'] = mass_shift_type 
                                    row_i['mass_shift_weight'] = mass_shift_weight 
                                    row_i['mass_shift_left_position'] = mass_shift_left_position
                                    row_i['mass_shift_right_position'] = mass_shift_right_position
                                else:
                                    row_i['mass_shift_type'] = None 
                                    row_i['mass_shift_weight'] = None
                                    row_i['mass_shift_left_position'] = None
                                    row_i['mass_shift_right_position'] = None
                                
                                if exist_ptm:
                                    row_i['ptm_abbrv'] = ptm_abbrv
                                    row_i['ptm_unimod'] = ptm_unimod
                                    row_i['ptm_mono_mass'] = ptm_mono_mass
                                    row_i['ptm_occurence_left_pos'] = ptm_occurence_left_pos
                                    row_i['ptm_occurence_right_pos'] = ptm_occurence_right_pos
                                    row_i['ptm_occurence_anno'] = ptm_occurence_anno
                                else:
                                    row_i['ptm_abbrv'] = None
                                    row_i['ptm_unimod'] = None
                                    row_i['ptm_mono_mass'] = None
                                    row_i['ptm_occurence_left_pos'] = None
                                    row_i['ptm_occurence_right_pos'] = None
                                    row_i['ptm_occurence_anno'] = None
                                    
                                #use charge, type, position, annotation_seq calculate -- get sequence--- check weight ---formula + xiushi
                                #row_i['theoretical_seq'] = None
                                
                                row_i['charge'] = peak_peak_charge
                        
                                row_i['fragment_mz_intensity_pairs_normalization'] =fragment_mz_intensity_pairs_normalization(row_i['fragment mz internsity pairs'])
                                
                                new_row_df = pd.DataFrame([row_i])

                                filtered_df = pd.concat([filtered_df, new_row_df], ignore_index=True)
                #break  #prsm
            #break #proteofrom break
                            
    return filtered_df 

                                
                            
                                
def fragment_mz_intensity_pairs_normalization(mz_intensity_list):
    mz_intensity_array = np.array(mz_intensity_list)
    intensities = mz_intensity_array[:, 1]
    scaling_num = np.max(intensities) / 100
    mz_intensity_list_nested = [[intensity[0], intensity[1]/ scaling_num] for intensity in mz_intensity_list]

    return mz_intensity_list_nested
                        

from rpy2.robjects import r

def findcheck(root_dir):
    html_folders = find_folders_with_suffix(root_dir, 'html')
    proteins_path = html_folders+'/toppic_prsm_cutoff/data_js/proteins'
    if not os.path.exists(proteins_path):
        print(f"path not exist: {proteins_path}")
        
    js_files = [os.path.join(proteins_path, f) for f in os.listdir(proteins_path)
                if os.path.isfile(os.path.join(proteins_path, f)) and f.endswith('.js')]
    
    js_files = [i.replace("\\","/") for i in js_files]
    
    matched_files = []
    for root, dirs, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, '*pearson_result_*.csv'):
            matched_files.append(os.path.join(root, filename))
    if len( matched_files) ==1:
        pearson_result_file = matched_files[0].replace("\\","/")
    else:
        print(" ")
    
    #print(pearson_result_file)

    pearson_result_df = pd.read_csv(pearson_result_file)
    pearson_result_df = pearson_result_df.astype({
        'precursor charge': float,
        'precursor mono mass':float,
        'fragment mono mass': float,
        'fragment ms2 scan': float,
        'fragment ms2 rt':float,
        'fragment charge':float
        })
    
    pearson_result_df['fragment mz internsity pairs'] = pearson_result_df['fragment mz internsity pairs'].apply(literal_eval)

    r_source = robjects.r
    r_script = '''
        library(enviPat)
        library(data.table)
        
        FragIon.IsoPattern <- function(FragIons.chemform, ChargeZ){
          data(isotopes)
         

          Final.chemform <- paste0(FragIons.chemform, paste0('H', ChargeZ))
          CalPeaks <- isopattern(isotopes ,
                                 chemforms = Final.chemform, 
                                 charge = ChargeZ, 
                                 plotit = FALSE,
                                 algo = 2,
                                 emass = 0.00054857990924,
                                 threshold=0.1,
                                 verbose = FALSE)
          Cal.envelope <- envelope(CalPeaks,
                                   verbose = FALSE,
                                   resolution = 1E5,
                                   dmz = 0.01)
          Cal.mz <- vdetect(Cal.envelope,detect="centroid",plotit= FALSE,verbose=FALSE)
          Cal.mz <- as.data.table(Cal.mz[[1]])
          return(Cal.mz)
        }
    '''

    r_source(r_script)
    r_env_dir = "D:\Software\R\R-4.3.1"
    os.environ['R_HOME'] = r_env_dir
    
    for js_file in js_files:
        #print(js_file)
        name_str = js_file.replace(proteins_path+'/','')
        name_str = name_str.replace('.js','')
        
        index = int(name_str.replace('protein',''))
        if index <100:
            print(name_str)
            new_row_df = process_js_file(root_dir,js_file, pearson_result_df)
            if new_row_df.shape[0] != 0:
                new_row_df.to_csv(root_directory+'/findandcheck/'+name_str+'_info.csv', index=False)
                store_theoretical_peaks(root_directory, name_str, r_source)
            else:
                print(f'name_str')
        

    
def genenrate_term_frag(peak_ion_type,peak_ion_position,test_protein, test_seq):
    

    if peak_ion_type=='Y':
        start_position = int(test_protein.SEQLEN) - int(peak_ion_position)
        end_position = int(test_protein.SEQLEN)
        fragment_seq = test_seq[start_position:end_position]
        test_fragment = Protein(fragment_seq)
        fragment_formula = test_fragment.FORMULA
       
    if peak_ion_type=='B':
        end_position = int(peak_ion_position)
        fragment_seq = test_seq[0:end_position]
        test_fragment = Protein(fragment_seq)
        fragment_formula = test_fragment.FORMULA
        
        seq_dic = mass.Composition(sequence = fragment_seq)
        tmp_s = ""
        for k,v in seq_dic.items():
            if k == 'H':
                tmp_s += str(k)+str(int(v)-2)
            elif k=='O':
                tmp_s += str(k)+str(int(v)-1)
            else:    
                tmp_s += str(k)+str(v)
        
        fragment_formula = tmp_s
        
    return fragment_formula



def get_iso_peak_arr(r_source, chem_comp, charge):
    iso_df = r_source["FragIon.IsoPattern"](chem_comp,charge)
    iso_peak_arr = np.array(list(zip(list(iso_df[0]),list(iso_df[1]))),dtype=float)
    return iso_peak_arr



def get_real_isotopic(ms2_mzml_file,scan, mzRange):
    run = pymzml.run.Reader(ms2_mzml_file)
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(ms2_mzml_file, exp)
    spec_count = 0
    real_peaks = []
    for spectrum in exp:
        spec_count = spec_count + 1
        if spec_count ==scan:
            for peak in spectrum:
                if peak.getMZ() in Interval(min(mzRange),max(mzRange)):
                    real_peaks.append([peak.getMZ(),peak.getIntensity()])
    return real_peaks


def get_toppic_isotopic(file_path, frag_id):
    file_pd = pd.read_csv(file_path)
    tp_frag_df = file_pd[(file_pd['fragment id'] == frag_id)]
    tp_frag =  tp_frag_df['fragment mz internsity pairs'].apply(literal_eval)
    scaling_num = np.max(tp_frag[:,1]) / 100
    tp_frag[:,1] = (tp_frag[:,1]/ scaling_num)
    print("toppic碎片:\n",tp_frag)
    return tp_frag
    

def cal_ppm(mz1,mz2):
    ppm = ((mz1-mz2)/mz2)*1e6
    return ppm


def get_closest_peak(iso_peak_arr,ms_peak_arr):
    match_peak_arr = np.zeros_like(iso_peak_arr,dtype=float)
    ms_peak_mz = ms_peak_arr[:,0]
    for i in range(len(iso_peak_arr)):
        iso_peak_i = iso_peak_arr[i]
        err_ppm = cal_ppm(ms_peak_mz,iso_peak_i[0])
        match_peak_index = np.argmin(abs(err_ppm))
        match_peak = ms_peak_arr[match_peak_index]
        match_peak_arr[i] = match_peak
    return match_peak_arr



def iso_peak_match(iso_peak_arr,ms_peak_arr,peak_match_error):
    match_peak_arr = get_closest_peak(iso_peak_arr,ms_peak_arr)
    peak_err_arr = cal_ppm(match_peak_arr[:,0],iso_peak_arr[:,0])
    missing_peak_index = abs(peak_err_arr)>peak_match_error
    
    match_peak_arr[:,0][missing_peak_index]=iso_peak_arr[:,0][missing_peak_index]
    
    match_peak_arr[:,1][missing_peak_index]=0
    return match_peak_arr



def cal_PCC(iso_peak_arr,match_peak_arr):

    X = iso_peak_arr[:,1]
    Y = match_peak_arr[:,1]
    mean1 = np.mean(X)
    mean2 = np.mean(Y)
    tmp1 = X-mean1
    tmp2 = Y-mean2
    if np.all(tmp2)==0:
        return 0
    PCC = np.sum(tmp1*tmp2)/np.sqrt(np.sum(np.square(tmp1)))/np.sqrt(np.sum(np.square(tmp2)))
    return PCC



def plot_1_pro(iso_peak_arr,tp_frag,PCC_score):                
    mzml_peak = get_real_isotopic(ms2_mzml_file="run_pearson_0_x_file_path/mass_spectrum.mzML",scan=129, mzRange=[1764,1769])

    plt.figure(figsize=(8, 6))
    plt.ylim(0, 105)
    plt.scatter(iso_peak_arr[:,0],iso_peak_arr[:,1],c='red',alpha=0.9, edgecolors='none',s=100,label='Theoretical peaks')
    plt.vlines(iso_peak_arr[:,0], np.zeros(len(iso_peak_arr[:,0])), iso_peak_arr[:,1], colors='tab:orange',alpha=0.9,linewidth=1,linestyles='--',)

    plt.scatter(tp_frag[:,0],tp_frag[:,1],c='tab:blue',alpha=0.9, edgecolors='none',s=100,label='Actual peaks')
    plt.vlines(tp_frag[:,0], np.zeros(len(tp_frag[:,0])), tp_frag[:,1], colors='tab:blue',alpha=0.9,linewidth=2)
    
    mzml_peak = np.array(mzml_peak)
    ss= np.max(mzml_peak[:, 1])/100
    mzml_peak[:, 1] /= ss
    #plt.plot(mzml_peak[:,0],mzml_peak[:,1],color='black',label='Original spectrum')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel("m/z",fontsize=14, fontweight='bold')
    plt.ylabel("Relative abundance (%)",fontsize=14, fontweight='bold')
    plt.legend()
    plt.title(PCC_score)
    plt.show()





def fastaToDcit(fastaDB_path, root_directory):
    fastaDB_dict = {}
    for seq_record in SeqIO.parse(fastaDB_path, "fasta"):
        fastaDB_dict[str(seq_record.id)] = str(seq_record.seq)
    
    result_filePath = root_directory+'/findandcheck'
    if not os.path.exists(result_filePath):
        print(f": {result_filePath}")
        os.mkdir(result_filePath)
        
    with open(result_filePath + '/DB_sequence.json', 'w', encoding='utf-8') as jsonfile:
        json.dump(fastaDB_dict, jsonfile, ensure_ascii=False, indent=4)
    print("fastaDB_dict has been written")
    
    
def chemicalFormula_to_dict(formula):
    pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
    elements = defaultdict(int)
    matches = pattern.findall(formula)

    for (element, count) in matches:
        count = int(count) if count else 1
        elements[element] += count

    # Convert defaultdict to a regular dictionary for output
    elements_dict = dict(elements)

    #print(elements_dict)
    return elements_dict



def dict_to_chemicalFormula(formula_dic):
    tmp_s = ""
    for k,v in formula_dic.items():
        if k == 'H':
            tmp_s += str(k)+str(int(v))
        elif k=='O':
            tmp_s += str(k)+str(int(v))
        else:    
            tmp_s += str(k)+str(v)

    frag_formula = tmp_s

   
    
    return frag_formula


def add_modification_to_formula(modification, frag_formula):
    if modification == 1:
        mod_formula = {"C":34, "H":30,"O":4,"N":4, "Fe":1}  
    elif modification == 2:
        mod_formula =  {"C":2, "H":2,"O":1} 
    elif modification == 3:
        mod_formula =  {"C":1, "H":2} 
    elif modification == 4:
        mod_formula =  {"P":1,"O":3} 

    if modification !=0:
        formula_dic = chemicalFormula_to_dict(frag_formula)
        for i in mod_formula.keys(): 
            if i in formula_dic.keys():
                formula_dic[i] = formula_dic[i] + mod_formula[i]
            else:
                formula_dic[i] = mod_formula[i]
    
    frag_formula = dict_to_chemicalFormula(formula_dic)
    mono_mass = mass.calculate_mass(formula=frag_formula)
              
    return frag_formula
    


def annotationnn_seq_preprocess(annotationnn_seq):
    annotationnn_seq_remove_bracket = re.sub(r'\[[^]]*\]', '', annotationnn_seq) 
    annotationnn_seq_remove_bracket = annotationnn_seq_remove_bracket.replace('-', '')
    first_dot_index = annotationnn_seq_remove_bracket.find('.')
    second_dot_index = annotationnn_seq_remove_bracket.find('.', first_dot_index + 1)
    
    if first_dot_index != -1 and second_dot_index != -1:
        substring_between_dots = annotationnn_seq_remove_bracket[first_dot_index + 1:second_dot_index]
        new_str = ''.join(re.findall(r'[a-zA-Z]+', substring_between_dots))

    return new_str



def find_corresponding_position(complete_seq, str_1, position): #nopr
    str_1 = annotationnn_seq_preprocess(str_1)
    str_1_start_index = complete_seq.index(str_1)
    offset = position - str_1_start_index
    
    if 0 <= offset < len(str_1):
        if str_1[offset] == complete_seq[position]:
            print(offset)
            return offset,str_1[offset],str_1
        else:
            raise ValueError("Corresponding character does not match.")
    else:
        raise ValueError("Position out of range in str_1.")
    





def get_theoretical_formula(root_directory, peak_ion_type,peak_ion_position,protein_sequence_name,annotated_seq,ptm_info):
    with open(root_directory+'/findandcheck/DB_sequence.json','r') as uf:
        json_data = uf.read()
        fastadb = json.loads(json_data) 
        complete_seq = fastadb[protein_sequence_name]
        
    annotationnn_seq_remove_bracket = re.sub(r'\[[^]]*\]', '', annotated_seq)
    annotationnn_seq_remove_bracket = annotationnn_seq_remove_bracket.replace('-', '')
    first_dot_index = annotationnn_seq_remove_bracket.find('.')
    second_dot_index = annotationnn_seq_remove_bracket.find('.', first_dot_index + 1)
    substring_between_dots = annotationnn_seq_remove_bracket[first_dot_index + 1:second_dot_index]
    new_str = ''.join(re.findall(r'[a-zA-Z]+', substring_between_dots))
    
    str_1_start_index = complete_seq.index(new_str)

    if peak_ion_type=='B':  
        result = complete_seq[str_1_start_index:peak_ion_position+str_1_start_index]
        #print(f"Processed substring: {result}")
        
    elif peak_ion_type=='Y': 
        result = complete_seq[peak_ion_position+str_1_start_index: str_1_start_index+len(new_str)]
        #print(f"Processed substring: {result}")
        
    test_protein = Protein(result)

    if peak_ion_type=='B': 
        fragment_formula = genenrate_term_frag(peak_ion_type='B',peak_ion_position=len(result),test_protein=test_protein, test_seq=result)
    elif peak_ion_type=='Y':
        fragment_formula = genenrate_term_frag(peak_ion_type='Y',peak_ion_position=len(result),test_protein=test_protein, test_seq=result)
    
    if ptm_info is None:
       
        return fragment_formula, mass.calculate_mass(formula=fragment_formula)
    else:
        ptm_abbrv = ptm_info[0]
        ptm_left_position = ptm_info[2]
        ptm_right_position = ptm_info[3]
        ptm_target_amoniAcid = ptm_info[4]
        
        if peak_ion_type=='B':
            if ptm_right_position <= (peak_ion_position+str_1_start_index):
                if ptm_abbrv=='Acetyl':
                    
                    frag_formula = add_modification_to_formula(modification=2, frag_formula=fragment_formula)
                    return frag_formula, mass.calculate_mass(formula=frag_formula)
            elif ptm_left_position >= (peak_ion_position+str_1_start_index):
                
                return fragment_formula, mass.calculate_mass(formula=fragment_formula)
            
        elif peak_ion_type=='Y':
            if ptm_right_position <= (peak_ion_position+str_1_start_index):
                return fragment_formula, mass.calculate_mass(formula=fragment_formula)
            elif ptm_left_position >= (peak_ion_position+str_1_start_index):
                if ptm_abbrv=='Acetyl':
                    frag_formula = add_modification_to_formula(modification=2, frag_formula=fragment_formula)
                    return frag_formula, mass.calculate_mass(formula=frag_formula)
                
def all_nan(lst):
    
    arr = np.array(lst)
    
    return np.all(np.isnan(arr))

def store_theoretical_peaks(root_directory, name_str, r_source):
    prsm_info_df = pd.read_csv(root_directory+'/findandcheck/'+name_str+'_info.csv')
    prsm_info_df = prsm_info_df.astype({
        "fragment charge":float,
        "ion_position":	int,
        'ion_display_position': int, 
        'fragment mono mass': float, 
        'precursor mono mass': float,
        'matched_fragment_number': int,
        'matched_peak_number': int,
        'ptm_occurence_left_pos': float,
        'ptm_occurence_right_pos': float
        })
    prsm_info_df['fragment_mz_intensity_pairs_normalization'] = prsm_info_df['fragment_mz_intensity_pairs_normalization'].apply(literal_eval)
    
    count = 0
    addIsotop_df = pd.DataFrame()
    for  _, rt_row in prsm_info_df.iterrows():
        count +=1
        if count <100000:
            
            peak_mass = float(rt_row['fragment mono mass'])
            #print(peak_mass)
            peak_peak_charge = int(rt_row['fragment charge'])
            peak_ion_type = rt_row['ion_type']
            peak_ion_position = int(rt_row['ion_position'])
            protein_sequence_name = rt_row['sequence_name']
            annotated_seq=rt_row['annotated_seq']
            ptm_info = [rt_row['ptm_abbrv'], float(rt_row['ptm_mono_mass']), float(rt_row['ptm_occurence_left_pos']),float(rt_row['ptm_occurence_right_pos']),rt_row['ptm_occurence_anno'] ]
            if pd.isna(rt_row['ptm_abbrv']):
                ptm_info = None
            tp_frag = rt_row['fragment_mz_intensity_pairs_normalization']
            tp_frag = np.array(tp_frag)
    
            frag_formula, frag_mass = get_theoretical_formula(root_directory=root_directory,
                                                   peak_ion_type = peak_ion_type,
                                                   peak_ion_position = peak_ion_position,
                                                   protein_sequence_name = protein_sequence_name,
                                                   annotated_seq = annotated_seq,
                                                   ptm_info = ptm_info)
            try:
                assert frag_mass in Interval(peak_mass-5,peak_mass+5), f" warn"
            except AssertionError as e:
                print(e)
         
            
            #print(f'frag_formula : {frag_formula}')
            iso_peak_arr = get_iso_peak_arr(r_source, frag_formula ,peak_peak_charge )
         
            match_peak_arr = iso_peak_match(iso_peak_arr = iso_peak_arr,
                                            ms_peak_arr = tp_frag,
                                            peak_match_error=200)
            
            #print("match_peak_arr:\n",match_peak_arr)
            
            PCC_score = cal_PCC(iso_peak_arr,match_peak_arr)
            
            #plot_1_pro(iso_peak_arr,tp_frag,PCC_score)
            new_row = rt_row.copy()
            new_row['theoretical_formula'] = frag_formula
            new_row['frag_mass'] = frag_mass
            new_row['iso_peak_arr'] = iso_peak_arr
            new_row['match_peak_arr'] = match_peak_arr 
            new_row['PCC_score'] = PCC_score 
            new_row_df = pd.DataFrame([new_row])
            addIsotop_df = pd.concat([addIsotop_df, new_row_df], ignore_index=True)
            
    addIsotop_df.to_csv(root_directory+'/findandcheck/'+name_str+'_pcc_info.csv', index=False)
    
        


def qualified_prsm(root_dic_list):
    with open(root_dic_list[0]+'/findandcheck/DB_sequence.json','r') as uf:
        json_data = uf.read()
        fastadb = json.loads(json_data) 
        complete_seq = fastadb.keys()
    complete_seq_set = set(complete_seq)
    valid_protein = []
    matched_files = []
    for root_dic in root_dic_list:
        pcc_info_folder = root_dic + '/findandcheck/'
        for root, dirs, files in os.walk(pcc_info_folder ):
            for filename in fnmatch.filter(files, '*_pcc_info.csv'):  
                matched_files.append(os.path.join(root, filename)) 
        
    results=[]
    for pccInfo in matched_files:
        pcc_df = pd.read_csv(pccInfo)
        pcc_df = pcc_df.astype({
            'PCC_score': float,
            'ion_position':float
            })
        filter_df = pcc_df[(pcc_df['PCC_score']>0.8)]
        filter_df = filter_df.drop_duplicates(subset=['ion_type', 'ion_position', 'annotated_seq'], keep='first')
        
        seq_counts = filter_df['annotated_seq'].value_counts()
        filtered_seqs = seq_counts[seq_counts > 4].index.tolist()
        
       

        if len(filtered_seqs) > 0:
            valid_protein.append(pcc_df.loc[0,'sequence_name'])
        
            protein_name = pcc_df.loc[0, 'sequence_name']
            protein_description = pcc_df.loc[0, 'sequence_description']
            proteoform_count = 0
            for seq in filtered_seqs:
                annotationnn_seq_remove_bracket = re.sub(r'\[[^]]*\]', '', seq)
                annotationnn_seq_remove_bracket = annotationnn_seq_remove_bracket.replace('-', '')
                first_dot_index = annotationnn_seq_remove_bracket.find('.')
                second_dot_index = annotationnn_seq_remove_bracket.find('.', first_dot_index + 1)
                substring_between_dots = annotationnn_seq_remove_bracket[first_dot_index + 1:second_dot_index]
                new_str = ''.join(re.findall(r'[a-zA-Z]+', substring_between_dots))
                
                temp_df = filter_df[filter_df['annotated_seq'] ==seq].reset_index(drop=True)
                
                
                proteoform_mass = temp_df.loc[0,'precursor mono mass']
                proteoform_deconv_id = temp_df.loc[0,'precursor id']
                matched_fragment_number = temp_df.loc[0,'matched_fragment_number']
                matched_peak_number = temp_df.loc[0,'matched_peak_number']
                
                cleavage_count = seq_counts[seq]
                cleavage_rate = cleavage_count / ((len(new_str)-1)*2)
                proteoform_count +=1
                results.append({
                    'protein_name': protein_name,
                    'protein_description': protein_description,
                    'proteoform_id':proteoform_count,
                    'proteoform_deconv_id':proteoform_deconv_id,
                    'proteoform_mass': proteoform_mass,
                    'annotated_seq': seq,
                    'matched_fragment_number':matched_fragment_number,
                    'qualified_cleavage_count': cleavage_count,
                    'cleavage_rate':  cleavage_rate,
                    'matched_peak_number':matched_peak_number
                })

    results_df = pd.DataFrame(results)

    results_df.to_csv('run_pearson_0_x_file_path/prsm_output_0_8_quatify1027_2.csv', index=False, encoding='utf-8')

    valid_protein_set = set(valid_protein)
    not_detected = complete_seq_set - valid_protein_set

    
        
        
        
    
if __name__ == "__main__":
    
    
    start_time = time.time()

    root_directory = "run_pearson_0_x_file_path"
    fastaToDcit(fastaDB_path = "path_/fasta/seq_fasta.fasta", root_directory = root_directory)
    findcheck(root_directory)
    
    root_directory_list = [
                           "run_pearson_0_x_file_path"
                           ]
    qualified_prsm(root_dic_list=root_directory_list)
   
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"run time: {elapsed_time:.2f} s")
    

    
    
    
    
    
