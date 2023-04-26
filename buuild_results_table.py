import argparse
import pandas as pd
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import glob

SIZE = 25

colors= ['cadetblue','k','navy','maroon','forestgreen','olive','darkseegreen','darkorchid']

def color_recall_graph(num_points,color='k',scale=15):
    c = np.array([color]*num_points)
    s = np.ones(num_points)*scale
    return(c,s)

def get_all_files(root):
    files = []
    for path in glob.glob(root, recursive=True):
        if path.endswith('csv'):
            files.append(path)
    
    return files

def parse_file_struct(files,files_struct):

    tags = list(files_struct.keys())
    for path in files:
        if path.endswith('csv'):
            path_struct = path.split('/')
            for tag in tags:
                if tag in path_struct:
                    #seq = path_struct[5]
                    files_struct[tag].append(path)
                    #print(path)
    return files_struct

def search_for_best_model(files,top_cand = None):
    best_model_dict = {}
    
    #idx = [1,5,10,15]
    for key ,value  in files.items():
        seq_files =  parse_file_struct(value,files_struct)
        best_model = []
        scores_vec = []
        
        for kk, vfile in seq_files.items():
            nam_v = []
            revall_seq = []
            for file in vfile:
                nam_v.append(file.split("/")[-1].split('-')[0])
                df = pd.read_csv(file)
                if 'base' in df:
                    scores = df[['reranked','base']].values
                else:
                    scores = df[['recall_rr','recall']].values
                if top_cand != None :
                    assert isinstance(top_cand,list),"Top cand is not a list format"
                    scores = scores[top_cand,:]
                #rr_score = np.round(np.sum(scores[idx,0]-scores[idx,1]),2)
                rr_score = np.round(np.sum(scores[:,0]-scores[:,1]),2)
                revall_seq.append(rr_score)
            
            scores_vec.extend(revall_seq)
            best_model.extend(nam_v)
            #print(f"{len(scores_vec)} == {len(best_model)}" )

        best_model_dict[key]={'model':best_model,'scores':scores_vec}
    return best_model_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
      '--root', '-f',
      type=str,
      default = "results/paperv2/batch_first_false/**", # logistic_loss prob_rank_loss "results/margin_rank_loss/ablation/**"
      required=False,
      help='Dataset to train with. No Default',
    )
    FLAGS, unparsed = parser.parse_known_args()
    
    files_struct = {'00':[],'02':[],'05':[],'06':[],'08':[]}
    files = get_all_files(FLAGS.root)

    model_struct = {'ORCHNet_pointnet':[],'VLAD_pointnet':[],'SPoC_pointnet':[],'GeM_pointnet':[]}
    files = parse_file_struct(files,model_struct)
    best_model_dict = search_for_best_model(files,[1,5,10])
    # print(best_model_dict)
    
    best_model_score = []
    best_model_model = []
    
    for model,values in best_model_dict.items():
        m = values['model']
        s = values['scores']
        
        un_m = np.unique(m)

        for mm in un_m:
            compare = [i for i,mi in enumerate(m) if mi in mm]
            ms = np.round(np.mean(np.array(s)[compare]),2)
            best_model_score.append(ms)
            best_model_model.append(mm)
            # print(f"{mm} = {ms}")
            #best_model.append(f"{mm} = {ms}")

    un_m = np.unique(best_model_model)
    best_score = []
    best_model = []
    for mm in un_m:
        compare = [i for i,mi in enumerate(best_model_model) if mi in mm]
        ms = np.round(np.mean(np.array(best_model_score)[compare]),4)
        best_score.append(ms)
        best_model.append(mm)
        #print(f"{mm} = {ms}")

    TopScore = np.argsort(best_score)[::-1]
    for j in TopScore:
        print(f"{best_model[j]}=={best_score[j]}")
  
            


    
    
    
    