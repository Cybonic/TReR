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

def parse_file_diff(file,top_cand):
    
    model_name = file.split("/")[-1].split('-')[0]
    df = pd.read_csv(file)

    if 'base' in df:
        scores = df[['reranked','base']].values
    else:
        scores = df[['recall','recall_rr']].values
    
    if top_cand != None:
        top_cand = np.array(top_cand)-1
        scores = scores[top_cand,:]

    rr_score = np.round(np.mean(scores[:,1]-scores[:,0]),2)
    return(model_name,rr_score)

def parse_file_scores(file,top_cand,metric='recall_rr'):
    
    model_name = file.split("/")[-1].split('-')[0]
    df = pd.read_csv(file)

    if 'base' in df:
        scores = df[['reranked']].values
    else:
        scores = df[[metric]].values
    
    if top_cand != None:
        top_cand = np.array(top_cand)-1
        scores = scores[top_cand,:]

    rr_score = np.round(scores,2)
    return(model_name,rr_score)


def model_wise_mean_diff_score(files,cand=[1,5,10]):
    
    # Parse files based on the model name
    model_list = []
    score_vec  = []
    for file in files:
        #model,score = parse_file_scores(file,cand)
        model,score = parse_file_diff(file,cand)
        model_list.append(model)
        score_vec.append(score)

    # Compute the average performance of each model
    un_m = np.unique(model_list)
    best_score = []
    best_model = []
    for mm in un_m:
        compare = [i for i,mi in enumerate(model_list) if mi in mm]
        ms = np.round(np.mean(np.array(score_vec)[compare]),4)
        best_score.append(ms)
        best_model.append(mm)
    return(best_model,best_score)

def model_wise_baseline_scores(files,cand=[1,5,10]):
     # Parse files based on the model name
    model_list = []
    score_vec  = []
    for file in files:
        model,score = parse_file_scores(file,cand,metric='recall')
        #model,score = parse_file_diff(file,cand)
        model_list.append(model)
        score_vec.append(score)

    # Compute the average performance of each model
    un_m = np.unique(model_list)
    best_score = []
    best_model = []
    for mm in un_m:
        compare = [i for i,mi in enumerate(model_list) if mi == mm]
        ms = np.array(score_vec)[compare]
        best_score.append(ms)
        best_model.append(mm)
    return(best_model,best_score)

def model_wise_mean_scores(files,cand=[1,5,10],metric='recall_rr'):
    
    # Parse files based on the model name
    model_list = []
    score_vec  = []
    for file in files:
        model,score = parse_file_scores(file,cand,metric)
        #model,score = parse_file_diff(file,cand)
        model_list.append(model)
        score_vec.append(score)

    # Compute the average performance of each model
    un_m = np.unique(model_list)
    best_score = []
    best_model = []
    for mm in un_m:
        #print(mm)
        compare = [i for i,mi in enumerate(model_list) if mi == mm]
        ms = np.array(score_vec)[compare]
        best_score.append(ms)
        best_model.append(mm)
    return(best_model,best_score)


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


def results_to_pandas(files,target):
    import pandas as pd
    for key,file in files.items():
        print(key)
        unique_models,scores = model_wise_mean_scores(file,[1,5,10],metric='recall_rr')
        for unique_model, score in zip(unique_models,scores):
            if unique_model.startswith(target):
                print(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
      '--root', '-f',
      type=str,
      default = "results/cross_seq_transformer_n_enc_study/**", # logistic_loss prob_rank_loss "results/margin_rank_loss/ablation/**"
      required=False,
      help='Dataset to train with. No Default',
    )
    FLAGS, unparsed = parser.parse_known_args()
    
    files_struct = {'08':[]}#,'02':[],'05':[],'06':[],'08':[]}
    files = get_all_files(FLAGS.root)

    # model_struct = {'ORCHNet_pointnet':[],'VLAD_pointnet':[],'SPoC_pointnet':[],'GeM_pointnet':[]}
    
    unique_models,scores = model_wise_mean_diff_score(files,[1,5,10])

    TopScore = np.argsort(scores)[::-1]
    for j in TopScore:
        print(f"{unique_models[j]}=={scores[j]}")

    #exit()
    
    #model_struct = {'VLAD_pointnet':[],'GeM_pointnet':[],'ORCHNet_pointnet':[],'SPoC_pointnet':[]}
    model_struct = {'GeM_pointnet':[]}
    files = parse_file_struct(files,model_struct)
    
    for key, values in files.items():
        print("\n==================================")
        print(key)
        #files_struct = {'00':[],'02':[],'05':[],'06':[],'08':[]}
        files_struct = {'00':[],'02':[],'05':[],'06':[],'08':[]}
        file_seq = parse_file_struct(values,files_struct)
        # TranformerEncoder_BFT_x1_headx1_max_fc_drop
        results_to_pandas(file_seq,'TranformerEncoder')
    
    
    for mm, values in zip(unique_models,scores):

        print(mm)
        values = np.mean(values,axis=0)
        print(values)
    
    
    unique_models,scores = model_wise_baseline_scores(files,[1,5,10,15])
    for mm, values in zip(unique_models,scores):
        print(mm)
        values = np.mean(values,axis=0)
        print(values)
    exit()
    
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
  
            


    
    
    
    