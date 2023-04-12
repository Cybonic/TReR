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

def save_file_based_model(root,path_to_save,model):
    
    path_to_save = os.path.join(path_to_save,model)
    if not os.path.isdir(path_to_save):
        os.makedirs(path_to_save)

    split = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    
    top = 5

    files_struct = {'00':[],'02':[],'05':[],'06':[],'08':[]}
    sequences = list(files_struct.keys())
    for path in glob.glob(root, recursive=True):
        if path.endswith('csv'):
            path_struct = path.split('/')
            if path_struct[3] == model: # slect only the results from the specific trainsplit 
                seq = path_struct[5]
                files_struct[seq].append(path)
                print(path)

    for i in sequences:
        fig = Figure(figsize=(5, 4), dpi=100,)
        fig, ax = plt.subplots()
        
        for c, top in zip(colors,[1,5,10,20]):
            
            files = files_struct[i]
            xx = []
            yy =[]
            for file in files:
                # Get file and check if it exists
                print(f'File: {file}')
                modelname = file.split('/')[3].split('_')[0]
                split_value = float(file.split('/')[6])
                file_path = file
                if not os.path.isfile(file_path):
                    raise NameError("File does not exist")
                # Load data
                df = pd.read_csv(file_path)
                
                xx.append(split_value)
                yy.append([df['base'][top],df['reranked'][top]])

            yy = np.array(yy)
            xx = np.array(xx)
            sort_xx = np.argsort(xx)
            yy = yy[sort_xx,:]
            xx = xx[sort_xx]
            ax.plot(xx,yy[:,0],color = c,label=f"@{top}" + '_Baseline',linewidth=5,linestyle = '--')
            ax.plot(xx,yy[:,1],color = c,label=f"@{top}" + '_AReR',linewidth=5,linestyle = '-')
        

        #ax.legend(names,fontsize=16)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
        file = os.path.join(path_to_save,f"{i}_zoom.png")
        #file_to_save = os.path.join("fig",i + '_zoom.png')
        plt.savefig(file)


        plt.xlabel('N-Number of top candidates',fontsize=15)
        plt.ylabel('Recall@N',fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim([0, 1])
        plt.legend()
        
        
        file = os.path.join(path_to_save,f"{i}.png")
        plt.savefig(file)
        #plt.show()

def save_file_based_seq(root,path_to_save,split):

    path_to_save = os.path.join(path_to_save,split)
    if not os.path.isdir(path_to_save):
        os.makedirs(path_to_save)

    files_struct = {'00':[],'02':[],'05':[],'06':[],'08':[]}
    sequences = list(files_struct.keys())
    for path in glob.glob(root, recursive=True):
        if path.endswith('csv'):
            path_struct = path.split('/')
            if path_struct[6] == split: # slect only the results from the specific trainsplit 
                seq = path_struct[5]
                files_struct[seq].append(path)
                print(path)

    for i in sequences:
        fig = Figure(figsize=(5, 4), dpi=100,)
        fig, ax = plt.subplots()
        files = files_struct[i]
        y_min = 1
        y_max = 0
        for file,c in zip(files,colors):
            # Get file and check if it exists
            print(f'File: {file}')
            modelname = file.split('/')[3].split('_')[0]
            file_path = file
            if not os.path.isfile(file_path):
                raise NameError("File does not exist")
            # Load data
            df = pd.read_csv(file_path)
            
            ymib = np.min(np.min(df[['base','reranked']][0:26],axis=1))
            ymab = np.max(np.max(df[['base','reranked']][0:26],axis=1))
            
            if ymib<y_min:
                y_min = ymib

            if ymab>y_max:
                y_max = ymab
            # Plot
            if 'ScanContext' == modelname:
                ax.plot(df['top'][:25],df['recall'][:25],color = c,label=modelname + '_Baseline',linewidth=5,linestyle = '--')
            else:
                ax.plot(df['top'][:25],df['base'][:25],color = c,label=modelname + '_Baseline',linewidth=5,linestyle = '--')
                ax.plot(df['top'][:25],df['reranked'][:25],color = c,label=modelname + '_AReR',linewidth=5,linestyle = '-')
        

        #ax.legend(names,fontsize=16)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim([0, 25.5])
        plt.ylim([y_min-0.01,ymab+0.04])
        file = os.path.join(path_to_save,f"{i}_zoom.png")
        #file_to_save = os.path.join("fig",i + '_zoom.png')
        plt.savefig(file)


        plt.xlabel('N-Number of top candidates',fontsize=15)
        plt.ylabel('Recall@N',fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim([0, 1])
        plt.xlim([0, 26])
        plt.legend()
        
        
        file = os.path.join(path_to_save,f"{i}.png")
        #file_to_save = os.path.join("fig",i + '_zoom.png')
        plt.savefig(file)
        #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
      '--root', '-f',
      type=str,
      default = "results/prob_rank_loss/**",
      required=False,
      help='Dataset to train with. No Default',
    )
    FLAGS, unparsed = parser.parse_known_args()
    
    TRAIN_SPLIT = str(0.2)

   
    #for m in ['VLAD_pointnet','SPoC_pointnet','ORCHNet_pointnet','GeM_pointnet']:
    #    save_file_based_model(FLAGS.root,'fig/train',m)

    save_file_based_seq(FLAGS.root,'fig/margin_rank_loss/split',TRAIN_SPLIT)

    
    
    
    