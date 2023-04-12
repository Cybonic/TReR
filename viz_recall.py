import argparse
import pandas as pd
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

SIZE = 25


def color_recall_graph(num_points,color='k',scale=15):
    c = np.array([color]*num_points)
    s = np.ones(num_points)*scale
    return(c,s)

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
    
    TRAIN_SPLIT = str(0.1)
    import glob
    files_struct = {'00':[],'02':[],'05':[],'06':[],'08':[]}
    sequences = list(files_struct.keys())
    for path in glob.glob(FLAGS.root, recursive=True):
        if path.endswith('csv'):
            path_struct = path.split('/')
            if path_struct[6] == TRAIN_SPLIT: # slect only the results from the specific trainsplit 
                seq = path_struct[5]
                files_struct[seq].append(path)

                print(path)
    
    

    

    colors= ['cadetblue','k','navy','maroon','forestgreen','olive','darkseegreen','darkorchid']
    
    for i in sequences:
        fig = Figure(figsize=(5, 4), dpi=100,)
        fig, ax = plt.subplots()
        files = files_struct[i]
        for file,c in zip(files,colors):
            # Get file and check if it exists
            print(f'File: {file}')
            modelname = file.split('/')[1].split('_')[0]
            file_path = file
            if not os.path.isfile(file_path):
                raise NameError("File does not exist")
            # Load data
            df = pd.read_csv(file_path)
            # Plot
            if 'ScanContext' == modelname:
                ax.plot(df['top'][:25],df['recall'][:25],color = c,label=modelname + '_Baseline',linewidth=5,linestyle = '--')
            else:
                ax.plot(df['top'][:25],df['base'][:25],color = c,label=modelname + '_Baseline',linewidth=5,linestyle = '--')
                ax.plot(df['top'][:25],df['reranked'][:25],color = c,label=modelname + '_ReR',linewidth=5,linestyle = '-')
        

        #ax.legend(names,fontsize=16)
        plt.xlabel('N-Number of top candidates',fontsize=15)
        plt.ylabel('Recall@N',fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim([0, 1])
        plt.legend()
        
        path_to_save = os.path.join("fig",TRAIN_SPLIT)
        if not os.path.isdir(path_to_save):
            os.makedirs(path_to_save)
        
        file = os.path.join(path_to_save,f"{i}.png")
        #file_to_save = os.path.join("fig",i + '_zoom.png')
        plt.savefig(file)
        #plt.show()
    