import argparse
import pandas as pd
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

#plt.rcParams['text.usetex'] = True

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
      #default = "results/cross_seq_transformer_n_enc_study/logistic_loss/VLAD_pointnet/**",
      default = "/home/tiago/Dropbox/SHARE/deepisrpc/reranking/results/paper_svg_comparison/**",
      required=False,
      help='Dataset to train with. No Default',
    )

    SEQ_IDX = 5
    FLAGS, unparsed = parser.parse_known_args()
    
    TRAIN_SPLIT = 'rerranking'
    target = 'TranformerEncoder'
    import glob
    
    y_min = 1
    y_max = 0
    
    files_struct = {'00':[],'02':[],'05':[],'06':[],'08':[]}
    #files_struct = {'08':[]}
    files_struct = []
    for path in glob.glob(FLAGS.root, recursive=True):
        if path.endswith('csv'):
            files_struct.append(path)
    

    

    colors=['cadetblue','navy','maroon','forestgreen','olive','plum','darkorchid','k']

    
    for file in files_struct:
        
        fig = Figure(figsize=(5, 4), dpi=100,)
        fig, ax = plt.subplots()
        ax.grid(True)
        seq = file.split('.')[0].split('/')[-1]
     
        df = pd.read_csv(file)
        modelnames = list(df.columns)
        modelnames = ["LOGG3D","LOGG3D+SVG","LOGG3D+TreR",'LOGG3D+QE']
        modelnames_real = ['LOGG3DNet','LOGG3DNet+SVG',"LOGG3DNet+TReR(our)",f'LOGG3DNet+{chr(945)}QE']
        for i,(modelname,c) in enumerate(zip(modelnames,colors)):
            ax.plot(df.index[0:25],df[modelname][0:25],color = c,label=modelnames_real[i],linewidth=2,linestyle = '-')
           


        #ax.legend(names,fontsize=16)
        plt.xlabel('N-Number of top candidates',fontsize=15)
        plt.ylabel('Recall@N',fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        #plt.ylim([0, 1])
        plt.legend()
        
        path_to_save = os.path.join("fig",TRAIN_SPLIT)
        if not os.path.isdir(path_to_save):
            os.makedirs(path_to_save)
        
        file = os.path.join(path_to_save,f"{seq}.pdf")
        #plt.ylim([y_min,y_max])
        #file = os.path.join(path_to_save,i + '_zoom.png')
        plt.savefig(file,format="pdf", bbox_inches="tight")
        #plt.show()
    