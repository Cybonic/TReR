
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch
from time import time


class alpha_QE():
    def __init__(self,k=25,radius=[2,5],n_samples=10):
        self.k = k
        self.radius = radius
        self.n_samples = n_samples
      


    def evaluation(self,dataloader):
        # Dictionary to store the number of true positives (for global desc. metrics) for different radius and NN number
        global_metrics = {'tp': {r: [0] * self.k for r in self.radius}}
        global_metrics['tp_rr'] = {r: [0] * self.k for r in self.radius}
        global_metrics['RR'] = {r: [] for r in self.radius}
        global_metrics['RR_rr'] = {r: [] for r in self.radius}
        global_metrics['t_RR'] = []

        for emb,scores,pos in tqdm(dataloader):
            
            query_emb = emb['q'].detach().numpy()
            map_embeddings = emb['map'].squeeze().detach().numpy()
            scores = scores.squeeze().detach().numpy()
            map_positions = pos['map'].squeeze().detach().numpy()
            query_pos = pos['q'].detach().numpy()
            # Re-Ranking:
            alpha = 0.01 # 0.1
            topk = query_pos.shape[1]
            tick = time()
            topk_gds = map_embeddings
            topk_gd_dists = scores
            query_emb = query_emb.reshape(1,-1)

            new_q = np.vstack([topk_gds[j] * topk_gd_dists[j]**alpha for j in range(len(topk_gds))])
            new_q = np.vstack([new_q, query_emb])
            alpha_avg_descriptor = np.average(new_q, axis=0).reshape(1,-1)
            rr_embed_dist = np.linalg.norm(topk_gds - alpha_avg_descriptor, axis=1)
            rr_nn_ndx = np.argsort(rr_embed_dist)[:self.k]

            t_rerank = time() - tick
            global_metrics['t_RR'].append(t_rerank)

            delta_rerank = query_pos - map_positions[rr_nn_ndx]
            euclid_dist_rr = np.linalg.norm(delta_rerank, axis=1)
                

            # Count true positives for different radius and NN number
            global_metrics['tp'] = {r: [global_metrics['tp'][r][nn] + (1 if (scores[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}
            global_metrics['tp_rr'] = {r: [global_metrics['tp_rr'][r][nn] + (1 if (euclid_dist_rr[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}
            global_metrics['RR'] = {r: global_metrics['RR'][r]+[next((1.0/(i+1) for i, x in enumerate(scores <= r) if x), 0)] for r in self.radius}
            global_metrics['RR_rr'] = {r: global_metrics['RR_rr'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist_rr <= r) if x), 0)] for r in self.radius}


        # Calculate mean metrics
        global_metrics["recall"] = {r: [global_metrics['tp'][r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        global_metrics["recall_rr"] = {r: [global_metrics['tp_rr'][r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        global_metrics['MRR'] = {r: np.mean(np.asarray(global_metrics['RR'][r])) for r in self.radius}
        global_metrics['MRR_rr'] = {r: np.mean(np.asarray(global_metrics['RR_rr'][r])) for r in self.radius}
        global_metrics['mean_t_RR'] = np.mean(np.asarray(global_metrics['t_RR']))


        return global_metrics

def load_data(root,model_name,seq,train_percentage,dataset='new',batch_size=50):
    from dataloaders.alphaqedata import AlphaQEData
    #if dataset == 'new':
    train_data = AlphaQEData(root,model_name,seq)
    train_size = int(len(train_data)*train_percentage)
    test_size = len(train_data) - train_size

    train, test =torch.utils.data.random_split(train_data,[train_size,test_size])
    #trainloader = DataLoader(train,batch_size = int(len(train)),shuffle=True)
    trainloader = DataLoader(train,batch_size = len(test),shuffle=True)
    testloader = DataLoader(test,batch_size = 1) #

    return trainloader,testloader,train_data.get_max_top_cand(),str(train_data)

if __name__ == '__main__':
    
    

    root = '/home/tiago/Dropbox/RAS-publication/predictions/paper/kitti/place_recognition'

    train_size = 0.2
    device = 'cuda:0'
    #device = 'cpu'

    Models = ['VLAD_pointnet', 'ORCHNet_pointnet' ,'SPoC_pointnet', 'GeM_pointnet']
    sequences = ['00','02','05','06','08']
    dataset_type = 'new'


    
    for model_name in Models:
        print(model_name)
        for seq in sequences:
            print(seq)
            torch.manual_seed(0)
            np.random.seed(0)
            #seq = '02'
            trainloader,testloader,max_top_cand,datasetname = load_data(root,model_name,seq,train_size,dataset_type,batch_size=100)
            model  = alpha_QE(k=max_top_cand,radius=[5],n_samples = len(testloader))
            #loader = alpha.AlphaQEData()
            
            global_metrics = model.evaluation(testloader)
            recall = np.array(global_metrics['recall'][5])
            print(recall[[0,4,9]])

            recall = np.array(global_metrics['recall_rr'][5])
            print(recall[[0,4,9]])
