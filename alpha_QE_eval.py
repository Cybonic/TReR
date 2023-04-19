
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch
from time import time

class alpha_QE():
    def __init__(self,alpha=0.1,k=25,radius=[2,5],n_samples=10):
        self.k = k
        self.radius = radius
        self.n_samples = n_samples
        self.alpha = alpha 

    def evaluation(self,dataloader):
        # Dictionary to store the number of true positives (for global desc. metrics) for different radius and NN number
        global_metrics = {'tp': {r: [0] * self.k for r in self.radius}}
        global_metrics['tp_rr'] = {r: [0] * self.k for r in self.radius}
        global_metrics['RR'] = {r: [] for r in self.radius}
        global_metrics['RR_rr'] = {r: [] for r in self.radius}
        global_metrics['t_RR'] = []

        for emb,scores,pos in tqdm(dataloader):
            
            query_emb = emb['q'].squeeze().detach().numpy()
            topk_gds = emb['map'].squeeze().detach().numpy()
            topk_gd_dists = scores.squeeze().detach().numpy()
            map_positions = pos['map'].squeeze().detach().numpy()
            query_pos = pos['q'].detach().numpy()

            # Re-Ranking:
            topk = query_pos.shape[1]
            
            tick = time()
            new_q = np.vstack([topk_gds[j] * topk_gd_dists[j]**self.alpha for j in range(len(topk_gds))])
            new_q = np.vstack([new_q, query_emb])
            alpha_avg_descriptor = np.average(new_q, axis=0)
            rr_embed_dist = np.linalg.norm(topk_gds - alpha_avg_descriptor, axis=1)
            rr_nn_ndx = np.argsort(rr_embed_dist)[:self.k]

            t_rerank = time() - tick
            global_metrics['t_RR'].append(t_rerank)

            delta_rerank = query_pos - map_positions[rr_nn_ndx]
            euclid_dist_rr = np.linalg.norm(delta_rerank, axis=1)
                

            # Count true positives for different radius and NN number
            global_metrics['tp'] = {r: [global_metrics['tp'][r][nn] + (1 if (topk_gd_dists[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}
            global_metrics['tp_rr'] = {r: [global_metrics['tp_rr'][r][nn] + (1 if (euclid_dist_rr[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}
            global_metrics['RR'] = {r: global_metrics['RR'][r]+[next((1.0/(i+1) for i, x in enumerate(topk_gd_dists <= r) if x), 0)] for r in self.radius}
            global_metrics['RR_rr'] = {r: global_metrics['RR_rr'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist_rr <= r) if x), 0)] for r in self.radius}


        # Calculate mean metrics
        global_metrics["recall"] = {r: [global_metrics['tp'][r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        global_metrics["recall_rr"] = {r: [global_metrics['tp_rr'][r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        global_metrics['MRR'] = {r: np.mean(np.asarray(global_metrics['RR'][r])) for r in self.radius}
        global_metrics['MRR_rr'] = {r: np.mean(np.asarray(global_metrics['RR_rr'][r])) for r in self.radius}
        global_metrics['mean_t_RR'] = np.mean(np.asarray(global_metrics['t_RR']))

        return global_metrics
    
    def print_results(self, global_metrics):
        # Global descriptor results are saved with the last n_k entry
        print('\n','Initial Retrieval:')
        recall = global_metrics['recall']
        for r in recall:
            print(f"Radius: {r} [m] : ")
            print(f"Recall@N : ", end='')
            for x in recall[r]:
                print("{:0.1f}, ".format(x*100.0), end='')
            print("")
            print('MRR: {:0.1f}'.format(global_metrics['MRR'][r]*100.0))
        
        print('\n','Re-Ranking:')
        recall_rr = global_metrics['recall_rr']
        for r_rr in recall_rr:
            print(f"Radius: {r_rr} [m] : ")
            print(f"Recall@N : ", end='')
            for x in recall_rr[r_rr]:
                print("{:0.1f}, ".format(x*100.0), end='')
            print("")
            print('MRR: {:0.1f}'.format(global_metrics['MRR_rr'][r_rr]*100.0))
        print('Re-Ranking Time: {:0.3f}'.format(1000.0 *global_metrics['mean_t_RR']))


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
    Models = ['GeM_pointnet']
    sequences = ['00','02','05','06','08']
    sequences = ['08']
    dataset_type = 'new'

    topcand = np.array([1,5,10])
    radius = [25]
    for model_name in Models:
        print(model_name)
        for seq in sequences:
            print(seq)
            torch.manual_seed(0)
            np.random.seed(0)
            #seq = '02'
            trainloader,testloader,max_top_cand,datasetname = load_data(root,model_name,seq,train_size,dataset_type,batch_size=100)
            model  = alpha_QE(alpha=0.01,k=max_top_cand,radius=radius,n_samples = len(testloader))
            #loader = alpha.AlphaQEData()
            
            for r in radius:
                print(f"Radius: {r}")
                global_metrics = model.evaluation(testloader)
                # model.print_results(global_metrics)

                recall = np.array(global_metrics['recall'][r])
                print(recall[topcand-1])

                recall = np.array(global_metrics['recall_rr'][r])
                print(recall[topcand-1])
