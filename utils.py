
import numpy as np
import torch
import os

def save_results_csv2(self,file,results,top,**argv):
    import pandas as pd
    if file == None:
      raise NameError    #file = self.results_file # Internal File name 
    
    decimal_res = 3
    metrics = ['recall','recall_rr','MRR','MRR_rr','mean_t_RR'] # list(results.keys())[5:]
    recall= np.round(np.transpose(np.array(results['recall'][25])),decimal_res).reshape(-1,1)
    recall_rr= np.round(np.transpose(np.array(results['recall_rr'][25])),decimal_res).reshape(-1,1)

    scores = np.zeros((recall.shape[0],3))
    scores[0,]
    scores[0,0]= np.round(results['MRR'][25],decimal_res)
    scores[0,1]= np.round(results['MRR_rr'][25],decimal_res)
    scores[0,2]= results['mean_t_RR']
    scores = np.concatenate((recall,recall_rr,scores),axis=1)

    df = pd.DataFrame(scores,columns = metrics)
    best = np.round(results['recall_rr'][25][top-1],decimal_res)
    checkpoint_dir = ''
    filename = os.path.join(checkpoint_dir,f'{file}-{str(best)}.csv')
    df.to_csv(filename)




def eval_place(queries,descriptrs,poses,k=25,radius=[25],reranking = None):

  if not isinstance(queries,np.ndarray):
     queries = np.array(queries)
     
  n_frames = queries.shape[0]
  whole_map = np.arange(descriptrs.shape[0])
  
  global_metrics = {'tp': {r: [0] * k for r in radius}}
  global_metrics['tp_rr'] = {r: [0] * k for r in radius}
  global_metrics['RR'] = {r: [] for r in radius}
  global_metrics['RR_rr'] = {r: [] for r in radius}
  global_metrics['t_RR'] = []
  
  for i,(q) in enumerate(queries):
    
    query_pos = poses[q]
    query_destps = descriptrs[q]

    #q = queries[query_ndx]
    map_idx = np.arange(q-50) # generate indices 
    filtered_map_idx = whole_map[map_idx]
    map_positions = poses[filtered_map_idx]
    map_descriptrs = descriptrs[filtered_map_idx]
    
    delta = query_pos - map_positions
    euclid_dist = np.linalg.norm(delta, axis=-1)

    delta_dscpts = query_destps - map_descriptrs
    embed_dist = np.linalg.norm(delta_dscpts, axis=-1)
    nn_ndx = np.argsort(embed_dist)[:k]

    if isinstance(reranking,(np.ndarray, np.generic,list)):
      nn_ndx_rr = nn_ndx[reranking[i]]
      euclid_dist_rr = euclid_dist[nn_ndx_rr]
      global_metrics['tp_rr'] = {r: [global_metrics['tp_rr'][r][nn] + (1 if (euclid_dist_rr[:nn + 1] <= r).any() else 0) for nn in range(k)] for r in radius}
      global_metrics['RR_rr'] = {r: global_metrics['RR_rr'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist_rr <= r) if x), 0)] for r in radius}
      

    euclid_dist = euclid_dist[nn_ndx]
    # Count true positives for different radius and NN number
    global_metrics['t_RR'].append(nn_ndx)
    global_metrics['tp'] = {r: [global_metrics['tp'][r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(k)] for r in radius}
    global_metrics['RR'] = {r: global_metrics['RR'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist <= r) if x), 0)] for r in radius}


    # Calculate mean metrics
  global_metrics["recall"] = {r: [global_metrics['tp'][r][nn] / n_frames for nn in range(k)] for r in radius}
  global_metrics['MRR'] = {r: np.mean(np.asarray(global_metrics['RR'][r])) for r in radius}
  
  if isinstance(reranking,(np.ndarray, np.generic,list)):
    global_metrics["recall_rr"] = {r: [global_metrics['tp_rr'][r][nn] / n_frames for nn in range(k)] for r in radius}
    global_metrics['MRR_rr'] = {r: np.mean(np.asarray(global_metrics['RR_rr'][r])) for r in radius}
  
  global_metrics['mean_t_RR'] = np.mean(np.asarray(global_metrics['t_RR']))
  return global_metrics



def comp_pair_permutations(n_samples):
    combo_idx = torch.arange(n_samples)
    permutation = torch.from_numpy(np.array([np.array([a, b]) for idx, a in enumerate(combo_idx) for b in combo_idx[idx + 1:]]))
    return permutation[:,0],permutation[:,1]



def comp_loops(sim_map,queries,window=500,max_top_cand=25):
  loop_cand = []
  loop_sim = []
  #eu_value = np.linalg.norm(x - data,axis=1)
  for i,q in enumerate(queries):
    sim = sim_map[i] # get loop similarities for query i 
    bottom = q-window # 
    elegible = sim[:bottom] 
    #elegible = sim
    cand = np.argsort(elegible)[:max_top_cand] # sort similarities and get top N candidates
    sim = elegible[cand]
    loop_sim.append(sim)
    loop_cand.append(cand)
  return np.array(loop_cand), np.array(loop_sim)




def parse_triplet_file(file):
    import os
    assert os.path.isfile(file)
    f = open(file)
    anchors = []
    positives = []
    negatives = []
    for line in f:
        value_str = line.rstrip().split('_')
        anchors.append(int(value_str[0].split(':')[-1]))
        positives.append(int(value_str[1].split(':')[-1]))
        negatives.append([int(i) for i in value_str[2].split(':')[-1].split(' ')])
    f.close()

    anchors = np.array(anchors,dtype=np.uint32)
    positives = np.array(positives,dtype=np.uint32)
    negatives = np.array(negatives,dtype=np.uint32)

    return anchors,positives,negatives


def calculateMahalanobis(y=None, data=None, inv_covmat=None):
  
    y_mu = y - data
    #if not cov:
    #    cov = np.cov(data.values.T)
    #inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    #return np.sqrt(mahal)
    return  np.sqrt(mahal.diagonal())


def retrieval_knn(query_dptrs,map_dptrs, top_cand,metric):
    
    #retrieved_loops ,scores = euclidean_knnv2(query_dptrs,map_dptrs, top_cand= max_top)
    metric_fun = loss_lib.get_distance_function(metric)
    scores,winner = [],[]

    for q in query_dptrs:
        q_torch,map_torch = totensorformat(q.reshape(1,-1),map_dptrs) 
        sim = metric_fun(q_torch,map_torch,dim=2).squeeze() # similarity-based metrics 0 :-> same; +inf: -> Dissimilar 
        sort_value,sort_idx = sim.sort() # Sort to get the most similar vectors first
        # save top candidates
        scores.append(sort_value.detach().cpu().numpy()[:top_cand])
        winner.append(sort_idx.detach().cpu().numpy()[:top_cand])

    return np.array(winner),np.array(scores)


def retrieve_eval(retrieved_map,true_relevant_map,top=1,**argv):
  '''
  In a relaxed setting, at each query it is only required to retrieve one loop. 
  so: 
    Among the retrieved loop in true loop 
    recall  = tp/1
  '''
  assert top > 0
  n_queries = retrieved_map.shape[0]
  precision, recall = 0,0
  for retrieved,relevant in zip(retrieved_map,true_relevant_map):
    top_retrieved = retrieved[:top] # retrieved frames for a given query
    
    tp = 0 # Reset 
    if any(([True  if cand in relevant else False for cand in top_retrieved])):
        # It is only required to find one loop per anchor in a set of retrieved frames
        tp = 1 
    
    recall += tp # recall = tp/1
    precision += tp/top # retrieved loops/ # retrieved frames (top candidates) (precision w.r.t the query)
  
  recall /= n_queries  # average recall of all queries 
  precision /= n_queries  # average precision of all queries 

  return({'recall':recall,'precision':precision})