
import numpy as np

def comp_loops(sim_map,queries,window=500,max_top_cand=25):
  loop_cand = []
  loop_sim = []
  #eu_value = np.linalg.norm(x - data,axis=1)
  for i,q in enumerate(queries):
    sim = sim_map[i]
    bottom = q-window 
    elegible = sim[:bottom]
    #elegible = sim
    cand = np.argsort(elegible)[:max_top_cand]
    sim = elegible[cand]
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