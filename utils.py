
import numpy as np

def comp_loops(sim_map,queries,window=500):
  loop_cand = []
  loop_sim = []
  for i,q in enumerate(queries):
    sim = sim_map[i]
    bottom = q-window 
    elegible = sim[:bottom]
    #elegible = sim
    cand = np.argsort(elegible)[:25]
    sim = elegible[cand]
    sim = elegible[cand]
    loop_sim.append(sim)
    loop_cand.append(cand)
  return np.array(loop_cand), np.array(loop_sim)




def calculateMahalanobis(y=None, data=None, inv_covmat=None):
  
    y_mu = y - data
    #if not cov:
    #    cov = np.cov(data.values.T)
    #inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    #return np.sqrt(mahal)
    return  np.sqrt(mahal.diagonal())



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