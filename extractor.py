import os
import random
import pandas as pd
import time

RAND_SEED = 5

def _get_triplet_neighbors(triplet, data_df, n_neighbors=3, random_seed=RAND_SEED):
    assert n_neighbors > 0, "n_neighbors for triplet entity should be a postive integer, get {} instead".format(n_neighbors)
    subj, rel, obj = triplet
    subj_neighbors = data_df[(data_df["subj"]==subj) | (data_df["obj"]==subj)]
    subj_neighbors = subj_neighbors.drop(subj_neighbors[(subj_neighbors["subj"]==subj) & (subj_neighbors["rel"]==rel) & (subj_neighbors["obj"]==obj)].index) 
    if len(subj_neighbors) >= n_neighbors:
        subj_neighbors = subj_neighbors.sample(n_neighbors,random_state=random_seed)
    obj_neighbors = data_df[(data_df["subj"]==obj) | (data_df["obj"]==obj)]
    obj_neighbors = obj_neighbors.drop(obj_neighbors[(obj_neighbors["subj"]==subj) & (obj_neighbors["rel"]==rel) & (obj_neighbors["obj"]==obj)].index)
    if len(obj_neighbors) >= n_neighbors:
        obj_neighbors = obj_neighbors.sample(n_neighbors,random_state=random_seed)
    neighbors_df = pd.concat([subj_neighbors,obj_neighbors])
    neighbors_df = neighbors_df.drop_duplicates()
    return neighbors_df


def _remove_list_duplicate(l1, l2):
    dedup = []
    for item in l1:
        if item not in l2:
            dedup.append(item)
    return dedup


def get_nhop_path(rel, data_df, nhop, n_neighbors=3, n_subgraphs=30,random_seed=RAND_SEED,verbose=False):
    assert isinstance(nhop, int), "nhop should be an integer, get {} of type {} instead".format(nhop, type(nhop).__name__)
    paths = None
    if nhop >= 0:
        zerohop_paths = _get_0hop_path(rel, data_df)
        if len(zerohop_paths) >= n_subgraphs:
            zerohop_paths = random.sample(zerohop_paths, n_subgraphs)
        if verbose: print("[{}]: {} zerohop sampled".format(time.strftime("%Y-%m-%d %H:%M"), len(zerohop_paths)),flush=True)
        paths = zerohop_paths
    if nhop >= 1:
        onehop_paths = _get_1hop_path(zerohop_paths, data_df, n_neighbors, random_seed)
        if len(onehop_paths) >= n_subgraphs:
            onehop_paths = random.sample(onehop_paths, n_subgraphs)
        if verbose: print("[{}]: {} onehop sampled".format(time.strftime("%Y-%m-%d %H:%M"), len(onehop_paths)),flush=True)
        paths = onehop_paths
    if nhop >= 2:
        twohop_paths = _get_2hop_path(onehop_paths, data_df, n_neighbors, random_seed)
        if len(twohop_paths) >= n_subgraphs:
            twohop_paths = random.sample(twohop_paths, n_subgraphs)
        if verbose: print("[{}]: {} twohop sampled".format(time.strftime("%Y-%m-%d %H:%M"), len(twohop_paths)),flush=True)
        paths = twohop_paths
    if nhop >= 3:
        threehop_paths = _get_3hop_path(twohop_paths, data_df, n_neighbors, random_seed)
        if len(threehop_paths) >= n_subgraphs:
            threehop_paths = random.sample(threehop_paths, n_subgraphs)
        if verbose: print("[{}]: {} threehop sampled".format(time.strftime("%Y-%m-%d %H:%M"), len(threehop_paths)),flush=True)
        paths = threehop_paths
    if nhop >= 4:
        raise Exception("should not exceed 3-hop, get nhop={} instead".format(nhop))
    return paths

def _get_0hop_path(rel, data_df):
    paths = []
    res_df = data_df[data_df["rel"]==rel]
    for _, row in res_df.iterrows():
        zerohop_df = pd.DataFrame([row])
        path = dict()
        path["zerohop"]=zerohop_df
        paths.append(path)
    return paths

def _get_1hop_path(zerohop_paths, data_df, n_neighbors=3, random_seed=RAND_SEED):
    paths = []
    for zerohop_path in zerohop_paths:
        path = zerohop_path.copy()
        neighbor_triplets = []
        for _, row in zerohop_path["zerohop"].iterrows():
            row_neighbors = _get_triplet_neighbors(row, data_df, n_neighbors=n_neighbors, random_seed=random_seed)
            neighbor_triplets += row_neighbors.values.tolist()
        neighbor_triplets = _remove_list_duplicate(neighbor_triplets, zerohop_path["zerohop"].values.tolist())
        onehop_df = pd.DataFrame(neighbor_triplets, columns=["subj","rel","obj"])
        onehop_df = onehop_df.drop_duplicates()
        path["onehop"]=onehop_df
        paths.append(path)
    return paths    

def _get_2hop_path(onehop_paths, data_df, n_neighbors=3, random_seed=RAND_SEED):
    paths = []
    for onehop_path in onehop_paths:
        path = onehop_path.copy()
        neighbor_triplets = []
        for _, row in onehop_path["onehop"].iterrows():
            row_neighbors = _get_triplet_neighbors(row, data_df, n_neighbors=n_neighbors, random_seed=random_seed)
            neighbor_triplets += row_neighbors.values.tolist()
        neighbor_triplets = _remove_list_duplicate(neighbor_triplets, onehop_path["zerohop"].values.tolist()+onehop_path["onehop"].values.tolist())
        twohop_df = pd.DataFrame(neighbor_triplets, columns=["subj","rel","obj"])
        twohop_df = twohop_df.drop_duplicates()
        path["twohop"]=twohop_df
        paths.append(path)
    return paths 

def _get_3hop_path(twohop_paths, data_df, n_neighbors=3, random_seed=RAND_SEED):
    paths = []
    for twohop_path in twohop_paths:
        path = twohop_path.copy()
        neighbor_triplets = []
        for _, row in twohop_path["twohop"].iterrows():
            row_neighbors = _get_triplet_neighbors(row, data_df, n_neighbors=n_neighbors, random_seed=random_seed)
            neighbor_triplets += row_neighbors.values.tolist()
        neighbor_triplets = _remove_list_duplicate(neighbor_triplets, twohop_path["zerohop"].values.tolist()+twohop_path["onehop"].values.tolist()+twohop_path["twohop"].values.tolist())
        threehop_df = pd.DataFrame(neighbor_triplets, columns=["subj","rel","obj"])
        threehop_df = threehop_df.drop_duplicates()
        path["threehop"]=threehop_df
        paths.append(path)
    return paths 


def _get_connecting_triplet(path, level, data_df):
    assert level in ["zerohop","onehop","twohop","threehop"], "invalid level key {} for path dict".format(level)
    init_subj, _, init_obj = path["zerohop"].values.tolist()[0]
    connecting_triplets = []
    hopend_no_connection_triplets = []
    hopend_df = path[level]
    for _, row in hopend_df.iterrows():
        end_subj, end_rel, end_obj = row
        if init_subj == end_subj or init_subj == end_obj or init_obj == end_subj or init_obj == end_obj:
            continue
        connecting_df = data_df[  ((data_df["subj"]==init_subj) & (data_df["subj"]==end_subj) )
                                | ((data_df["subj"]==init_subj) & (data_df["subj"]==end_obj)) 
                                | ((data_df["subj"]==init_obj) & (data_df["subj"]==end_subj) )
                                | ((data_df["subj"]==init_obj) & (data_df["subj"]==end_obj) )
                                | ((data_df["subj"]==end_subj) & (data_df["obj"]==init_subj) )
                                | ((data_df["subj"]==end_subj) & (data_df["obj"]==init_obj)) 
                                | ((data_df["subj"]==end_obj) & (data_df["obj"]==init_subj)) 
                                | ((data_df["subj"]==end_obj) & (data_df["obj"]==init_obj)) ]
        connecting_triplets += connecting_df.values.tolist()
        if connecting_df.empty:
            hopend_no_connection_triplets.append([end_subj, end_rel, end_obj])
    return connecting_triplets, hopend_no_connection_triplets


def get_nhop_closed_path(rel, data_df, nhop, n_neighbors=3, n_subgraphs=30, random_seed=RAND_SEED,verbose=False, remove_outmost_open=True):
    assert isinstance(nhop, int), "nhop should be an integer, get {} of type {} instead".format(nhop, type(nhop).__name__)
    assert nhop >= 1, "a closed path should of len>=1, get {} instead".format(nhop)
    if verbose: print("\n[{}]: work on {}".format(time.strftime("%Y-%m-%d %H:%M"), rel))
    paths = get_nhop_path(rel, data_df, nhop-1, n_neighbors, n_subgraphs=n_subgraphs, random_seed=random_seed,verbose=verbose)
    closed_paths = []
    for path in paths:
        connecting_triplets = []
        if nhop >= 1:
            zerohop_connecting, zerohop_endopened = _get_connecting_triplet(path, "zerohop", data_df)
            connecting_triplets += zerohop_connecting
            connecting_triplets = _remove_list_duplicate(connecting_triplets, path["zerohop"].values.tolist())
            outmost_level="zerohop"
            hopend_open=zerohop_endopened
        if nhop >= 2:
            onehop_connecting, onehop_endopened = _get_connecting_triplet(path, "onehop", data_df)
            connecting_triplets += onehop_connecting
            connecting_triplets = _remove_list_duplicate(connecting_triplets, path["zerohop"].values.tolist()+path["onehop"].values.tolist())
            outmost_level="onehop"
            hopend_open=onehop_endopened
        if nhop >= 3:
            twohop_connecting, twohop_endopened = _get_connecting_triplet(path, "twohop", data_df)
            connecting_triplets += twohop_connecting
            connecting_triplets = _remove_list_duplicate(connecting_triplets, path["zerohop"].values.tolist()+path["onehop"].values.tolist()+path["twohop"].values.tolist())
            outmost_level="twohop"
            hopend_open=twohop_endopened
        if nhop >= 4:
            threehop_connecting, threehop_endopened = _get_connecting_triplet(path, "threehop", data_df)
            connecting_triplets += threehop_connecting
            connecting_triplets = _remove_list_duplicate(connecting_triplets, path["zerohop"].values.tolist()+path["onehop"].values.tolist()+path["twohop"].values.tolist()+path["threehop"].values.tolist())
            outmost_level="threehop"
            hopend_open=threehop_endopened
        if connecting_triplets:
            closed_path = dict()
            for level in path:
                closed_path[level]=path[level].values.tolist()
            if remove_outmost_open:
                hopend_triplets = path[outmost_level].values.tolist()
                hopend_triplets_connecting = _remove_list_duplicate(hopend_triplets, hopend_open)
                closed_path[outmost_level]=hopend_triplets_connecting
                if verbose: print("[{}]: removed {}/{} hop-end triplets that does not connect back to center knowledge".format(time.strftime("%Y-%m-%d %H:%M"), len(hopend_open), len(hopend_triplets)))
            connecting_df = pd.DataFrame(connecting_triplets,columns=["subj","rel","obj"])
            connecting_df = connecting_df.drop_duplicates()
            closed_path["connecting"]=connecting_df.values.tolist()
            closed_paths.append(closed_path)
    if verbose: print("[{}]: finished adding connecting triplets to form closed paths to {} samples".format(time.strftime("%Y-%m-%d %H:%M"), len(closed_paths)),flush=True)
    return closed_paths

def save_subgraphs(closed_paths, fname):
    subgraph_df = pd.DataFrame(closed_paths)
    subgraph_df.to_csv(fname)

def load_subgraphs(fname):
    loaded_df = pd.read_csv(fname,index_col=[0])
    loaded_subgraphs=loaded_df.to_dict('records')
    trimed_subgraphs = []
    for subgraph in loaded_subgraphs:
        g = dict()
        for key in subgraph:
            g[key] = eval(subgraph[key])
        trimed_subgraphs.append(g)
    return trimed_subgraphs