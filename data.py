import pandas as pd
import numpy as np
import os
import re

from utils import _build_dict_from_lst, _build_inverse_dict, _build_dict_from_file, file_exists


def get_KG_data(kg_name, verbose=True):
    if kg_name == "WD15K":
        train_df, test_df, valid_df, id2relation_dict, id2entity_dict, all_relations, all_entities, all_relations_long_text  = get_WD15K_data(verbose)
    elif kg_name == "UMLs":
        train_df, test_df, valid_df, id2relation_dict, id2entity_dict, all_relations, all_entities, all_relations_long_text  = get_UMLs_data(verbose)
    elif kg_name == "FB15K":
        train_df, test_df, valid_df, id2relation_dict, id2entity_dict, all_relations, all_entities, all_relations_long_text  =  get_FB15K_data(verbose)
    elif kg_name == "WN18RR":
        train_df, test_df, valid_df, id2relation_dict, id2entity_dict, all_relations, all_entities, all_relations_long_text  = get_WN18RR_data(verbose)
    elif kg_name == "ConceptNet":
        train_df, test_df, valid_df, id2relation_dict, id2entity_dict, all_relations, all_entities, all_relations_long_text =  get_ConceptNet_data(verbose)
    else:
        raise Exception("Invalid kg_name={}".format(kg_name))
    density = len(pd.concat([train_df,test_df,valid_df])) / (len(all_entities) * (len(all_entities) -1))
    degree = len(pd.concat([train_df,test_df,valid_df])) / len(all_entities)
    print("KG of edge density {} and average degree {}".format(density, degree))
    return train_df, test_df, valid_df, id2relation_dict, id2entity_dict, all_relations, all_entities, all_relations_long_text 

def get_WD15K_data(verbose=True,build_dict_from_file=True):
    print("KG DATASET: WD15k")
    train_df = pd.read_csv("data/WD15K/train.txt",sep="\t",header=None)
    test_df = pd.read_csv("data/WD15K/test.txt",sep="\t",header=None)
    valid_df = pd.read_csv("data/WD15K/valid.txt",sep="\t",header=None)
    data_df = pd.concat([train_df,test_df,valid_df])
    train_df.columns = ["subj", "rel", "obj"]
    test_df.columns = ["subj", "rel", "obj"]
    valid_df.columns = ["subj", "rel", "obj"]
    data_df.columns = ["subj", "rel", "obj"]    
    if verbose:
        print("SAMPLE COUNT: {} triplets in the whole dataset".format(len(data_df)))
        print("\tTrain set:{}\n\tTest:{}\n\tValid:{}".format(len(train_df), len(test_df),len(valid_df)))
    n_rel = data_df.nunique()
    if verbose:
        print("VALUE COUNT: unique value count: {} subject, {} relations, {} objects".format(n_rel.iloc[0], n_rel.iloc[1], n_rel.iloc[2]))
    all_relations = data_df["rel"].unique()
    subj_entities = pd.unique(data_df["subj"])
    obj_entities =  pd.unique(data_df["obj"])
    all_entities = np.concatenate([subj_entities, obj_entities[~np.isin(obj_entities,subj_entities)]])
    shared_entities = np.intersect1d(subj_entities,obj_entities)
    if verbose: print("\t{} subject entities, {} object entities\n\t{} unique entities, {} entities appears as both subj and obj". format(len(subj_entities), len(obj_entities), len(all_entities),len(shared_entities)))
    id2relation_dict = _build_dict_from_lst(all_relations)
    id2entity_dict = _build_dict_from_lst(all_entities)
    assert len(list(set(id2relation_dict.values()))) == len(id2relation_dict.values())
    assert len(list(set(id2entity_dict.values()))) == len(id2entity_dict.values())
    if build_dict_from_file:
        id2relation_dict_1 = _build_dict_from_file('data/WD15K/relations.dict')
        id2entity_dict_1 = _build_dict_from_file('data/WD15K/entities.dict')
        assert  id2relation_dict_1 == id2relation_dict
        assert id2entity_dict_1 == id2entity_dict
    all_relations_long_text = all_relations
    return train_df, test_df, valid_df, id2relation_dict, id2entity_dict, all_relations, all_entities, all_relations_long_text 

def get_UMLs_data(verbose=True):
    print("KG DATASET: UMLs")
    train_df = pd.read_csv("data/umls/train.txt",sep="\t",header=None)
    test_df = pd.read_csv("data/umls/test.txt",sep="\t",header=None)
    valid_df = pd.read_csv("data/umls/valid.txt",sep="\t",header=None)
    data_df = pd.concat([train_df,test_df,valid_df])
    train_df.columns = ["subj", "rel", "obj"]
    test_df.columns = ["subj", "rel", "obj"]
    valid_df.columns = ["subj", "rel", "obj"]
    data_df.columns = ["subj", "rel", "obj"]    
    if verbose:
        print("SAMPLE COUNT: {} triplets in the whole dataset".format(len(data_df)))
        print("\tTrain set:{}\n\tTest:{}\n\tValid:{}".format(len(train_df), len(test_df),len(valid_df)))
    n_rel = data_df.nunique()
    if verbose:
        print("VALUE COUNT: unique value count: {} subject, {} relations, {} objects".format(n_rel.iloc[0], n_rel.iloc[1], n_rel.iloc[2]))
    all_relations = data_df["rel"].unique()
    subj_entities = pd.unique(data_df["subj"])
    obj_entities =  pd.unique(data_df["obj"])
    all_entities = np.concatenate([subj_entities, obj_entities[~np.isin(obj_entities,subj_entities)]])
    shared_entities = np.intersect1d(subj_entities,obj_entities)
    if verbose: print("\t{} subject entities, {} object entities\n\t{} unique entities, {} entities appears as both subj and obj". format(len(subj_entities), len(obj_entities), len(all_entities),len(shared_entities)))
    id2relation_dict = _build_dict_from_file('data/umls/relations.dict')
    id2entity_dict = _build_dict_from_file('data/umls/entities.dict')
    assert len(list(set(id2relation_dict.values()))) == len(id2relation_dict.values())
    assert len(list(set(id2entity_dict.values()))) == len(id2entity_dict.values())
    all_relations_long_text = all_relations
    return train_df, test_df, valid_df, id2relation_dict, id2entity_dict, all_relations, all_entities, all_relations_long_text 


def get_FB15K_data(verbose=True):
    print("KG DATASET: FB15K")
    train_df = pd.read_csv("data/FB15K-237/train.txt",sep="\t",header=None)
    test_df = pd.read_csv("data/FB15K-237/test.txt",sep="\t",header=None)
    valid_df = pd.read_csv("data/FB15K-237/valid.txt",sep="\t",header=None)
    train_df.columns = ["subj", "rel", "obj"]
    test_df.columns = ["subj", "rel", "obj"]
    valid_df.columns = ["subj", "rel", "obj"]
    train_df = train_df.apply(_transform_row, axis=1)
    train_df = train_df.drop_duplicates()
    valid_df = valid_df.apply(_transform_row, axis=1)
    valid_df = valid_df.drop_duplicates()
    test_df = test_df.apply(_transform_row, axis=1)
    test_df = test_df.drop_duplicates()
    relation_df = pd.read_csv("data/FB15K-237/fb15k_rels.csv",index_col=False)
    relation_df = relation_df.set_index('dict-idx', inplace=False)
    id2relation_dict = relation_df['long-text'].to_dict()
    id2rel_dict = relation_df['short-text'].to_dict()
    train_df = update_dataframe(train_df, id2relation_dict, id2rel_dict)
    test_df = update_dataframe(test_df, id2relation_dict, id2rel_dict)
    valid_df = update_dataframe(valid_df, id2relation_dict, id2rel_dict)
    id2entity_dict = _build_dict_from_file('data/FB15K-237/entities.dict')
    assert len(list(set(id2relation_dict.values()))) == len(id2relation_dict.values())
    assert len(list(set(id2entity_dict.values()))) == len(id2entity_dict.values())
    data_df = pd.concat([train_df,test_df,valid_df])
    data_df = data_df.drop_duplicates()
    data_df.columns = ["subj", "rel", "obj"]    
    if verbose:
        print("SAMPLE COUNT: {} triplets in the whole dataset".format(len(data_df)))
        print("\tTrain set:{}\n\tTest:{}\n\tValid:{}".format(len(train_df), len(test_df),len(valid_df)))
    n_rel = data_df.nunique()
    if verbose:
        print("VALUE COUNT: unique value count: {} subject, {} relations, {} objects".format(n_rel.iloc[0], n_rel.iloc[1], n_rel.iloc[2]))
    all_relations = data_df["rel"].unique()
    subj_entities = pd.unique(data_df["subj"])
    obj_entities =  pd.unique(data_df["obj"])
    all_entities = np.concatenate([subj_entities, obj_entities[~np.isin(obj_entities,subj_entities)]])
    shared_entities = np.intersect1d(subj_entities,obj_entities)
    if verbose: print("\t{} subject entities, {} object entities\n\t{} unique entities, {} entities appears as both subj and obj". format(len(subj_entities), len(obj_entities), len(all_entities),len(shared_entities)))
    all_relations_long_text = translate_to_long_text(id2rel_dict, id2relation_dict, all_relations)
    return train_df, test_df, valid_df, id2rel_dict, id2entity_dict, all_relations, all_entities, all_relations_long_text


def get_WN18RR_data(verbose=True):
    print("KG DATASET: WN18RR")
    train_df = pd.read_csv("data/wn18rr/train.txt",sep="\t",header=None)
    test_df = pd.read_csv("data/wn18rr/test.txt",sep="\t",header=None)
    valid_df = pd.read_csv("data/wn18rr/valid.txt",sep="\t",header=None)
    train_df.columns = ["subj", "rel", "obj"]
    test_df.columns = ["subj", "rel", "obj"]
    valid_df.columns = ["subj", "rel", "obj"]
    train_df['subj'] = train_df['subj'].apply(str)
    train_df['obj'] = train_df['obj'].apply(str)
    test_df['subj'] = test_df['subj'].apply(str)
    test_df['obj'] = test_df['obj'].apply(str)
    valid_df['subj'] = valid_df['subj'].apply(str)
    valid_df['obj'] = valid_df['obj'].apply(str)
    train_df['subj'] = train_df['subj'].str.zfill(8)
    train_df['obj'] = train_df['obj'].str.zfill(8)
    test_df['subj'] = test_df['subj'].str.zfill(8)
    test_df['obj'] = test_df['obj'].str.zfill(8)
    valid_df['subj'] = valid_df['subj'].str.zfill(8)
    valid_df['obj'] = valid_df['obj'].str.zfill(8)
    train_df = train_df.apply(_transform_row, axis=1)
    train_df = train_df.drop_duplicates()
    valid_df = valid_df.apply(_transform_row, axis=1)
    valid_df = valid_df.drop_duplicates()
    test_df = test_df.apply(_transform_row, axis=1)
    test_df = test_df.drop_duplicates()
    train_df['rel'] = train_df['rel'].str.lstrip('_')
    test_df['rel'] = test_df['rel'].str.lstrip('_')
    valid_df['rel'] = valid_df['rel'].str.lstrip('_')
    data_df = pd.concat([train_df,test_df,valid_df])
    data_df = data_df.drop_duplicates()
    data_df.columns = ["subj", "rel", "obj"]    
    if verbose:
        print("SAMPLE COUNT: {} triplets in the whole dataset".format(len(data_df)))
        print("\tTrain set:{}\n\tTest:{}\n\tValid:{}".format(len(train_df), len(test_df),len(valid_df)))
    n_rel = data_df.nunique()
    if verbose:
        print("VALUE COUNT: unique value count: {} subject, {} relations, {} objects".format(n_rel.iloc[0], n_rel.iloc[1], n_rel.iloc[2]))
    all_relations = data_df["rel"].unique()
    subj_entities = pd.unique(data_df["subj"])
    obj_entities =  pd.unique(data_df["obj"])
    all_entities = np.concatenate([subj_entities, obj_entities[~np.isin(obj_entities,subj_entities)]])
    shared_entities = np.intersect1d(subj_entities,obj_entities)
    if verbose: print("\t{} subject entities, {} object entities\n\t{} unique entities, {} entities appears as both subj and obj". format(len(subj_entities), len(obj_entities), len(all_entities),len(shared_entities)))
    id2relation_dict = _build_dict_from_file('data/wn18rr/relations.dict')
    id2relation_dict = {k: v for k, v in id2relation_dict.items() if not v.startswith('!')} 
    id2relation_dict = {k: v.lstrip('_') if isinstance(v, str) else v for k, v in id2relation_dict.items()}
    id2entity_dict = _build_dict_from_file('data/wn18rr/entities.dict')
    assert len(list(set(id2relation_dict.values()))) == len(id2relation_dict.values())
    assert len(list(set(id2entity_dict.values()))) == len(id2entity_dict.values())
    all_relations_long_text = all_relations
    return train_df, test_df, valid_df, id2relation_dict, id2entity_dict, all_relations, all_entities, all_relations_long_text 

from utils import _build_dict_from_lst, _build_inverse_dict, _build_dict_from_file, file_exists

def get_ConceptNet_data(verbose=True):
    print("KG DATASET: ConceptNet-100K")
    train_df = pd.read_csv("data/ConceptNet/train100k.txt",sep="\t",header=None)
    train_df.columns = ["rel", "subj", "obj","score"]
    train_df = train_df[train_df["score"]>0]
    train_df = train_df[["subj","rel","obj"]]
    test_df = pd.read_csv("data/ConceptNet/test.txt",sep="\t",header=None)
    test_df.columns = ["rel", "subj", "obj","score"]
    test_df = test_df[test_df["score"]>0]
    test_df = test_df[["subj","rel","obj"]]
    dev1_df = pd.read_csv("data/ConceptNet/dev1.txt",sep="\t",header=None)
    dev1_df.columns = ["rel", "subj", "obj","score"]
    dev1_df = dev1_df[dev1_df["score"]>0]
    dev1_df = dev1_df[["subj","rel","obj"]]
    dev2_df = pd.read_csv("data/ConceptNet/dev2.txt",sep="\t",header=None)
    dev2_df.columns = ["rel", "subj", "obj","score"]
    dev2_df = dev2_df[dev2_df["score"]>0]
    dev2_df = dev2_df[["subj","rel","obj"]]
    valid_df = pd.concat([dev1_df, dev2_df])
    data_df = pd.concat([train_df,test_df,valid_df])
    if verbose:
        print("SAMPLE COUNT: {} triplets in the whole dataset".format(len(data_df)))
        print("\tTrain set:{}\n\tTest:{}\n\tValid:{}".format(len(train_df), len(test_df),len(valid_df)))
    n_rel = data_df.nunique()
    if verbose:
        print("VALUE COUNT: unique value count: {} subject, {} relations, {} objects".format(n_rel.iloc[0], n_rel.iloc[1], n_rel.iloc[2]))
    all_relations = data_df["rel"].unique()
    subj_entities = pd.unique(data_df["subj"])
    obj_entities =  pd.unique(data_df["obj"])
    all_entities = np.concatenate([subj_entities, obj_entities[~np.isin(obj_entities,subj_entities)]])
    shared_entities = np.intersect1d(subj_entities,obj_entities)
    if verbose: print("\t{} subject entities, {} object entities\n\t{} unique entities, {} entities appears as both subj and obj". format(len(subj_entities), len(obj_entities), len(all_entities),len(shared_entities)))
    id2relation_dict = _build_dict_from_lst(all_relations)
    id2entity_dict = _build_dict_from_lst(all_entities)
    assert len(list(set(id2relation_dict.values()))) == len(id2relation_dict.values())
    assert len(list(set(id2entity_dict.values()))) == len(id2entity_dict.values())
    all_relations_long_text = all_relations
    return train_df, test_df, valid_df, id2relation_dict, id2entity_dict, all_relations, all_entities, all_relations_long_text


def _map_strings_to_ints(array, dictionaries):
    mapped_array = np.empty(array.shape, dtype=int)
    for col, mapping_dict in enumerate(dictionaries):
        mapped_array[:, col] = np.vectorize(lambda x: mapping_dict.get(x, -1))(array[:, col])
    return mapped_array

def encode_kg_to_arr(graph_df, id_to_entity,id_to_relation):
    entity_to_id = _build_inverse_dict(id_to_entity)
    relation_to_id =  _build_inverse_dict(id_to_relation)
    graph_arr = graph_df.to_numpy()
    dictionaries = [entity_to_id, relation_to_id, entity_to_id]
    graph_arr = _map_strings_to_ints(graph_arr, dictionaries)
    return graph_arr

def _remove_wikidata_prefix(s):
    return "_".join(s.split("_")[1:])

def remove_wikidata_prefix(s):
    match = re.match(r'^[QP]\d+_(.*)', s)
    if match:
        return match.group(1)
    return s

def convert_kgtxt_to_natlang(kg_item_text):
    kg_item_text = str(kg_item_text)
    kg_item_natlang = kg_item_text.replace("_", " ")
    kg_item_natlang = kg_item_natlang.replace(".", " ")
    kg_item_natlang = kg_item_natlang.lstrip()
    kg_item_natlang = kg_item_natlang.rstrip()
    return kg_item_natlang

def convert_kgtxt_to_natlang_plus(kg_item_text):
    kg_item_text = remove_wikidata_prefix(kg_item_text)
    return convert_kgtxt_to_natlang(kg_item_text)

def _transform_row(row):
    if row.iloc[1].startswith('!'):
        row.iloc[0], row.iloc[2] = row.iloc[2], row.iloc[0]
        row.iloc[1] = row.iloc[1][1:]
    return row

def get_KGE(dataset):
    if dataset == "UMLs":
        kge_dir = "KGE/RotatE/models/RotatE_umls_0"
    elif dataset == "WN18RR":
        kge_dir = "KGE/RotatE/models/RotatE_wn18rr_0"
    elif dataset == "FB15K":
        kge_dir = "KGE/RotatE/models/RotatE_FB15k-237_0"
    else:
        raise Exception("invalid dataset option {}".format(dataset))
    kge_entities =  np.load(os.path.join(kge_dir, "entity_embedding.npy")) 
    kge_relations = np.load(os.path.join(kge_dir, "relation_embedding.npy"))
    kge_dimension = kge_relations.shape[1]
    assert kge_entities.shape[1] == 2 * kge_dimension
    return kge_entities, kge_relations, kge_dimension

def trim_relations(relations):
    phrase_count = {}
    trimmed_relations = {}
    max_depth = max(len(rel_text.split('/')) for rel_text in relations.values()) - 1
    def get_phrase(rel_text, depth):
        parts = rel_text.split('/')
        if len(parts) > depth:
            return '/'.join(parts[-depth:])
        return rel_text
    for rel_id, rel_text in relations.items():
        for depth in range(1, max_depth + 1):
            phrase = get_phrase(rel_text, depth)
            if phrase in phrase_count:
                phrase_count[phrase] += 1
            else:
                phrase_count[phrase] = 1
    for rel_id, rel_text in relations.items():
        trimmed = False
        for depth in range(1, max_depth + 1):
            phrase = get_phrase(rel_text, depth)
            if phrase_count[phrase] == 1:
                trimmed_relations[rel_id] = phrase
                trimmed = True
                break
        if not trimmed:
            trimmed_relations[rel_id] = rel_text
    return trimmed_relations


def shorten_relations(relations):
    after_dot_parts = {}
    shortened_relations = {}
    def get_after_dot_part(text):
        parts = text.split(".")
        return parts[-1] if len(parts) > 1 else None
    for rel_id, rel_text in relations.items():
        after_dot_part = get_after_dot_part(rel_text)
        if after_dot_part:
            if after_dot_part in after_dot_parts:
                after_dot_parts[after_dot_part].append(rel_id)
            else:
                after_dot_parts[after_dot_part] = [rel_id]
    for rel_id, rel_text in relations.items():
        after_dot_part = get_after_dot_part(rel_text)
        if after_dot_part and len(after_dot_parts[after_dot_part]) > 1:
            shortened_part = after_dot_part.split("/")[-1]
            shortened_relations[rel_id] = rel_text.split(".")[0] + "." + shortened_part
        else:
            shortened_relations[rel_id] = rel_text
    return shortened_relations



def replace_dots(relations):
    replaced_relations = {}
    for rel_id, rel_text in relations.items():
        replaced_relations[rel_id] = rel_text.replace(".", "/")
    return replaced_relations


def shorten_freebase_relations(id2rel):
    rdict_1 = trim_relations(id2rel)
    rdict_2 = shorten_relations(rdict_1)
    rdict_2
    rdict_3 = replace_dots(rdict_2)
    rdict_3
    rdict_4 = trim_relations(rdict_3)
    return rdict_4


def update_dataframe(df, long_text_relations, short_text_relations):
    for rel_id, rel_long_text in long_text_relations.items():
        if rel_id in short_text_relations:
            rel_short_text = short_text_relations[rel_id]
            df['rel'] = df['rel'].replace(rel_long_text, rel_short_text)
    return df


def translate_to_long_text(short_text_relations, long_text_relations, short_text_list):
    long_text_list = []
    for rel_short_text in short_text_list:
        for rel_id, short_text in short_text_relations.items():
            if short_text == rel_short_text:
                long_text_list.append(long_text_relations[rel_id])
                break
        else:
            long_text_list.append(rel_short_text)
    return long_text_list