import numpy as np
import pandas as pd
import random
import math
import re
import torch
from collections import defaultdict
import torch
from transformers import T5Tokenizer, T5EncoderModel
from transformers import RobertaModel, RobertaTokenizer
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from nltk.stem import WordNetLemmatizer

from kge import kge_inference, get_KGE
from utils import file_exists, load_nested_list, save_nested_list,is_empty_dir
from data import remove_wikidata_prefix

import torch
import torch.nn as nn
import torch.optim as optim


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

RAND_SEED=5 
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)

USE_ENHANCED_KGE_SCORING = True
RELATION_ID2Text_MAPPING_MODE = "lemma"

def get_proposed_rules(fname):
    proposed_df = pd.read_csv(fname,index_col=[0])
    proposed_rules = proposed_df.values.tolist()
    proposed_rules = [l[0] for l in proposed_rules]
    return proposed_rules

def filter_rules(proposed_rules,relation,verbose=False):
    rule_sets = []
    for rule in proposed_rules:
        rule_set = []
        for r in rule.split("\n"):
            if r == "":
                continue
            pattern = r'^\d+\.\s'    
            r = re.sub(pattern, '', r)
            pattern = r'^\s+|\s+$'
            r = re.sub(pattern, '', r)
            pattern = r'^[\-\*]\s'   
            r = re.sub(pattern, '', r)
            pattern = r'^\s+|\s+$'
            r = re.sub(pattern, '', r)
            r = r.replace("{", "(")
            r = r.replace("[", "(")
            r = r.replace("}", ")")
            r = r.replace("]", ")")
            rule_set.append(r)
        rule_sets.append(rule_set)
    if verbose: print("totalling {} rules including duplicates".format(sum([len(rs) for rs in rule_sets])))
    pattern = r'^IF\s.+?\sTHEN\s.+?$'
    irregular_rules = []
    irregular_counts = 0
    filter1_rule_sets = []
    for rs in rule_sets:
        rule_set = []
        irregular = []
        for r in rs:
            if re.search(pattern, r):
                rule_set.append(r)
            else:
                irregular.append(r)
        filter1_rule_sets.append(rule_set)
        irregular_rules.append(irregular)
        irregular_counts += len(irregular)
    if verbose: 
        if irregular_counts>0:
            print("there exists {} proposed rules that violate IF ... THEN ... syntax:".format(irregular_counts))
        else:
            print("all proposed rule follow IF ... THEN ... syntax")
        print("after filter1, totalling {} rules including duplicates".format(sum([len(rs) for rs in filter1_rule_sets])))
    pattern = r'^IF\s.+?\sTHEN\s\(.+?,\s'+re.escape(relation)+',\s.+?\)$'
    irregular_rules = []
    irregular_counts = 0
    filter2_rule_sets = []
    for rs in filter1_rule_sets:
        rule_set = []
        irregular = []
        for r in rs:
            if re.search(pattern, r):
                rule_set.append(r)
            else:
                irregular.append(r)
        filter2_rule_sets.append(rule_set)
        irregular_rules.append(irregular)
        irregular_counts += len(irregular)
    if verbose:
        if irregular_counts>0:
            print("there exists {} proposed rules that violate IF ... THEN (..., REL, ...) syntax:".format(irregular_counts))
        else:
            print("all proposed rule follow IF ... THEN (..., REL, ...) syntax")
        print("after filter2, totalling {} rules including duplicate".format(sum([len(rs) for rs in filter2_rule_sets])))
    pattern = r'IF\s+(.*?)\s+(?:(?:AND|OR)\s+(.*?)\s+)*THEN\s+(.*?)$'
    irregular_rules = []
    irregular_counts = 0
    filter3_rule_sets = []
    for rs in filter2_rule_sets:
        rule_set = []
        irregular = []
        for r in rs:
            if re.search(pattern, r):
                rule_set.append(r)
            else:
                irregular.append(r)
        filter3_rule_sets.append(rule_set)
        irregular_rules.append(irregular)
        irregular_counts += len(irregular)
    if verbose:
        if irregular_counts>0:
            print("there exists {} proposed rules that violate IF (...) AND/OR (...) THEN (...) syntax :".format(irregular_counts))
        else:
            print("all proposed rule follow IF (...) AND/OR (...) THEN (...) syntax")
        print("after filter 3, totalling {} rules including duplicate".format(sum([len(rs) for rs in filter3_rule_sets])))
    pattern = r'\((.*?)\)'
    irregular_rules = []
    irregular_counts = 0
    filter4_rule_sets = []
    for rs in filter3_rule_sets:
        rule_set = []
        irregular = []
        for r in rs:
            substrings = re.findall(pattern, r)
            is_irreg = False
            for substring in substrings:
                triplet = substring.split(", ")
                if len(triplet)== 3:
                    if len(triplet[0]) != 1:
                        is_irreg = True
                    if len(triplet[2]) != 1:
                        is_irreg = True
                else:
                    is_irreg = True 
            if is_irreg:
                irregular.append(r)
            else:
                rule_set.append(r)    
        filter4_rule_sets.append(rule_set)
        irregular_rules.append(irregular)
        irregular_counts += len(irregular)
    if verbose:
        if irregular_counts>0:
            print("there exists {} proposed rules that violate syntax check for content within triplet parenthese :".format(irregular_counts))
        else:
            print("all proposed rule follow syntax check for content within triplet parenthese")
        print("after filter 4, totalling {} rules including duplicate".format(sum([len(rs) for rs in filter4_rule_sets])))
    pattern = r'IF\s+(.*?)\s+THEN\s+(.*)$'
    irregular_rules = []
    irregular_counts = 0
    filter5_rule_sets = []
    for rs in filter4_rule_sets:
        rule_set = []
        irregular = []
        for r in rs:
            match = re.match(pattern, r)
            if match:
                if_conditions = match.group(1)
                then_conclusion = match.group(2)
                if then_conclusion in if_conditions:
                    irregular.append(r)
                else:
                    rule_set.append(r)
            else:
                irregular.append(r)
        filter5_rule_sets.append(rule_set)
        irregular_rules.append(irregular)
        irregular_counts += len(irregular)
    if verbose:
        if irregular_counts>0:
            print("there exists {} proposed rules that THEN conclusion appears in IF conditions:".format(irregular_counts))
        else:
            print("all proposed rule with THEN conclusion exclude from IF conditions")
        print("after filter5, totalling {} rules including duplicate".format(sum([len(rs) for rs in filter5_rule_sets])))
    return filter5_rule_sets

def encode_rules(filtered_rules,verbose=False):
    rule2id_dict = dict()
    id2rule_dict = dict()
    rule_count = 0
    rule_sets_ids = []
    for rs in filtered_rules:
        rs_ids = []
        for r in rs:
            if r in rule2id_dict.keys():
                rs_ids.append(rule2id_dict[r])
            else:
                rule2id_dict[r]=rule_count
                id2rule_dict[rule_count] = r
                rule_count += 1
                rs_ids.append(rule2id_dict[r])
        rule_sets_ids.append(rs_ids)
    if verbose:
        print("number of subgraphs: {}".format(len(rule_sets_ids)))
        print("number of unique rules: {}".format(rule_count))
    assert len(id2rule_dict.keys()) == len(rule2id_dict.keys())
    assert rule_count == len(id2rule_dict.keys())
    return rule_sets_ids, id2rule_dict, rule2id_dict

def prepare_rule_no_or(rules):
    filtered_rules = []
    for rule in rules:
        if rule.count("OR") == 0:
            filtered_rules.append(rule)
    return filtered_rules

def prepare_rule_no_not(rules):
    filtered_rules = []
    for rule in rules:
        if rule.count("NOT") == 0:
            filtered_rules.append(rule)
    return filtered_rules

def prepare_rule_max_nhop(rules, nhop=3):
    filtered_rules = []
    for rule in rules:
        if rule.count("AND") <= nhop-1:
            filtered_rules.append(rule)
    return filtered_rules

def prepare_rule_matching_brackets(rules):
    filtered_rules = []
    for rule in rules:
        if _check_rule_matching_brackets(rule):
            filtered_rules.append(rule)
    return filtered_rules

def _check_rule_matching_brackets(statement):
    statement = statement.replace("IF", "").replace("THEN", "")
    num_and = statement.count("AND")
    left_parenthese, right_parenthese = 0,0
    for char in statement:
        if char == '(':
            left_parenthese += 1
        if char == ')':
            right_parenthese += 1
    if left_parenthese != right_parenthese:
        return False
    elif left_parenthese != num_and+2:
        return False
    else:
        assert right_parenthese == num_and+2
        return True


def prepare_rule_check_type(rules,verbose=False):
    filtered_rules = []
    removed_rules = []
    for rule in rules:
        if verbose: print("checking type of logical rule: {}".format(rule))
        checked_results = _prepare_rule_check_type_relax(rule,verbose=verbose)
        if checked_results == -1:
            removed_rules.append(rule)
        else:
            filtered_rules.append(checked_results)
    if verbose: print("{} out of {} low-quality rules removed".format(len(removed_rules), len(rules)))
    return filtered_rules, removed_rules

def _prepare_rule_check_type_strict(rule,verbose=False):
    pattern = r'IF\s+(.*?)\s+THEN\s+(.*)$'
    match = re.match(pattern, rule)
    if match:
        if_conditions = match.group(1)
        then_conclusion = match.group(2)
    else:
        if verbose: Exception("rule {} does not follow IF (...) THEN (...) syntax")
        return -1
    num_and = if_conditions.count("AND")
    pattern=r'\((.*?)\)'
    if_triplets = re.findall(pattern, if_conditions)
    if_triplets = [s.split(', ') for s in if_triplets]
    then_triplet = re.findall(pattern, then_conclusion)
    if len(then_triplet)!=1:
        if verbose: print("multiple triplets inside THEN conlusion!")
        return -1
    then_triplet = then_triplet[0].split(', ')
    if if_triplets[0][0] != then_triplet[0] and if_triplets[0][2] != then_triplet[0]:
        if verbose: print("skipping low-quality rule {} that does not start with core concepts".format(rule))
        return -1
    if if_triplets[-1][0] != then_triplet[2] and if_triplets[-1][2] != then_triplet[2]:
        if verbose: print("skipping low-quality rule {} that does not end with core concept".format(rule))
        return -1
    rule_type = -1
    if num_and == 0:
        if if_triplets[0][0] == then_triplet[0] and if_triplets[0][2] == then_triplet[2]:
            rule_type = "01"
        elif if_triplets[0][0] == then_triplet[2] and if_triplets[0][2] == then_triplet[0]:
            rule_type = "02"
    elif num_and ==1:
        concept_A = then_triplet[0]
        concept_C = then_triplet[2]
        rtemplate = []
        if if_triplets[0][0] == concept_A:
            rtemplate.append("->")
            concept_B = if_triplets[0][2]
        elif if_triplets[0][2] == concept_A:
            rtemplate.append("<-")
            concept_B = if_triplets[0][0]
        if if_triplets[1][0] == concept_C:
            rtemplate.append("<-")
            concept_B_ = if_triplets[1][2]
        elif if_triplets[1][2] == concept_C:
            rtemplate.append("->")
            concept_B_ = if_triplets[1][0]
        if concept_B == concept_B_:
            if rtemplate == ["->","->"]:
                rule_type="11"
            elif rtemplate == ["<-","->"]:
                rule_type="12"
            elif rtemplate == ["->","<-"]:
                rule_type="13"
            elif rtemplate == ["<-","<-"]:
                rule_type="14"
    elif num_and ==2:
        concept_A = then_triplet[0]
        concept_B = None
        concept_C = None
        concept_D_ = None
        concept_D = then_triplet[2]
        rtemplate = []
        if if_triplets[0][0] == concept_A:
            rtemplate.append("->")
            concept_B = if_triplets[0][2]
        elif if_triplets[0][2] == concept_A:
            rtemplate.append("<-")
            concept_B = if_triplets[0][0]
        if if_triplets[1][0] == concept_B:
            rtemplate.append("->")
            concept_C = if_triplets[1][2]
        elif if_triplets[1][2] == concept_B:
            rtemplate.append("<-")
            concept_C = if_triplets[1][0]
        if if_triplets[2][0] == concept_C:
            rtemplate.append("->")
            concept_D_ = if_triplets[2][2]
        elif if_triplets[2][2] == concept_C:
            rtemplate.append("<-")
            concept_D_ = if_triplets[2][0]
        if concept_D_ and concept_D == concept_D_:
            if rtemplate == ["->","->","->"]:
                rule_type="21"
            elif rtemplate == ["->","->","<-"]:
                rule_type="22"
            elif rtemplate == ["->","<-","->"]:
                rule_type="23"
            elif rtemplate == ["<-","->","->"]:
                rule_type="24"
            elif rtemplate == ["->","<-","<-"]:
                rule_type="25"
            elif rtemplate == ["<-","->","<-"]:
                rule_type="26"
            elif rtemplate == ["<-","<-","->"]:
                rule_type="27"
            elif rtemplate == ["<-","<-","<-"]:
                rule_type="28"
    else:
        raise Exception("invalid triplets inside IF condition")
    return rule_type

def _prepare_rule_check_type_relax(rule,verbose=False):
    pattern = r'IF\s+(.*?)\s+THEN\s+(.*)$'
    match = re.match(pattern, rule)
    if match:
        if_conditions = match.group(1)
        then_conclusion = match.group(2)
    else:
        if verbose: Exception("rule {} does not follow IF (...) THEN (...) syntax")
        return -1
    num_and = if_conditions.count("AND")
    pattern=r'\((.*?)\)'
    if_triplets = re.findall(pattern, if_conditions)
    if_triplets = [s.split(', ') for s in if_triplets]
    then_triplet = re.findall(pattern, then_conclusion)
    if len(then_triplet)!=1:
        if verbose: print("multiple triplets inside THEN conlusion!")
        return -1
    then_triplet = then_triplet[0].split(', ')
    start_concepts = [if_triplets[0][0], if_triplets[0][2]]
    end_concepts = [if_triplets[-1][0], if_triplets[-1][2]]
    if then_triplet[0] not in start_concepts and then_triplet[2] not in start_concepts:
        if verbose: print("skipping low-quality rule {} that does not start with core concepts".format(rule))
        return -1
    if then_triplet[0] not in end_concepts and then_triplet[2] not in end_concepts:
        if verbose: print("skipping low-quality rule {} that does not end with core concept".format(rule))
        return -1
    if then_triplet[0] in end_concepts and then_triplet[2] in start_concepts:
        if_triplets = if_triplets[::-1]
    rule_type = -1
    if num_and == 0:
        if if_triplets[0][0] == then_triplet[0] and if_triplets[0][2] == then_triplet[2]:
            rule_type = "01"
        elif if_triplets[0][0] == then_triplet[2] and if_triplets[0][2] == then_triplet[0]:
            rule_type = "02"
    elif num_and ==1:
        concept_A = then_triplet[0]
        concept_C = then_triplet[2]
        rtemplate = []
        if if_triplets[0][0] == concept_A:
            rtemplate.append("->")
            concept_B = if_triplets[0][2]
        elif if_triplets[0][2] == concept_A:
            rtemplate.append("<-")
            concept_B = if_triplets[0][0]
        if if_triplets[1][0] == concept_C:
            rtemplate.append("<-")
            concept_B_ = if_triplets[1][2]
        elif if_triplets[1][2] == concept_C:
            rtemplate.append("->")
            concept_B_ = if_triplets[1][0]
        if len(rtemplate)!= 2:
            if verbose: print("invalid concept linking inside IF condition")
            return -1
        if concept_B == concept_B_:
            if rtemplate == ["->","->"]:
                rule_type="11"
            elif rtemplate == ["<-","->"]:
                rule_type="12"
            elif rtemplate == ["->","<-"]:
                rule_type="13"
            elif rtemplate == ["<-","<-"]:
                rule_type="14"
    elif num_and ==2:
        concept_A = then_triplet[0]
        concept_B = None
        concept_C = None
        concept_D_ = None
        concept_D = then_triplet[2]
        rtemplate = []
        if if_triplets[0][0] == concept_A:
            rtemplate.append("->")
            concept_B = if_triplets[0][2]
        elif if_triplets[0][2] == concept_A:
            rtemplate.append("<-")
            concept_B = if_triplets[0][0]
        if if_triplets[1][0] == concept_B:
            rtemplate.append("->")
            concept_C = if_triplets[1][2]
        elif if_triplets[1][2] == concept_B:
            rtemplate.append("<-")
            concept_C = if_triplets[1][0]
        if if_triplets[2][0] == concept_C:
            rtemplate.append("->")
            concept_D_ = if_triplets[2][2]
        elif if_triplets[2][2] == concept_C:
            rtemplate.append("<-")
            concept_D_ = if_triplets[2][0]
        if len(rtemplate)!= 3:
            if verbose: print("invalid concept linking inside IF condition")
            return -1
        if concept_D_ and concept_D == concept_D_:
            if rtemplate == ["->","->","->"]:
                rule_type="21"
            elif rtemplate == ["->","->","<-"]:
                rule_type="22"
            elif rtemplate == ["->","<-","->"]:
                rule_type="23"
            elif rtemplate == ["<-","->","->"]:
                rule_type="24"
            elif rtemplate == ["->","<-","<-"]:
                rule_type="25"
            elif rtemplate == ["<-","->","<-"]:
                rule_type="26"
            elif rtemplate == ["<-","<-","->"]:
                rule_type="27"
            elif rtemplate == ["<-","<-","<-"]:
                rule_type="28"
    else:
        if verbose: print("invalid triplets inside IF condition")
        return -1
    if rule_type == -1:
        if verbose: print("low quality rule removed")
        return -1
    return [rule, rule_type, if_triplets, then_triplet]


def prepare_rules(proposed_fname, relation_text, rules_fname, verbose=False):
    if file_exists(rules_fname):
        print("already processed raw rules, read from disk")
        checked_rules = load_nested_list(rules_fname)
    else:
        if verbose: print("working on relation {} from file {}".format(relation_text, proposed_fname))
        proposed_rules = get_proposed_rules(proposed_fname)
        filtered_rules = filter_rules(proposed_rules, relation_text,verbose=verbose)
        rule_sets_ids, id2rule_dict, rule2id_dict = encode_rules(filtered_rules,verbose=verbose)
        rules = list(id2rule_dict.values())
        rules = prepare_rule_no_or(rules)
        rules = prepare_rule_no_not(rules)
        rules = prepare_rule_max_nhop(rules, 3)
        rules = prepare_rule_matching_brackets(rules)
        checked_rules = prepare_rule_check_type(rules,verbose=False)[0]    
        save_nested_list(checked_rules, rules_fname)
        assert load_nested_list(rules_fname) == checked_rules
    if verbose: print("{} proposed rules for relation {}".format(len(checked_rules), relation_text))
    return checked_rules

def get_phrase_embedding(phrase, model, tokenizer, device, method="cls"):
    if tokenizer == None: 
        embedding = model.encode(phrase)
    else:
        inputs = tokenizer(phrase, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        if method == "cls":
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        elif method == 'mean':
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        elif method == 'max':
            embedding, _ = outputs.last_hidden_state.max(dim=1)
        embedding = embedding.cpu().numpy()
    return embedding

def custom_similarity(embeddings, threshold=0.5, do_clipping=True):
    n = len(embeddings)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0, 0]
            if do_clipping: sim = np.clip(sim, -1, 1)
            if threshold is not None:
                if sim.item() < threshold: sim=0
            sim_matrix[i, j] = sim
    return sim_matrix

def find_similar_phrases(phrases, similarity_matrix,include_self=False,verbose=False):
    counter = 0
    for i in range(len(phrases)):
        j_start = i+1
        if include_self: j_start -= 1
        for j in range(j_start, len(phrases)):
            similarity_score = similarity_matrix[i, j]
            if similarity_score != 0:
                counter += 1
                if verbose: print("({}, {}) of semantic similarity {}".format(phrases[i], phrases[j],similarity_score))
    return counter


def get_model_and_tokenizer(model_name, device):
    if model_name in ["t5-base", "t5-large"]:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5EncoderModel.from_pretrained(model_name)
        model.to(device)
    elif model_name in ["roberta-base", "roberta-large"]:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaModel.from_pretrained(model_name)
        model.to(device)
    elif model_name in ["bert-base-uncased", "bert-large-uncased"]:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        model.to(device)
    elif model_name.startswith("sentence-transformers/"):
        model = SentenceTransformer(model_name)
        model.to(device)
        tokenizer = None
    else:
        raise Exception("invalid model_name={}".format(model_name))

    return model, tokenizer

def get_relation_id_by_text(rel_kg_text, relation2id_dict, mode="strict"):
    if mode == "strict":
        rel_id = relation2id_dict.get(rel_kg_text, -1)
        return rel_id
    elif mode == "lemma":
        lemmatizer = WordNetLemmatizer()
        phrase_dict = relation2id_dict
        input_words = rel_kg_text.lower().split()
        input_lemmas = [lemmatizer.lemmatize(word) for word in input_words]
        for phrase, idx in phrase_dict.items():
            phrase_words = phrase.lower().split()
            phrase_lemmas = [lemmatizer.lemmatize(word) for word in phrase_words]
            if set(input_lemmas) == set(phrase_lemmas):
                return idx
        return -1
    elif mode == "sbert":
        sbert, _ = get_model_and_tokenizer("sentence-transformers/paraphrase-MiniLM-L6-v2", "cuda")
        import logging
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        input_embedding = sbert.encode(rel_kg_text.replace("_", " "), convert_to_tensor=True)
        highest_similarity = 0
        best_match_id = -1
        for phrase, idx in relation2id_dict.items():
            phrase_embedding = sbert.encode(phrase.replace("_", " "), convert_to_tensor=True)
            similarity = torch.nn.functional.cosine_similarity(input_embedding, phrase_embedding, dim=0)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_id = idx
        if highest_similarity >= 0.80:
            return best_match_id
        else:
            return -1
    else:
        raise Exception("invalid mode {} for relation lookup".format(mode))
    

def prepare_rule_map_relations(rule_cond, rule_head, relation2id_dict, verbose=True):
    triplets = rule_cond + [rule_head]
    assert RELATION_ID2Text_MAPPING_MODE in ["strict", "lemma", "sbert"]
    relations = []
    for tpl in triplets:
        rel_text = tpl[1]
        rel_kg_text = rel_text
        rel_kg_text = rel_text.replace("_", " ")
        rel_id = get_relation_id_by_text(rel_kg_text, relation2id_dict,mode=RELATION_ID2Text_MAPPING_MODE)
        if rel_id >= 0:
            relations.append(rel_id)
        else:
            rel_kg_text = rel_text.replace(" ", "_") 
            rel_id = get_relation_id_by_text(rel_kg_text, relation2id_dict,mode=RELATION_ID2Text_MAPPING_MODE)
            if rel_id >= 0:
                relations.append(rel_id)
            else:
                if verbose: print("!! rel_text={} no match in rel-id mapping".format(rel_kg_text))
                return None
    return relations


def convert_arr_to_sparse_coo(arr, n_ent, n_rel):
    n_data= arr.shape[0]
    coords = arr.swapaxes(0,1)
    vals = torch.ones(n_data)
    s = torch.sparse_coo_tensor(coords, vals, (n_ent, n_rel, n_ent))
    return s

def torch_sparse_coo_2d_slicing(sparse_tensor, row_start, row_end, col_start, col_end):
       mask = (sparse_tensor._indices()[0] >= row_start) & (sparse_tensor._indices()[0] < row_end) & \
              (sparse_tensor._indices()[1] >= col_start) & (sparse_tensor._indices()[1] < col_end)
       new_indices = sparse_tensor._indices()[:, mask]
       new_values = sparse_tensor._values()[mask]
       new_indices[0] -= row_start
       new_indices[1] -= col_start
       new_size = torch.Size([row_end - row_start, col_end - col_start])
       sliced_sparse_tensor = torch.sparse_coo_tensor(new_indices, new_values, new_size)
       return sliced_sparse_tensor

def torch_sparse_coo_3d_slicing_by_rel(sparse_tensor,slicing_idx):
    mask = (sparse_tensor._indices()[1] == slicing_idx)
    new_indices = sparse_tensor._indices()[:, mask]
    new_indices = new_indices[[0,2]] 
    new_values = sparse_tensor._values()[mask]
    sliced_sparse_tensor = torch.sparse_coo_tensor(new_indices, new_values, (sparse_tensor.shape[0], sparse_tensor.shape[0]))
    return sliced_sparse_tensor

def sparse_elementwise_mul_2d(sparse_a, sparse_b):
    assert sparse_a.is_sparse and sparse_a.layout == torch.sparse_coo
    assert sparse_b.is_sparse and sparse_b.layout == torch.sparse_coo
    sparse_and = sparse_a * sparse_b
    non_zero_indices = sparse_and._indices()[:, sparse_and._values() != 0]
    non_zero_values = sparse_and._values()[sparse_and._values() != 0]
    sparse_and = torch.sparse_coo_tensor(non_zero_indices, non_zero_values, sparse_and.shape)
    return sparse_and

def torch_sparse_coo_get_nonzero_list(sparse_M):
    return sparse_M._indices().t().tolist()

def lookup_bridging_XYZ(L1, L2, L3):
    L2_dict = defaultdict(list)
    L3_dict = defaultdict(list)
    for x, y in L2:
        L2_dict[x].append(y)
    for y, z in L3:
        L3_dict[y].append(z)
    result = []
    for x, z in L1:
        potential_ys = []
        if x in L2_dict:
            for y in L2_dict[x]:
                if z in L3_dict[y]:
                    potential_ys.append(y)
                    result.append([x, y, z])
    return result

def lookup_bridging_ABCD(L_AD, L_AB, L_BC, L_CD):
    L_AB_dict = {}
    for a, b in L_AB:
        if a not in L_AB_dict:
            L_AB_dict[a] = set()
        L_AB_dict[a].add(b)
    L_BC_dict = {}
    for b, c in L_BC:
        if b not in L_BC_dict:
            L_BC_dict[b] = set()
        L_BC_dict[b].add(c)
    L_CD_dict = {}
    for c, d in L_CD:
        if c not in L_CD_dict:
            L_CD_dict[c] = set()
        L_CD_dict[c].add(d)
    ABCD_result = []
    for a, d in L_AD:
        potential_bs_cs = []
        if a in L_AB_dict:
            for b in L_AB_dict[a]:
                if b in L_BC_dict:
                    for c in L_BC_dict[b]:
                        if c in L_CD_dict and d in L_CD_dict[c]:
                            potential_bs_cs.append((b, c))
        for b, c in potential_bs_cs:
            ABCD_result.append([a, b, c, d])
    return ABCD_result


def get_mask_by_rel_semantic(sparse_M, rel, semantic_similarities):
    n_ent = sparse_M.shape[0]
    n_rel = sparse_M.shape[1]
    assert sparse_M.shape[2]==n_ent and semantic_similarities.shape == (n_rel, n_rel)
    rel_similars = semantic_similarities[rel, :]
    rel_mask = torch.sparse_coo_tensor([[],[]],[], (n_ent, n_ent))
    for rid in range(n_rel):
        if rel_similars[rid] != 0:
            rel_mask += rel_similars[rid] * torch_sparse_coo_3d_slicing_by_rel(sparse_M, rid)
    return rel_mask



def tensor_logic_0and_case1_semantic(sparse_M, r1, r2, semantic_similarities, verbose=False):
    mask_Xr1Y = get_mask_by_rel_semantic(sparse_M, r1,semantic_similarities)
    mask_Xr2Y = get_mask_by_rel_semantic(sparse_M, r2,semantic_similarities)
    mask_align = sparse_elementwise_mul_2d(mask_Xr1Y, mask_Xr2Y) 
    ground_results = []
    for X, Y in torch_sparse_coo_get_nonzero_list(mask_align):
        if verbose: print(f"X:{X}, Y:{Y}")
        ground_results.append([X, Y])
    ground_rconds = []
    for X, Y in torch_sparse_coo_get_nonzero_list(mask_Xr1Y):
        if verbose: print(f"X:{X}, Y:{Y}")
        ground_rconds.append([X, Y])
    return mask_align, mask_Xr1Y, ground_results, ground_rconds


def tensor_logic_0and_case2_semantic(sparse_M, r1, r2, semantic_similarities, verbose=False):
    mask_Yr1X = get_mask_by_rel_semantic(sparse_M, r1,semantic_similarities) 
    mask_Xr2Y = get_mask_by_rel_semantic(sparse_M, r2,semantic_similarities)
    mask_align = sparse_elementwise_mul_2d(mask_Yr1X.transpose(0,1), mask_Xr2Y) 
    ground_results = []
    for X, Y in torch_sparse_coo_get_nonzero_list(mask_align):
        if verbose: print(f"X:{X}, Y:{Y}")
        ground_results.append([X, Y])
    ground_rconds = []
    for X, Y in torch_sparse_coo_get_nonzero_list(mask_Yr1X.transpose(0,1)):
        if verbose: print(f"X:{X}, Y:{Y}")
        ground_rconds.append([X, Y])
    return mask_align, mask_Yr1X.transpose(0,1), ground_results,ground_rconds


def tensor_logic_1and_case1_semantic(sparse_M, r1, r2, r3, semantic_similarities, verbose=False):
    mask_Xr1Y = get_mask_by_rel_semantic(sparse_M, r1,semantic_similarities)
    mask_Yr2Z = get_mask_by_rel_semantic(sparse_M, r2,semantic_similarities)
    mask_Xr3Z = get_mask_by_rel_semantic(sparse_M, r3,semantic_similarities)
    mask_XcmpZ = mask_Xr1Y @ mask_Yr2Z  
    mask_align = sparse_elementwise_mul_2d(mask_XcmpZ, mask_Xr3Z) 
    if verbose: print("concepts from rule body and rule head aligned! ")
    L_XZ=torch_sparse_coo_get_nonzero_list(mask_align)
    L_XY=torch_sparse_coo_get_nonzero_list(mask_Xr1Y)
    L_YZ=torch_sparse_coo_get_nonzero_list(mask_Yr2Z)
    ground_results = lookup_bridging_XYZ(L_XZ, L_XY, L_YZ)
    if verbose: print("bridging concepts grounded! ")
    L_XZ=torch_sparse_coo_get_nonzero_list(mask_XcmpZ)
    L_XY=torch_sparse_coo_get_nonzero_list(mask_Xr1Y)
    L_YZ=torch_sparse_coo_get_nonzero_list(mask_Yr2Z)
    ground_rconds = lookup_bridging_XYZ(L_XZ, L_XY, L_YZ)
    if verbose: print("condition concepts grounded! ")
    return mask_align, mask_XcmpZ, ground_results,ground_rconds

def tensor_logic_1and_case2_semantic(sparse_M, r1, r2, r3, semantic_similarities, verbose=False):
    mask_Yr1X = get_mask_by_rel_semantic(sparse_M, r1,semantic_similarities)
    mask_Yr2Z = get_mask_by_rel_semantic(sparse_M, r2,semantic_similarities)
    mask_Xr3Z = get_mask_by_rel_semantic(sparse_M, r3,semantic_similarities)
    mask_XcmpZ = mask_Yr1X.transpose(0, 1) @ mask_Yr2Z  
    mask_align = sparse_elementwise_mul_2d(mask_XcmpZ, mask_Xr3Z) 
    if verbose: print("concepts from rule body and rule head aligned! ")
    L_XZ=torch_sparse_coo_get_nonzero_list(mask_align)
    L_XY=torch_sparse_coo_get_nonzero_list(mask_Yr1X.transpose(0, 1))
    L_YZ=torch_sparse_coo_get_nonzero_list(mask_Yr2Z)
    ground_results = lookup_bridging_XYZ(L_XZ, L_XY, L_YZ)
    if verbose: print("bridging concepts grounded! ")
    L_XZ=torch_sparse_coo_get_nonzero_list(mask_XcmpZ)
    L_XY=torch_sparse_coo_get_nonzero_list(mask_Yr1X.transpose(0, 1))
    L_YZ=torch_sparse_coo_get_nonzero_list(mask_Yr2Z)
    ground_rconds = lookup_bridging_XYZ(L_XZ, L_XY, L_YZ)
    if verbose: print("condition concepts grounded! ")
    return mask_align, mask_XcmpZ, ground_results,ground_rconds

def tensor_logic_1and_case3_semantic(sparse_M, r1, r2, r3, semantic_similarities, verbose=False):
    mask_Xr1Y = get_mask_by_rel_semantic(sparse_M, r1,semantic_similarities)
    mask_Zr2Y = get_mask_by_rel_semantic(sparse_M, r2,semantic_similarities)
    mask_Xr3Z = get_mask_by_rel_semantic(sparse_M, r3,semantic_similarities)
    mask_XcmpZ = mask_Xr1Y @ mask_Zr2Y.transpose(0, 1)
    mask_align = sparse_elementwise_mul_2d(mask_XcmpZ, mask_Xr3Z) 
    if verbose: print("concepts from rule body and rule head aligned! ")
    L_XZ=torch_sparse_coo_get_nonzero_list(mask_align)
    L_XY=torch_sparse_coo_get_nonzero_list(mask_Xr1Y)
    L_YZ=torch_sparse_coo_get_nonzero_list(mask_Zr2Y.transpose(0, 1))
    ground_results = lookup_bridging_XYZ(L_XZ, L_XY, L_YZ)
    if verbose: print("bridging concepts grounded! ")
    L_XZ=torch_sparse_coo_get_nonzero_list(mask_XcmpZ)
    L_XY=torch_sparse_coo_get_nonzero_list(mask_Xr1Y)
    L_YZ=torch_sparse_coo_get_nonzero_list(mask_Zr2Y.transpose(0, 1))
    ground_rconds = lookup_bridging_XYZ(L_XZ, L_XY, L_YZ)
    if verbose: print("condition concepts grounded! ")
    return mask_align, mask_XcmpZ, ground_results,ground_rconds

def tensor_logic_1and_case4_semantic(sparse_M, r1, r2, r3, semantic_similarities, verbose=False):
    mask_Yr1X = get_mask_by_rel_semantic(sparse_M, r1,semantic_similarities)
    mask_Zr2Y = get_mask_by_rel_semantic(sparse_M, r2,semantic_similarities)
    mask_Xr3Z = get_mask_by_rel_semantic(sparse_M, r3,semantic_similarities)
    mask_XcmpZ = mask_Yr1X.transpose(0, 1) @ mask_Zr2Y.transpose(0, 1)
    mask_align = sparse_elementwise_mul_2d(mask_XcmpZ, mask_Xr3Z) 
    if verbose: print("concepts from rule body and rule head aligned! ")
    L_XZ=torch_sparse_coo_get_nonzero_list(mask_align)
    L_XY=torch_sparse_coo_get_nonzero_list(mask_Yr1X.transpose(0, 1))
    L_YZ=torch_sparse_coo_get_nonzero_list(mask_Zr2Y.transpose(0, 1))
    ground_results = lookup_bridging_XYZ(L_XZ, L_XY, L_YZ)
    if verbose: print("bridging concepts grounded! ")
    L_XZ=torch_sparse_coo_get_nonzero_list(mask_XcmpZ)
    L_XY=torch_sparse_coo_get_nonzero_list(mask_Yr1X.transpose(0, 1))
    L_YZ=torch_sparse_coo_get_nonzero_list(mask_Zr2Y.transpose(0, 1))
    ground_rconds = lookup_bridging_XYZ(L_XZ, L_XY, L_YZ)
    if verbose: print("condition concepts grounded! ")
    return mask_align, mask_XcmpZ, ground_results,ground_rconds


def tensor_logic_2and_case1_semantic(sparse_M, r1, r2, r3, r4,semantic_similarities, verbose=False):
    mask_Ar1B = get_mask_by_rel_semantic(sparse_M, r1,semantic_similarities)
    mask_Br2C = get_mask_by_rel_semantic(sparse_M, r2,semantic_similarities)
    mask_Cr3D = get_mask_by_rel_semantic(sparse_M, r3,semantic_similarities)
    mask_Ar4D = get_mask_by_rel_semantic(sparse_M, r4,semantic_similarities)
    mask_A_to_B = mask_Ar1B
    mask_A_to_C = mask_A_to_B @ mask_Br2C
    mask_A_to_D = mask_A_to_C @ mask_Cr3D
    mask_align = sparse_elementwise_mul_2d(mask_A_to_D, mask_Ar4D) 
    if verbose: print("concepts from rule body and rule head aligned! ")
    L_AD=torch_sparse_coo_get_nonzero_list(mask_align)
    L_AB=torch_sparse_coo_get_nonzero_list(mask_Ar1B)
    L_BC=torch_sparse_coo_get_nonzero_list(mask_Br2C)
    L_CD=torch_sparse_coo_get_nonzero_list(mask_Cr3D)
    ground_results = lookup_bridging_ABCD(L_AD, L_AB,L_BC,L_CD)
    if verbose: print("bridging concepts grounded! ")
    L_AD=torch_sparse_coo_get_nonzero_list(mask_A_to_D)
    L_AB=torch_sparse_coo_get_nonzero_list(mask_Ar1B)
    L_BC=torch_sparse_coo_get_nonzero_list(mask_Br2C)
    L_CD=torch_sparse_coo_get_nonzero_list(mask_Cr3D)
    ground_rconds = lookup_bridging_ABCD(L_AD, L_AB,L_BC,L_CD)
    if verbose: print("condition concepts grounded! ")
    return mask_align, mask_A_to_D, ground_results,ground_rconds

def tensor_logic_2and_case2_semantic(sparse_M, r1, r2, r3, r4,semantic_similarities, verbose=False):
    mask_Ar1B = get_mask_by_rel_semantic(sparse_M, r1,semantic_similarities)
    mask_Br2C = get_mask_by_rel_semantic(sparse_M, r2,semantic_similarities)
    mask_Dr3C = get_mask_by_rel_semantic(sparse_M, r3,semantic_similarities)
    mask_Ar4D = get_mask_by_rel_semantic(sparse_M, r4,semantic_similarities)
    mask_A_to_B = mask_Ar1B
    mask_A_to_C = mask_A_to_B @ mask_Br2C
    mask_A_to_D = mask_A_to_C @ mask_Dr3C.transpose(0,1)
    mask_align = sparse_elementwise_mul_2d(mask_A_to_D, mask_Ar4D) 
    if verbose: print("concepts from rule body and rule head aligned! ")
    L_AD=torch_sparse_coo_get_nonzero_list(mask_align)
    L_AB=torch_sparse_coo_get_nonzero_list(mask_Ar1B)
    L_BC=torch_sparse_coo_get_nonzero_list(mask_Br2C)
    L_CD=torch_sparse_coo_get_nonzero_list(mask_Dr3C.transpose(0,1))
    ground_results = lookup_bridging_ABCD(L_AD, L_AB,L_BC,L_CD)
    if verbose: print("bridging concepts grounded! ")
    L_AD=torch_sparse_coo_get_nonzero_list(mask_A_to_D)
    ground_rconds = lookup_bridging_ABCD(L_AD, L_AB,L_BC,L_CD)
    if verbose: print("condition concepts grounded! ")
    return mask_align, mask_A_to_D, ground_results,ground_rconds


def tensor_logic_2and_case3_semantic(sparse_M, r1, r2, r3, r4,semantic_similarities, verbose=False):
    mask_Ar1B = get_mask_by_rel_semantic(sparse_M, r1,semantic_similarities)
    mask_Cr2B = get_mask_by_rel_semantic(sparse_M, r2,semantic_similarities)
    mask_Cr3D = get_mask_by_rel_semantic(sparse_M, r3,semantic_similarities)
    mask_Ar4D = get_mask_by_rel_semantic(sparse_M, r4,semantic_similarities)
    mask_A_to_B = mask_Ar1B
    mask_A_to_C = mask_A_to_B @ mask_Cr2B.transpose(0,1)
    mask_A_to_D = mask_A_to_C @ mask_Cr3D
    mask_align = sparse_elementwise_mul_2d(mask_A_to_D, mask_Ar4D) 
    if verbose: print("concepts from rule body and rule head aligned! ")
    L_AD=torch_sparse_coo_get_nonzero_list(mask_align)
    L_AB=torch_sparse_coo_get_nonzero_list(mask_Ar1B)
    L_BC=torch_sparse_coo_get_nonzero_list(mask_Cr2B.transpose(0,1))
    L_CD=torch_sparse_coo_get_nonzero_list(mask_Cr3D)
    ground_results = lookup_bridging_ABCD(L_AD, L_AB,L_BC,L_CD)
    if verbose: print("bridging concepts grounded! ")
    L_AD=torch_sparse_coo_get_nonzero_list(mask_A_to_D)
    ground_rconds = lookup_bridging_ABCD(L_AD, L_AB,L_BC,L_CD)
    if verbose: print("condition concepts grounded! ")
    return mask_align, mask_A_to_D, ground_results,ground_rconds

def tensor_logic_2and_case4_semantic(sparse_M, r1, r2, r3, r4, semantic_similarities,verbose=False):
    mask_Br1A = get_mask_by_rel_semantic(sparse_M, r1,semantic_similarities)
    mask_Br2C = get_mask_by_rel_semantic(sparse_M, r2,semantic_similarities)
    mask_Cr3D = get_mask_by_rel_semantic(sparse_M, r3,semantic_similarities)
    mask_Ar4D = get_mask_by_rel_semantic(sparse_M, r4,semantic_similarities)
    mask_A_to_B = mask_Br1A.transpose(0,1)
    mask_A_to_C = mask_A_to_B @ mask_Br2C
    mask_A_to_D = mask_A_to_C @ mask_Cr3D
    mask_align = sparse_elementwise_mul_2d(mask_A_to_D, mask_Ar4D) 
    if verbose: print("concepts from rule body and rule head aligned! ")
    L_AD=torch_sparse_coo_get_nonzero_list(mask_align)
    L_AB=torch_sparse_coo_get_nonzero_list(mask_Br1A.transpose(0,1))
    L_BC=torch_sparse_coo_get_nonzero_list(mask_Br2C)
    L_CD=torch_sparse_coo_get_nonzero_list(mask_Cr3D)
    ground_results = lookup_bridging_ABCD(L_AD, L_AB,L_BC,L_CD)
    if verbose: print("bridging concepts grounded! ")
    L_AD=torch_sparse_coo_get_nonzero_list(mask_A_to_D)
    ground_rconds = lookup_bridging_ABCD(L_AD, L_AB,L_BC,L_CD)
    if verbose: print("condition concepts grounded! ")
    return mask_align, mask_A_to_D, ground_results,ground_rconds

def tensor_logic_2and_case5_semantic(sparse_M, r1, r2, r3, r4, semantic_similarities,verbose=False):
    mask_Ar1B = get_mask_by_rel_semantic(sparse_M, r1,semantic_similarities)
    mask_Cr2B = get_mask_by_rel_semantic(sparse_M, r2,semantic_similarities)
    mask_Dr3C = get_mask_by_rel_semantic(sparse_M, r3,semantic_similarities)
    mask_Ar4D = get_mask_by_rel_semantic(sparse_M, r4,semantic_similarities)
    mask_A_to_B = mask_Ar1B
    mask_A_to_C = mask_A_to_B @ mask_Cr2B.transpose(0,1)
    mask_A_to_D = mask_A_to_C @ mask_Dr3C.transpose(0,1)
    mask_align = sparse_elementwise_mul_2d(mask_A_to_D, mask_Ar4D) 
    if verbose: print("concepts from rule body and rule head aligned! ")
    L_AD=torch_sparse_coo_get_nonzero_list(mask_align)
    L_AB=torch_sparse_coo_get_nonzero_list(mask_Ar1B)
    L_BC=torch_sparse_coo_get_nonzero_list(mask_Cr2B.transpose(0,1))
    L_CD=torch_sparse_coo_get_nonzero_list(mask_Dr3C.transpose(0,1))
    ground_results = lookup_bridging_ABCD(L_AD, L_AB,L_BC,L_CD)
    if verbose: print("bridging concepts grounded! ")
    L_AD=torch_sparse_coo_get_nonzero_list(mask_A_to_D)
    ground_rconds = lookup_bridging_ABCD(L_AD, L_AB,L_BC,L_CD)
    if verbose: print("condition concepts grounded! ")
    return mask_align, mask_A_to_D, ground_results,ground_rconds

def tensor_logic_2and_case6_semantic(sparse_M, r1, r2, r3, r4,semantic_similarities, verbose=False):
    mask_Br1A = get_mask_by_rel_semantic(sparse_M, r1,semantic_similarities)
    mask_Br2C = get_mask_by_rel_semantic(sparse_M, r2,semantic_similarities)
    mask_Dr3C = get_mask_by_rel_semantic(sparse_M, r3,semantic_similarities)
    mask_Ar4D = get_mask_by_rel_semantic(sparse_M, r4,semantic_similarities)
    mask_A_to_B = mask_Br1A.transpose(0,1)
    mask_A_to_C = mask_A_to_B @ mask_Br2C
    mask_A_to_D = mask_A_to_C @ mask_Dr3C.transpose(0,1)
    mask_align = sparse_elementwise_mul_2d(mask_A_to_D, mask_Ar4D)
    if verbose: print("concepts from rule body and rule head aligned! ")
    L_AD=torch_sparse_coo_get_nonzero_list(mask_align)
    L_AB=torch_sparse_coo_get_nonzero_list(mask_Br1A.transpose(0,1))
    L_BC=torch_sparse_coo_get_nonzero_list(mask_Br2C)
    L_CD=torch_sparse_coo_get_nonzero_list(mask_Dr3C.transpose(0,1))
    ground_results = lookup_bridging_ABCD(L_AD, L_AB,L_BC,L_CD)
    if verbose: print("bridging concepts grounded! ")
    L_AD=torch_sparse_coo_get_nonzero_list(mask_A_to_D)
    ground_rconds = lookup_bridging_ABCD(L_AD, L_AB,L_BC,L_CD)
    if verbose: print("condition concepts grounded! ")
    return mask_align, mask_A_to_D, ground_results,ground_rconds

def tensor_logic_2and_case7_semantic(sparse_M, r1, r2, r3, r4,semantic_similarities, verbose=False):
    mask_Br1A = get_mask_by_rel_semantic(sparse_M, r1,semantic_similarities)
    mask_Cr2B = get_mask_by_rel_semantic(sparse_M, r2,semantic_similarities)
    mask_Cr3D = get_mask_by_rel_semantic(sparse_M, r3,semantic_similarities)
    mask_Ar4D = get_mask_by_rel_semantic(sparse_M, r4,semantic_similarities)
    mask_A_to_B = mask_Br1A.transpose(0,1)
    mask_A_to_C = mask_A_to_B @ mask_Cr2B.transpose(0,1)
    mask_A_to_D = mask_A_to_C @ mask_Cr3D
    mask_align = sparse_elementwise_mul_2d(mask_A_to_D, mask_Ar4D) 
    if verbose: print("concepts from rule body and rule head aligned! ")
    L_AD=torch_sparse_coo_get_nonzero_list(mask_align)
    L_AB=torch_sparse_coo_get_nonzero_list(mask_Br1A.transpose(0,1))
    L_BC=torch_sparse_coo_get_nonzero_list(mask_Cr2B.transpose(0,1))
    L_CD=torch_sparse_coo_get_nonzero_list(mask_Cr3D)
    ground_results = lookup_bridging_ABCD(L_AD, L_AB,L_BC,L_CD)
    if verbose: print("bridging concepts grounded! ")
    L_AD=torch_sparse_coo_get_nonzero_list(mask_A_to_D)
    ground_rconds = lookup_bridging_ABCD(L_AD, L_AB,L_BC,L_CD)
    if verbose: print("condition concepts grounded! ")
    return mask_align, mask_A_to_D, ground_results,ground_rconds
    

def tensor_logic_2and_case8_semantic(sparse_M, r1, r2, r3, r4,semantic_similarities, verbose=False):
    mask_Br1A = get_mask_by_rel_semantic(sparse_M, r1,semantic_similarities)
    mask_Cr2B = get_mask_by_rel_semantic(sparse_M, r2,semantic_similarities)
    mask_Dr3C = get_mask_by_rel_semantic(sparse_M, r3,semantic_similarities)
    mask_Ar4D = get_mask_by_rel_semantic(sparse_M, r4,semantic_similarities)
    mask_A_to_B = mask_Br1A.transpose(0,1)
    mask_A_to_C = mask_A_to_B @ mask_Cr2B.transpose(0,1)
    mask_A_to_D = mask_A_to_C @ mask_Dr3C.transpose(0,1)
    mask_align = sparse_elementwise_mul_2d(mask_A_to_D, mask_Ar4D) 
    if verbose: print("concepts from rule body and rule head aligned! ")
    L_AD=torch_sparse_coo_get_nonzero_list(mask_align)
    L_AB=torch_sparse_coo_get_nonzero_list(mask_Br1A.transpose(0,1))
    L_BC=torch_sparse_coo_get_nonzero_list(mask_Cr2B.transpose(0,1))
    L_CD=torch_sparse_coo_get_nonzero_list(mask_Dr3C.transpose(0,1))
    ground_results = lookup_bridging_ABCD(L_AD, L_AB,L_BC,L_CD)
    if verbose: print("bridging concepts grounded! ")
    L_AD=torch_sparse_coo_get_nonzero_list(mask_A_to_D)
    ground_rconds = lookup_bridging_ABCD(L_AD, L_AB,L_BC,L_CD)
    if verbose: print("condition concepts grounded! ")
    return mask_align, mask_A_to_D, ground_results,ground_rconds


def ground_rules_over_kg_semantic(kg_sparse, checked_rules, relation2id_dict, semantic_similarities,verbose=False):
    aligned_masks, chained_masks, ground_results,ground_rconds = [],[],[],[]
    idx = 0
    relation2id_dict = {remove_wikidata_prefix(key): value for key, value in relation2id_dict.items()}
    for checked_rule in checked_rules:
        rule_text, rule_type, rule_cond, rule_head = checked_rule
        rule_rel_ids = prepare_rule_map_relations(rule_cond, rule_head, relation2id_dict, verbose=verbose)
        if rule_rel_ids is None:
            if verbose: print("rule {} contains relation that does not exist in KG, skip".format(rule_text))
            ground_result = []
            ground_rcond = []
            mask_aligned = None
            mask_chained = None
        else:
            if rule_type == "01":
                mask_aligned, mask_chained, ground_result, ground_rcond = tensor_logic_0and_case1_semantic(kg_sparse, rule_rel_ids[0],rule_rel_ids[1],semantic_similarities)
            elif rule_type == "02":
                mask_aligned, mask_chained,ground_result, ground_rcond = tensor_logic_0and_case2_semantic(kg_sparse, rule_rel_ids[0],rule_rel_ids[1],semantic_similarities)
            elif rule_type == "11":
                mask_aligned, mask_chained,ground_result, ground_rcond = tensor_logic_1and_case1_semantic(kg_sparse, rule_rel_ids[0],rule_rel_ids[1],rule_rel_ids[2],semantic_similarities)
            elif rule_type == "12":
                mask_aligned, mask_chained,ground_result, ground_rcond = tensor_logic_1and_case2_semantic(kg_sparse, rule_rel_ids[0],rule_rel_ids[1],rule_rel_ids[2],semantic_similarities)
            elif rule_type == "13":
                mask_aligned, mask_chained,ground_result, ground_rcond = tensor_logic_1and_case3_semantic(kg_sparse, rule_rel_ids[0],rule_rel_ids[1],rule_rel_ids[2],semantic_similarities)
            elif rule_type == "14":
                mask_aligned, mask_chained,ground_result, ground_rcond = tensor_logic_1and_case4_semantic(kg_sparse, rule_rel_ids[0],rule_rel_ids[1],rule_rel_ids[2],semantic_similarities)
            elif rule_type == "21":
                mask_aligned, mask_chained,ground_result, ground_rcond = tensor_logic_2and_case1_semantic(kg_sparse, rule_rel_ids[0],rule_rel_ids[1],rule_rel_ids[2],rule_rel_ids[3],semantic_similarities)
            elif rule_type == "22":
                mask_aligned, mask_chained,ground_result, ground_rcond = tensor_logic_2and_case2_semantic(kg_sparse, rule_rel_ids[0],rule_rel_ids[1],rule_rel_ids[2],rule_rel_ids[3],semantic_similarities)
            elif rule_type == "23":
                mask_aligned, mask_chained,ground_result, ground_rcond = tensor_logic_2and_case3_semantic(kg_sparse, rule_rel_ids[0],rule_rel_ids[1],rule_rel_ids[2],rule_rel_ids[3],semantic_similarities)
            elif rule_type == "24":
                mask_aligned, mask_chained,ground_result, ground_rcond = tensor_logic_2and_case4_semantic(kg_sparse, rule_rel_ids[0],rule_rel_ids[1],rule_rel_ids[2],rule_rel_ids[3],semantic_similarities)
            elif rule_type == "25":
                mask_aligned, mask_chained,ground_result, ground_rcond = tensor_logic_2and_case5_semantic(kg_sparse, rule_rel_ids[0],rule_rel_ids[1],rule_rel_ids[2],rule_rel_ids[3],semantic_similarities)
            elif rule_type == "26":
                mask_aligned, mask_chained,ground_result, ground_rcond = tensor_logic_2and_case6_semantic(kg_sparse, rule_rel_ids[0],rule_rel_ids[1],rule_rel_ids[2],rule_rel_ids[3],semantic_similarities)
            elif rule_type == "27":
                mask_aligned, mask_chained,ground_result, ground_rcond = tensor_logic_2and_case7_semantic(kg_sparse, rule_rel_ids[0],rule_rel_ids[1],rule_rel_ids[2],rule_rel_ids[3],semantic_similarities)
            elif rule_type == "28":
                mask_aligned, mask_chained,ground_result, ground_rcond = tensor_logic_2and_case8_semantic(kg_sparse, rule_rel_ids[0],rule_rel_ids[1],rule_rel_ids[2],rule_rel_ids[3],semantic_similarities)
            else:
                raise Exception("invalid rule_type={}".format(rule_type))
        ground_results.append(ground_result)
        ground_rconds.append(ground_rcond)
        aligned_masks.append(mask_aligned)
        chained_masks.append(mask_chained)
        if verbose: print("{}: type {}, {}/{} aligned".format(idx, rule_type, len(ground_result),len(ground_rcond)))
        idx += 1
    return aligned_masks, chained_masks, ground_results,ground_rconds

def keep_good_rules(checked_rules, aligned_masks, chained_masks, ground_results, ground_rconds, verbose=False, criteria="chain"):
    assert len(ground_results) == len(ground_rconds)
    assert len(aligned_masks) == len(chained_masks)
    assert len(checked_rules) == len(aligned_masks)
    assert criteria in ["ground", "chain", "chain3","chain5", "chain10"]
    n_rules = len(ground_results)
    good_checked_rules, good_rule_aligned_masks,  good_rule_chained_masks, good_rule_ground_results, good_rule_ground_rconds = [],[],[],[],[]
    for idx in range(n_rules):
        is_good_rule = False
        if criteria == "ground":
            is_good_rule = (len(ground_results[idx]) != 0 or len(ground_rconds[idx]) != 0)
        elif criteria == "chain":
            is_good_rule = (len(ground_results[idx]) != 0)
        elif criteria == "chain3":
            is_good_rule = (len(ground_results[idx]) >= 3)
        elif criteria == "chain5":
            is_good_rule = (len(ground_results[idx]) >= 5)
        elif criteria == "chain10":
            is_good_rule = (len(ground_results[idx]) >= 10)
        else:
            raise Exception("invalid keep good rule critiera {}".format(criteria))
        if is_good_rule:
            good_checked_rules.append(checked_rules[idx])
            good_rule_aligned_masks.append(aligned_masks[idx])
            good_rule_chained_masks.append(chained_masks[idx])
            good_rule_ground_results.append(ground_results[idx])
            good_rule_ground_rconds.append(ground_rconds[idx])
    if verbose: print("\t{} out of {} are good groundable rules".format(len(good_checked_rules), n_rules))
    return good_checked_rules, good_rule_aligned_masks,  good_rule_chained_masks, good_rule_ground_results, good_rule_ground_rconds

def save_grounding_results(save_prefix, aligned_masks, chained_masks, ground_results, ground_rconds):
    aligned_mask_fname = save_prefix + "_mask_aligned.pt"
    torch.save(aligned_masks, aligned_mask_fname)
    chained_mask_fname = save_prefix + "_mask_chained.pt"
    torch.save(chained_masks, chained_mask_fname)
    chained_grnd_fname =  save_prefix + "_grnd_chained.json"
    with open(chained_grnd_fname, 'w') as file:
        json.dump(ground_rconds, file)
    aligned_grnd_fname =  save_prefix + "_grnd_aligned.json"
    with open(aligned_grnd_fname, 'w') as file:
        json.dump(ground_results, file)


def load_grounding_results(save_prefix):
    aligned_mask_fname = save_prefix + "_mask_aligned.pt"
    aligned_masks=torch.load(aligned_mask_fname)
    chained_mask_fname = save_prefix + "_mask_chained.pt"
    chained_masks=torch.load(chained_mask_fname)
    chained_grnd_fname = save_prefix + "_grnd_chained.json"
    with open(chained_grnd_fname, 'r') as file:
        ground_rconds = json.load(file)
    aligned_grnd_fname = save_prefix + "_grnd_aligned.json"
    with open(aligned_grnd_fname, 'r') as file:
        ground_results = json.load(file)
    return aligned_masks, chained_masks, ground_results, ground_rconds

def print_grounding_stats(checked_rules, ground_rconds, ground_results):
    for idx in range(len(checked_rules)):
        checked_rule = checked_rules[idx]
        ground_rcond = ground_rconds[idx]
        ground_result = ground_results[idx]
        rule_text, rule_type, rule_cond, rule_head = checked_rule
        print("{}/{} aligned to type {} rule {}".format(len(ground_result), len(ground_rcond), rule_type, rule_text))


def get_unique_eval_query_ids(trainset_triplets, evalset_triplets):
    eval_only_ids = []
    for idx in range(len(evalset_triplets)):
        if evalset_triplets[idx] not in trainset_triplets:
            eval_only_ids.append(idx)
    return eval_only_ids


def _get_triplets_and_scores_old(sparse_M, aligned_masks, chained_masks, relation_wk_idx, verbose=True):
    relation_triplets = torch_sparse_coo_get_nonzero_list(torch_sparse_coo_3d_slicing_by_rel(sparse_M, relation_wk_idx))
    n_triplets = len(relation_triplets)
    n_rules = len(aligned_masks)
    relation_scores = torch.zeros((n_triplets, n_rules))
    for triplet_id in range(n_triplets):
        subj_id, obj_id = relation_triplets[triplet_id]
        for rule_id in range(n_rules):
            aligned = aligned_masks[rule_id][subj_id, obj_id]
            chained = chained_masks[rule_id][subj_id, obj_id]
            if chained == 0:
                score = 0
            elif aligned == 0:
                score = -chained
            else:
                score = aligned
            relation_scores[triplet_id, rule_id]=score
    nonzero_mask = (relation_scores.sum(dim=1) != 0)
    nonzero_triplets = []
    for idx in range(len(relation_triplets)):
        if nonzero_mask[idx]:
            nonzero_triplets.append(relation_triplets[idx])
    nonzero_scores = relation_scores[nonzero_mask]
    assert len(nonzero_triplets) == nonzero_scores.shape[0]
    if verbose: print("{} out of {} trainset queries are answerable".format(len(nonzero_triplets), len(relation_triplets)))
    return nonzero_triplets, nonzero_scores

def get_triplets_and_scores(sparse_M, aligned_masks, chained_masks, relation_wk_idx, kge_model, all_true_triple, n_entity, n_relation, kge_bsize=32, cpu_num=10, device='cuda',dataset_name=None, verbose=True):
    relation_triplets = torch_sparse_coo_get_nonzero_list(torch_sparse_coo_3d_slicing_by_rel(sparse_M, relation_wk_idx))
    n_triplets = len(relation_triplets)
    n_rules = len(aligned_masks)
    logical_rule_scores = torch.zeros((n_triplets, n_rules), device=device)
    for triplet_id in range(n_triplets):
        subj_id, obj_id = relation_triplets[triplet_id]
        for rule_id in range(n_rules):
            aligned = aligned_masks[rule_id][subj_id, obj_id]
            chained = chained_masks[rule_id][subj_id, obj_id]
            if chained == 0:
                score = 0
            elif aligned == 0:
                score = -chained
            else:
                score = aligned
            logical_rule_scores[triplet_id, rule_id]=score
    nonzero_mask = (logical_rule_scores.sum(dim=1) != 0)
    effective_triplets = []
    for idx in range(n_triplets):
        if nonzero_mask[idx]:
            effective_triplets.append(relation_triplets[idx])
    logical_rule_scores = logical_rule_scores[nonzero_mask]
    assert len(effective_triplets) == logical_rule_scores.shape[0]
    if verbose: print("{} out of {} queries are effective".format(len(effective_triplets), n_triplets))
    if len(effective_triplets) == 0:
        return None, None
    n_triplets = len(effective_triplets)
    embedding_rule_scores = torch.zeros((n_triplets, 1), device=device)
    tt = torch.Tensor(effective_triplets)
    R = relation_wk_idx
    R_tensor = torch.full((tt.size(0), 1), R, dtype=torch.int64, device=tt.device)
    effective_quries = torch.cat((tt[:, 0].unsqueeze(1), R_tensor, tt[:, 1].unsqueeze(1)), dim=1)
    effective_quries = effective_quries.to(torch.int64)
    effective_quries_arr = effective_quries.to(torch.int64).numpy()
    try:
        _, _, _, kge_scores, kge_preds = kge_inference(kge_model, effective_quries_arr, all_true_triple, n_entity,n_relation,cpu_num,kge_bsize, use_cuda=True, tail_only=True)
    except torch.cuda.OutOfMemoryError: 
        print("CUDA OOM. Falling back to CPU for KGE. dataset_name={}".format(dataset_name))
        torch.cuda.empty_cache() 
        kge_model_cpu = get_KGE(dataset_name,use_cuda=False)
        _, _, _, kge_scores, kge_preds = kge_inference(kge_model_cpu, effective_quries_arr, all_true_triple, n_entity,n_relation,cpu_num,kge_bsize, use_cuda=False, tail_only=True)
    kge_scores = kge_scores.to(device) ; torch.cuda.empty_cache() 
    try: 
        argsort = torch.argsort(kge_scores, dim=1, descending=True)
    except torch.cuda.OutOfMemoryError: 
        print("CUDA OOM. Falling back to CPU for argsort.")
        torch.cuda.empty_cache()
        argsort = torch.argsort(kge_scores.cpu(), dim=1, descending=True).to(kge_scores.device)
    for j in range(n_triplets):
        ranking = (argsort[j, :] == effective_quries_arr[j, 2]).nonzero() 
        assert ranking.size(0) == 1
        rank = 1 + ranking.item()
        if rank <= 1:
            embedding_rule_scores[j] += 1.0
        if rank <= 3:
            embedding_rule_scores[j] += 0.3
        if rank <= 10:
            embedding_rule_scores[j] += 0.1
    assert embedding_rule_scores.size() == (n_triplets, 1)
    rule_scores = torch.cat((logical_rule_scores, embedding_rule_scores), dim=1)
    return effective_triplets, rule_scores


class ReasonerModel(nn.Module):
    def __init__(self, num_rj):
        super(ReasonerModel, self).__init__()
        self.raw_weights = nn.Parameter(torch.zeros(num_rj))
    def forward(self, sij):
        weights = torch.softmax(self.raw_weights, dim=0)
        masked_sij = torch.where(sij > 0, sij, torch.tensor(0.0))
        final_score = torch.matmul(masked_sij, weights)
        return final_score, weights

class ReasonerModelPlus(nn.Module):
    def __init__(self, num_rj):
        super(ReasonerModelPlus, self).__init__()
        self.n_logical_rules = num_rj - 1
        self.raw_weights = nn.Parameter(torch.zeros(num_rj - 1))
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))
    def forward(self, sij):
        weights = torch.softmax(self.raw_weights, dim=0)
        weighted_sum = torch.matmul(sij[:, :-1], weights)
        alpha = torch.sigmoid(self.raw_alpha)
        final_score = alpha * weighted_sum + (1 - alpha) * sij[:, -1]
        return final_score, weights

   
def train_loop(model, epoch, optimizer, scheduler, trainset_scores, validset_scores, epsilon=1e-32,verbose=True):
    model.train()    
    optimizer.zero_grad()
    outputs, weights = model(trainset_scores)
    loss = - torch.sum(torch.log(outputs + epsilon))
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    if verbose and (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}], Loss: {loss.item():.4f}')
    model.eval()
    with torch.no_grad():
        valid_outputs, _ = model(validset_scores)
        valid_loss = - torch.sum(torch.log(valid_outputs + epsilon))
    return model, valid_loss.item()


def print_rule_weights_and_quality(n_rules, learned_weights, train_ground_rconds, train_ground_results, top_k):
    print("top-K best rules: ", torch.topk(torch.exp(learned_weights), top_k, largest=True).indices)
    rule_perf_on_train = []
    for rid in range(n_rules):
        n_chained = len(train_ground_rconds[rid])
        n_aligned = len(train_ground_results[rid])
        if n_chained == 0:
            rule_perf = 0 
        else:
            rule_perf = n_aligned / n_chained
        rule_perf_on_train.append(rule_perf)
    print("top-K grnd rules: ", torch.topk(torch.tensor(rule_perf_on_train), top_k, largest=True).indices)
    for rule_id in range(n_rules):
        print("id={}, w={:.4f}, r={:.4f}, n={}/{}".format(rule_id, learned_weights[rule_id].item(),rule_perf_on_train[rule_id], len(train_ground_results[rule_id]),len(train_ground_rconds[rule_id])))



def test_loop(test_queries, learned_weights, n_rules, n_entities, n_relations, all_true_triple, test_chained_masks, kge_model, b_size, cpu_num=10, dataset_name=None, device="cuda"):
    n_logical = n_rules - 1
    logical_rule_weights = learned_weights[:-1]
    emb_weight = learned_weights[-1]
    logical_rule_preds_batch = torch.zeros((len(test_queries), n_logical, n_entities)).to(device)
    predictions = torch.zeros((len(test_queries), n_entities)).to(device)
    for batch_start in range(0, len(test_queries), b_size):
        batch_end = min(batch_start + b_size, len(test_queries))
        batch_queries = test_queries[batch_start:batch_end]
        for b_idx, test_query in enumerate(batch_queries):
            query_subj, query_rel, query_obj = test_query[0], test_query[1], test_query[2]
            rule_chained_results = []
            for rid in range(n_logical):
                rule_chained_mask = test_chained_masks[rid]
                chained_entity = []
                for grnd_tpl in torch_sparse_coo_get_nonzero_list(rule_chained_mask):
                    if grnd_tpl[0] == query_subj:
                        chained_entity.append(grnd_tpl[1])
                rule_chained_results.append(chained_entity)
            for i, indices in enumerate(rule_chained_results):
                logical_rule_preds_batch[batch_start + b_idx, i, indices] = logical_rule_weights[i]
        _, _, _, kge_scores, kge_preds = kge_inference(kge_model, batch_queries, all_true_triple, n_entities,n_relations,cpu_num=cpu_num,test_batch_size=b_size, use_cuda=True, tail_only=True)
        scores_batch = kge_scores
        for b_idx, test_query in enumerate(batch_queries):
            logical_rule_preds = logical_rule_preds_batch[batch_start + b_idx]
            query_score = scores_batch[b_idx] 
            prediction = torch.vstack((logical_rule_preds, emb_weight * query_score.unsqueeze(0)))
            prediction = torch.sum(prediction, axis=0)
            predictions[batch_start + b_idx] = prediction
    return predictions


def test_loop_plus(test_queries, learned_weights, learned_alpha, n_rules, n_entities, n_relations, all_true_triple, test_chained_masks, kge_model, b_size, cpu_num=10, dataset_name=None, device="cuda"):
    n_logical = n_rules - 1
    logical_rule_weights = learned_weights
    alpha = learned_alpha
    logical_rule_preds_batch = torch.zeros((len(test_queries), n_logical, n_entities)).to(device)
    predictions = torch.zeros((len(test_queries), n_entities)).to(device)
    for batch_start in range(0, len(test_queries), b_size):
        batch_end = min(batch_start + b_size, len(test_queries))
        batch_queries = test_queries[batch_start:batch_end]
        for b_idx, test_query in enumerate(batch_queries):
            query_subj, query_rel, query_obj = test_query[0], test_query[1], test_query[2]
            rule_chained_results = []
            for rid in range(n_logical):
                rule_chained_mask = test_chained_masks[rid]
                chained_entity = []
                for grnd_tpl in torch_sparse_coo_get_nonzero_list(rule_chained_mask):
                    if grnd_tpl[0] == query_subj:
                        chained_entity.append(grnd_tpl[1])
                rule_chained_results.append(chained_entity)
            for i, indices in enumerate(rule_chained_results):
                logical_rule_preds_batch[batch_start + b_idx, i, indices] = logical_rule_weights[i]
        _, _, _, kge_scores, kge_preds = kge_inference(kge_model, batch_queries, all_true_triple, n_entities,n_relations,cpu_num=cpu_num,test_batch_size=b_size, use_cuda=True, tail_only=True)
        scores_batch = kge_scores
        for b_idx, test_query in enumerate(batch_queries):
            logical_rule_preds = logical_rule_preds_batch[batch_start + b_idx]
            query_score = scores_batch[b_idx] 
            logical_rule_combined = torch.sum(logical_rule_preds, axis=0)  
            final_prediction = alpha * logical_rule_combined + (1 - alpha) * query_score
            predictions[batch_start + b_idx] = final_prediction
    return predictions


def compute_metrics(ranks, k_vals=[1,3,10]):
    mean_rank = torch.mean(ranks.float())
    reciprocal_ranks = 1.0 / ranks.float()
    mean_reciprocal_rank = torch.mean(reciprocal_ranks)
    hits_at_K = []
    for K in k_vals:
        hits = (ranks <= K).float()
        hit_at_K = torch.mean(hits)
        hits_at_K.append(hit_at_K.item())
    return mean_rank.item(), mean_reciprocal_rank.item(), hits_at_K