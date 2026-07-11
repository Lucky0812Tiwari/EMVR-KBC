import argparse
import logging
import sys
import os
import json
import yaml
import numpy as np
import random
import pandas as pd
import time
import torch
import torch.optim as optim

from kge import get_KGE, kge_inference

from data import get_KG_data, encode_kg_to_arr, convert_kgtxt_to_natlang_plus

from utils import file_exists, load_nested_list, save_nested_list,is_empty_dir, _build_inverse_dict

from extractor import get_nhop_closed_path, save_subgraphs, load_subgraphs

from proposer import convert_subgraphs_to_inputs, llm_input_len_check, llm_propose_rule

from reasoner import prepare_rules, get_model_and_tokenizer, get_phrase_embedding, custom_similarity, find_similar_phrases, convert_arr_to_sparse_coo,  load_grounding_results, save_grounding_results, ground_rules_over_kg_semantic, keep_good_rules, print_grounding_stats, get_unique_eval_query_ids, ReasonerModel, ReasonerModelPlus, train_loop, get_triplets_and_scores, compute_metrics,test_loop,test_loop_plus

# ---------------- EMVR-KBC (NEW) ----------------
from evidence import (build_adjacency, verify_rules_for_relation, apply_warmstart,
                       ExternalEvidenceIndex, filtering_recall, ev_weight_correlation,
                       grounding_flops_saved)
# --------------------------------------------------

USE_LONG_REL_TEXT = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="LeSR: Project Code KREA")
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')
    parser.add_argument('-c', '--config_file', type=str, required=False, help='Path to the configuration file')
    parser.add_argument('--debug', action='store_true', help='whether to run script in debug mode')
    parser.add_argument('--debug_relation_limit', type=int, default=None,
        help='quick-debug: run only the first N relations (in dataset order) instead of all/debug_idx. '
             'Independent of --debug; e.g. --debug_relation_limit 3 for a fast smoke test.')
    parser.add_argument("--rand_seed", type=int, help="random seed for reproducability", default=5)
    parser.add_argument('-name', '--run_name', type=str, help='Path to the configuration file',default="a_simple_run")
    parser.add_argument('-dir', '--run_dir', type=str, help='Path to the run directory',default=None)
    parser.add_argument("--dataset", type=str, help="name of KG dataset", choices=["UMLs", "FB15K", "WN18RR", "WD15K","ConceptNet"], default="WD15K")
    parser.add_argument('--run_extractor', action='store_true', help='whether to run subgraph extractor')
    parser.add_argument("--max_hop", type=int, help="max number of hops to build KG subgraph", default=3)
    parser.add_argument("--max_neighbor", type=int, help="max number of neighboring nodes for each hop to build KG subgraph", default=3)
    parser.add_argument("--max_sample", type=int, help="max number of KG subgraphs to build", default=30)
    parser.add_argument("--max_subgraph", type=int, help="max number of KG subgraphs to put into llm proposer. this is in case max_sample set to a larger value when the kg is very sparse and max_sample has too little subgraphs with closed path", default=None)
    parser.add_argument('--run_proposer', action='store_true', help='whether to run llm proposer')
    parser.add_argument("--llm_name", type=str, help="name of llm as rule proposer", choices=["gpt35","gpt40","gemini15","llama3"], default="gpt35")
    parser.add_argument("--llm_max_input_chars", type=int, help="max number of chars for llm input", default=2048*4*1.2)
    parser.add_argument("--llm_api_key", type=str, help="api key for llm model")
    parser.add_argument('--run_reasoner', action='store_true', help='whether to run rule reasoner')
    parser.add_argument('--use_reasoner_plus', action='store_true', help='whether the reasoner should use the plus model')
    parser.add_argument('--reason_with_relation_semantic', action='store_true', help='whether the reasoner should consider relation semantic')
    parser.add_argument("--semantic_lm", type=str, help="name of lm used for relation semantic", choices=["sentence-transformers/paraphrase-MiniLM-L6-v2", "t5-base", "t5-large", "roberta-base", "roberta-large", "bert-base-uncased", "bert-large-uncased"], default="sentence-transformers/paraphrase-MiniLM-L6-v2")
    parser.add_argument('--set_semantic_threshold', action='store_true', help='whether to set a min similairty threshold for relation semantics')
    parser.add_argument("--semantic_threshold_value", type=float, help="threshold for relation semantic", default=0.60)
    parser.add_argument("--semantic_representation", type=str, help="method of obtaining relation representation. using sentence transformer will override this argument", choices=["cls","mean","max"], default="cls")
    parser.add_argument("--kge_bsize", type=int, help="eval batch size of KGE", default=32)
    parser.add_argument("--good_rule_criteria", type=str, help="criteria for good learnable rules", choices=["ground","chain","chain3","chain5", "chain10"], default="chain")
    parser.add_argument('--compute_rule_scoring_on_cpu', action='store_true', help='whether to compute rule quality scoring on cpu device instead of cuda')
    parser.add_argument('--run_reasoner_models_on_cpu', action='store_true', help='whether to run reasoner models on cpu device instead of cuda')
    parser.add_argument("--num_epochs", type=int, help="max number of epoch for rule learning", default=1000)
    parser.add_argument("--initial_lr", type=float, help="initial learning rate for rule learning", default=0.001)
    parser.add_argument("--weight_decay", type=float, help="weight decay for rule learning", default=0.01)
    parser.add_argument("--scheduler_step", type=int, help="scheduler step for rule learning", default=100)
    parser.add_argument("--scheduler_gamma", type=int, help="scheduler gamma for rule learning", default=0.1)
    parser.add_argument("--early_stop_patience", type=int, help="number of epochs to wait before early stopping", default=30)
    parser.add_argument('--exclude_unlearnable_relation', action='store_true', help='whether queries with unlearnable relations are included when computing metrics')
    # ---------------- EMVR-KBC (NEW) ----------------
    parser.add_argument('--use_emvr', action='store_true',
        help='enable EMVR-KBC evidence verification before grounding')
    parser.add_argument('--emvr_sample_size', type=int, default=500,
        help='S in Eq.(1): max sampled entity pairs per rule for KBSample')
    parser.add_argument('--emvr_beta', type=float, default=0.7,
        help='beta in Eq.(5): weight on internal KB evidence vs external sim')
    parser.add_argument('--emvr_tau_min', type=float, default=0.1)
    parser.add_argument('--emvr_tau_max', type=float, default=0.5)
    parser.add_argument('--emvr_external_density_gate', type=float, default=0.001,
        help='only compute ExternalSim when density_r is below this')
    parser.add_argument('--emvr_conceptnet_patterns', type=str, default=None,
        help='optional path to a .json list of ConceptNet relation-chain '
             'pattern strings for ExternalSim; omit to disable external evidence')
    parser.add_argument('--use_emvr_warmstart', action='store_true',
        help='warm-start ReasonerModel.raw_weights from EV_i instead of zeros')
    parser.add_argument('--emvr_save_stats', action='store_true',
        help='save per-relation EMVR-KBC evidence rows + EV scores to reasoner_dir for later analysis')
    # --------------------------------------------------
    return parser.parse_args()

def setup_logging(verbose, logfile=None, printlog=False):
    log_level = logging.DEBUG if verbose else logging.INFO
    handlers = []
    if printlog:
        handlers.append(logging.StreamHandler(sys.stdout))
    elif logfile:
        touch_file(logfile)
        handlers.append(logging.FileHandler(logfile))
    formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=log_level,handlers=handlers)
    logger = logging.getLogger()
    logger.propagate = False


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(args, output_file):
    with open(output_file, 'w') as file:
        yaml.dump(vars(args), file)

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Directory created: {directory}")

def touch_file(file_path):
    if not os.path.exists(file_path):
        try:
            with open(file_path, 'w') as file:
                file.write("") 
            print(f"File created: {file_path}")
        except Exception as e:
            print(f"Failed to create file: {e}")

def set_random_seed(random_seed=5):
    random.seed(random_seed)
    np.random.seed(random_seed)


def main(args):  
    print("\nhello!!!! and happy sunshine\n")
    # ---------------- CUDA auto-detect / CPU fallback (NEW) ----------------
    # Several downstream calls used to hardcode "cuda"/use_cuda=True regardless
    # of --run_reasoner_models_on_cpu, which crashed with
    # "AssertionError: Torch not compiled with CUDA enabled" on any machine
    # without a CUDA-enabled torch build. RUN_DEVICE/RUN_USE_CUDA below are
    # resolved once and reused everywhere a device was previously hardcoded.
    cuda_available = torch.cuda.is_available()
    if args.run_reasoner_models_on_cpu or not cuda_available:
        if not cuda_available and not args.run_reasoner_models_on_cpu:
            print("!! CUDA not available (torch.cuda.is_available()==False) — falling back to CPU. !!")
            print("   If you have an NVIDIA GPU and expected CUDA to work, your torch install is "
                  "likely the CPU-only build. Fix with (adjust cu### to your CUDA version):")
            print("     pip uninstall torch")
            print("     pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118")
        RUN_DEVICE = "cpu"
        RUN_USE_CUDA = False
    else:
        RUN_DEVICE = "cuda"
        RUN_USE_CUDA = True
    print("[EMVR-KBC] resolved compute device: {}".format(RUN_DEVICE))
    # -------------------------------------------------------------------------
    train_df, test_df, valid_df, id2relation_dict, id2entity_dict, all_relations, all_entities, all_relations_long_text = get_KG_data(args.dataset, verbose=True)
    entity2id_dict = _build_inverse_dict(id2entity_dict)
    relation2id_dict =  _build_inverse_dict(id2relation_dict)
    n_relations = len(all_relations)
    n_entities = len(all_entities)
    logging.info("finished loading KG data from disk")
    if args.debug:
        print("!! Running script in debug mode !!")
        debug_idx = [1,2,4,8,16]
        all_relations_ = [all_relations[idx] for idx in debug_idx]
        print("Use specific relations instead of all relations: ")
        for relation in all_relations_: print("{} ".format(relation),end="")
        print("")
    elif args.debug_relation_limit:
        # ---------------- debug quick-run (merged) ----------------
        all_relations_ = all_relations[:args.debug_relation_limit]
        print("!! debug_relation_limit={} set: running only the first {} relations !!".format(
            args.debug_relation_limit, len(all_relations_)))
        print("Use specific relations instead of all relations: ")
        for relation in all_relations_: print("{} ".format(relation),end="")
        print("")
        # ------------------------------------------------------------
    else:
        all_relations_ = all_relations
    print("\nlearning rules for the following relations:\n",all_relations_,"\n")
    subgraphs_dir = os.path.join(args.run_dir, "subgraphs")
    make_dir(subgraphs_dir)
    relation_wo_subgraphs = []
    if args.run_extractor:
        for relation in all_relations_:
            relation_wk_idx = relation2id_dict[relation]
            relation_wk_txt = relation
            relation_text = convert_kgtxt_to_natlang_plus(relation_wk_txt)
            subgraphs = get_nhop_closed_path(relation, train_df, args.max_hop, n_neighbors=args.max_neighbor, n_subgraphs=args.max_sample, random_seed=args.rand_seed,verbose=args.verbose,remove_outmost_open=True)
            if args.max_subgraph:
                max_subgraph = args.max_subgraph
            else:
                max_subgraph = args.max_sample
            if len(subgraphs) > max_subgraph:
                subgraphs = random.sample(subgraphs, max_subgraph)           
            if len(subgraphs) == 0:
                relation_wo_subgraphs.append(relation)
            fname = os.path.join(subgraphs_dir, "{}_train_subgraphs.csv".format(relation_wk_idx))
            save_subgraphs(subgraphs, fname)
            print("[{}] sampled {} subgraphs for relation {}={} from train set".format(time.strftime("%Y-%m-%d %H:%M"), len(subgraphs), relation_wk_idx, relation_wk_txt),flush=True)
        logging.info("finished extracting sample subgraphs for all relations in KG")
        print("the following relations are without subgraphs:", relation_wo_subgraphs)
        save_nested_list(relation_wo_subgraphs, os.path.join(subgraphs_dir, "relations_wo_subgraphs.txt"))
    else:
        if is_empty_dir(subgraphs_dir):
            print("You should have subgraphs extracted first!")
            logging.error("empty subgraph directory, should run the script with --run_extractor flag set to True")
            return 1
    proposed_dir = os.path.join(args.run_dir, "proposed")
    make_dir(proposed_dir)
    relation_wo_subgraphs = load_nested_list(os.path.join(subgraphs_dir, "relations_wo_subgraphs.txt"))
    if args.run_proposer:
        for relation in all_relations_:
            relation_wk_idx = relation2id_dict[relation]
            relation_wk_txt = relation
            relation_text = convert_kgtxt_to_natlang_plus(relation_wk_txt)
            if relation in relation_wo_subgraphs:
                print("no subgraphs for relation={}, skip".format(relation))
                continue
            rawrules_fname = os.path.join(proposed_dir, "{}_proposed.csv".format(relation_wk_idx))
            if file_exists(rawrules_fname):
                print("already proposed rules for relation {}. moving on to the next relation...".format(relation))
                continue
            print("[{}] prepare to propose rules for relation {}".format(time.strftime("%Y-%m-%d %H:%M"), relation))
            fname = os.path.join(subgraphs_dir, "{}_train_subgraphs.csv".format(relation_wk_idx))
            subgraphs = load_subgraphs(fname)               
            llm_inputs = convert_subgraphs_to_inputs(subgraphs)
            # cross-platform path parsing fix (merged): original used fname.split("/")[-1],
            # which breaks on Windows paths; os.path.basename works on both.
            save_pfx = os.path.splitext(os.path.basename(fname))[0].split("_")[0]
            save_dir = proposed_dir
            if args.debug:
                print("fname    :", fname)
                print("save_pfx :", save_pfx)
                print("save_dir :", save_dir)
            llm_inputs = llm_input_len_check(llm_inputs, max_char=args.llm_max_input_chars, verbose=True)
            print("[{}] removed {} / {} samples that exceed max input char count".format(time.strftime("%Y-%m-%d %H:%M"), len(subgraphs)-len(llm_inputs),len(subgraphs)),flush=True)
            proposed_rules = llm_propose_rule(llm_inputs, args.llm_name, save_dir, save_pfx, args.llm_api_key,save_pickle=False, verbose=args.verbose)
            print("[{}] raw logical rules proposed for relation {}".format(time.strftime("%Y-%m-%d %H:%M"), relation_wk_txt),flush=True)
        logging.info("finished proposing logical rules using {}".format(args.llm_name))
    else:
        if is_empty_dir(proposed_dir):
            print("You should have raw rules proposed first!")
            logging.error("empty proposed raw rule directory, should run the script with --run_proposer flag set to True")
            return 1
    if not args.run_reasoner:
        print("extracter/proposer finished. gnite...")
        return 0
    reasoner_dir = os.path.join(args.run_dir, "reasoner")
    make_dir(reasoner_dir)
    print("\n the followiong relations has no subgraphs:", relation_wo_subgraphs)
    relation_wo_logicrules = []
    for relation in all_relations_:
        if relation in relation_wo_subgraphs:
            print("no proposed rules for relation no {}, skip".format(relation))
            relation_wo_logicrules.append(relation)
            continue
        relation_wk_idx = relation2id_dict[relation]
        relation_wk_txt = relation
        relation_text = convert_kgtxt_to_natlang_plus(relation_wk_txt)
        proposed_fname = os.path.join(proposed_dir, "{}_proposed.csv".format(relation_wk_idx))
        all_rules_fname = os.path.join(reasoner_dir, "{}_rules_all.json".format(relation_wk_idx))
        checked_rules = prepare_rules(proposed_fname, relation_text, all_rules_fname,verbose=args.debug)
        if len(checked_rules) == 0:
            print("no quality rules for relation={}".format(relation))
            relation_wo_logicrules.append(relation)
        else:
            print("{} logical rules for relation {}={}".format(len(checked_rules), relation_wk_idx,relation))
    print("the following relations has no logic rules to learn: ", relation_wo_logicrules, "\n")
    logging.info("finish preparing logical rules for grounding")
    train_arr = encode_kg_to_arr(train_df, id2entity_dict, id2relation_dict)
    test_arr = encode_kg_to_arr(test_df, id2entity_dict, id2relation_dict)
    valid_arr = encode_kg_to_arr(valid_df, id2entity_dict, id2relation_dict)
    train_test_arr = encode_kg_to_arr(pd.concat([train_df, valid_df]), id2entity_dict, id2relation_dict)
    train_valid_arr = encode_kg_to_arr(pd.concat([train_df, valid_df]), id2entity_dict, id2relation_dict)
    logging.info("KG arrays loaded successfully, totalling {} entities and {} relations".format(n_entities,n_relations))
    if args.reason_with_relation_semantic:
        semsim_fname = os.path.join(reasoner_dir, "semantic_similarities.pt")
        if USE_LONG_REL_TEXT:
            phrases = [convert_kgtxt_to_natlang_plus(rel) for rel in all_relations_long_text]
        else:
            phrases = [convert_kgtxt_to_natlang_plus(rel) for rel in all_relations]
        if file_exists(semsim_fname):
            logging.info("already computed semantice similarities, read from disk")
            similarity_matrix = torch.load(semsim_fname)
        else:
            print("[{}] Compute semantic similarity using model={} with representation type={}".format(time.strftime("%Y-%m-%d %H:%M"), args.semantic_lm, args.semantic_representation))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model, tokenizer = get_model_and_tokenizer(args.semantic_lm, device)
            if args.semantic_lm.startswith("sentence-transformers/"):
                logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
            embeddings = [get_phrase_embedding(phrase,model, tokenizer, device, args.semantic_representation) for phrase in phrases]
            if args.set_semantic_threshold:
                similarity_matrix = custom_similarity(embeddings, args.semantic_threshold_value, do_clipping=True)
            else:
                similarity_matrix = custom_similarity(embeddings, None, do_clipping=True)
            del model, tokenizer
            if RUN_USE_CUDA:
                torch.cuda.empty_cache()
            torch.save(similarity_matrix, semsim_fname)
            assert np.array_equal(torch.load(semsim_fname), similarity_matrix)
        n_similar_pairs = find_similar_phrases(phrases, similarity_matrix, include_self=False, verbose=False)
        print("{} pairs of similar relation phrases out of {} relations".format(n_similar_pairs, len(phrases)))
    else:
        print("[{}] do rigid reasoning without relation semantic".format(time.strftime("%Y-%m-%d %H:%M")))
        similarity_matrix = np.eye(n_relations)
    logging.info("finish computing semantic similarity matrix")
    kge_model = get_KGE(args.dataset,use_cuda=RUN_USE_CUDA)
    train_triple = [tuple(row) for row in train_arr]
    test_triple = [tuple(row) for row in test_arr]
    valid_triple = [tuple(row) for row in valid_arr]
    all_true_triple = train_triple + valid_triple + test_triple
    logging.info("finish loading pretrained KGE")
    grounding_dir =  os.path.join(args.run_dir, "grounding")
    make_dir(grounding_dir)
    for relation in all_relations_:
        if relation in relation_wo_logicrules:
            print("no logic rules for relation={}, skip grounding".format(relation))
            continue
        relation_wk_idx = relation2id_dict[relation]
        relation_wk_txt = relation
        relation_text = convert_kgtxt_to_natlang_plus(relation_wk_txt)
        checked_rules_fname=os.path.join(args.run_dir, "reasoner", "{}_rules.json".format(relation_wk_idx)) 
        grnd_save_prefix = os.path.join(grounding_dir, "{}".format(relation_wk_idx))
        if args.debug:
            print(checked_rules_fname)
        if file_exists(checked_rules_fname):
            checked_rules = load_nested_list(checked_rules_fname)
            logging.info("load grounding results from disk")

            train_aligned_masks, train_chained_masks, train_ground_results,train_ground_rconds = load_grounding_results(grnd_save_prefix+"_train")
            test_aligned_masks, test_chained_masks, test_ground_results,test_ground_rconds = load_grounding_results(grnd_save_prefix+"_test")
            valid_aligned_masks, valid_chained_masks, _,_ = load_grounding_results(grnd_save_prefix+"_valid")
        else:
            all_rules_fname = os.path.join(reasoner_dir, "{}_rules_all.json".format(relation_wk_idx))
            checked_rules = load_nested_list(all_rules_fname)
            logging.info("load {} checked rules from disk for relation {}={}".format(len(checked_rules), relation_wk_idx, relation_text))
            train_sparse = convert_arr_to_sparse_coo(train_arr, n_entities, n_relations)

            # ---------------- EMVR-KBC: evidence verification (NEW) ----------------
            emvr_ev_by_text = {}
            if args.use_emvr and len(checked_rules) > 0:
                fwd_adj, bwd_adj = build_adjacency(train_arr)
                external_index = None
                if args.emvr_conceptnet_patterns:
                    with open(args.emvr_conceptnet_patterns) as f:
                        patterns = json.load(f)
                    external_index = ExternalEvidenceIndex(pattern_strings=patterns)
                checked_rules, ev_scores_for_warmstart, emvr_stats = verify_rules_for_relation(
                    checked_rules, relation2id_dict, train_arr, n_entities,
                    target_relation_id=relation_wk_idx, fwd=fwd_adj, bwd=bwd_adj,
                    sample_size=args.emvr_sample_size, beta=args.emvr_beta,
                    tau_min=args.emvr_tau_min, tau_max=args.emvr_tau_max,
                    external_density_gate=args.emvr_external_density_gate,
                    external_index=external_index, verbose=args.debug)
                emvr_ev_by_text = {row["rule_text"]: row["ev_i"] for row in emvr_stats["rows"] if row["verified"]}
                print("[EMVR-KBC] relation {}={}: {}/{} rules verified, "
                      "filtering_rate={:.1%}, tau_r={:.3f}".format(
                          relation_wk_idx, relation_text, emvr_stats["n_verified"],
                          emvr_stats["n_candidates"], emvr_stats["filtering_rate"],
                          emvr_stats["tau_r"]))
                if args.emvr_save_stats:
                    emvr_stats_fname = os.path.join(reasoner_dir, "{}_emvr_stats.json".format(relation_wk_idx))
                    with open(emvr_stats_fname, 'w') as f:
                        json.dump(emvr_stats, f)
                if len(checked_rules) == 0:
                    print("no rules survived EMVR-KBC verification for relation={}, skip".format(relation))
                    relation_wo_logicrules.append(relation)
                    continue
            # -------------------------------------------------------------------------

            train_aligned_masks, train_chained_masks, train_ground_results,train_ground_rconds = ground_rules_over_kg_semantic(train_sparse, checked_rules, relation2id_dict, similarity_matrix,verbose=False)
            checked_rules, train_aligned_masks, train_chained_masks, train_ground_results,train_ground_rconds = keep_good_rules(checked_rules, train_aligned_masks, train_chained_masks, train_ground_results,train_ground_rconds,verbose=args.debug, criteria=args.good_rule_criteria)
            if args.use_emvr and len(emvr_ev_by_text) > 0:
                # persisted regardless of emvr_save_stats: the reasoner-training loop
                # below reloads checked_rules from disk in a separate pass and needs
                # this file to align EV_i with the final post-keep_good_rules rule set
                # for warm-start init (Eq. 10).
                ev_scores_fname = os.path.join(reasoner_dir, "{}_ev_scores.json".format(relation_wk_idx))
                with open(ev_scores_fname, 'w') as f:
                    json.dump([emvr_ev_by_text.get(cr[0], 0.0) for cr in checked_rules], f)
            save_grounding_results(grnd_save_prefix+"_train", train_aligned_masks, train_chained_masks, train_ground_results, train_ground_rconds)
            logging.info("ground over train kg arr for relation {}={}".format(relation_wk_idx, relation_text))
            test_sparse = convert_arr_to_sparse_coo(train_test_arr, n_entities, n_relations)
            test_aligned_masks, test_chained_masks, test_ground_results,test_ground_rconds = ground_rules_over_kg_semantic(test_sparse, checked_rules, relation2id_dict, similarity_matrix,verbose=False)
            save_grounding_results(grnd_save_prefix+"_test", test_aligned_masks, test_chained_masks, test_ground_results,test_ground_rconds)
            logging.info("ground over train+test kg arr for relation {}={}".format(relation_wk_idx, relation_text))
            valid_sparse = convert_arr_to_sparse_coo(train_valid_arr, n_entities, n_relations)
            valid_aligned_masks, valid_chained_masks, valid_ground_results,valid_ground_rconds = ground_rules_over_kg_semantic(valid_sparse, checked_rules,relation2id_dict, similarity_matrix,verbose=False)
            save_grounding_results(grnd_save_prefix+"_valid", valid_aligned_masks, valid_chained_masks, valid_ground_results,valid_ground_rconds)
            logging.info("ground over train+valid kg arr for relation {}={}".format(relation_wk_idx, relation_text))
            save_nested_list(checked_rules, checked_rules_fname)
            logging.info("save learnable checked rules to disk for relation {}={}".format(relation_wk_idx, relation_text))
        print("[{}] ground {} rules for relation {}={}".format(time.strftime("%Y-%m-%d %H:%M"),len(checked_rules), relation_wk_idx, relation_text))
    if RUN_USE_CUDA:
        torch.cuda.empty_cache()
    del train_aligned_masks, train_chained_masks, train_ground_results, train_ground_rconds
    del test_aligned_masks, test_chained_masks, test_ground_results,test_ground_rconds
    del valid_aligned_masks, valid_chained_masks
    epsilon = 1e-32
    hit_k_vals = [1,3,10]
    relation_wo_learnable_logic = []
    relation_notin_test_set = []
    for relation in all_relations_:
        run_krea_modelling = True
        if relation in relation_wo_logicrules:
            print("no logic rules for relation={}".format(relation))
            relation_wo_learnable_logic.append(relation)
            run_krea_modelling = False
        else:       
            relation_wk_idx = relation2id_dict[relation]
            relation_wk_txt = relation
            relation_text = convert_kgtxt_to_natlang_plus(relation_wk_txt)
            checked_rules_fname=os.path.join(args.run_dir, "reasoner", "{}_rules.json".format(relation_wk_idx))
            grnd_save_prefix = os.path.join(grounding_dir, "{}".format(relation_wk_idx))
            train_scores_fname = os.path.join(reasoner_dir, "{}_train_score.pt".format(relation_wk_idx))
            valid_scores_fname = os.path.join(reasoner_dir, "{}_valid_score.pt".format(relation_wk_idx))
            learned_weights_fname = os.path.join(reasoner_dir, "{}_learned_weights.pt".format(relation_wk_idx))
            test_save_prefix = os.path.join(reasoner_dir, "{}_test".format(relation_wk_idx))
            checked_rules = load_nested_list(checked_rules_fname)
            n_rules = len(checked_rules)
            print("[{}] reasoning over {} rules + kge for relation={}".format(time.strftime("%Y-%m-%d %H:%M"), n_rules, relation_text))
            if file_exists(train_scores_fname) and file_exists(valid_scores_fname):
                print("reasoning already done")
                logging.info("relation {}={}: already computed rule quality scores, load from disk".format(relation_wk_idx, relation_text))
                trainset_scores = torch.load(train_scores_fname)
                validset_scores = torch.load(valid_scores_fname)
            else:
                logging.info("relation {}={}: compute trainset rule quality scores, keep triplets with at least one corresponding rule for training.".format(relation_wk_idx, relation_text))
                train_aligned_masks, train_chained_masks, _,_ = load_grounding_results(grnd_save_prefix+"_train")
                train_sparse = convert_arr_to_sparse_coo(train_arr, n_entities, n_relations)
                if args.compute_rule_scoring_on_cpu or not RUN_USE_CUDA:
                    rule_scoring_device="cpu"
                    rule_scoring_verbose=True
                else:
                    rule_scoring_device=RUN_DEVICE
                    rule_scoring_verbose=args.debug
                trainset_triplets, trainset_scores = get_triplets_and_scores(train_sparse, train_aligned_masks, train_chained_masks, relation_wk_idx, kge_model, all_true_triple, n_entities,n_relations, kge_bsize=args.kge_bsize, cpu_num=10, device=rule_scoring_device,dataset_name=args.dataset, verbose=rule_scoring_verbose)
                del train_aligned_masks,train_chained_masks, train_sparse
                if RUN_USE_CUDA:
                    torch.cuda.empty_cache()
                if trainset_triplets is None:
                    print("no logic rules for relation={} to train on".format(relation))
                    relation_wo_learnable_logic.append(relation)
                    run_krea_modelling = False
                if run_krea_modelling:
                    logging.info("relation {}={}: compute validset rule quality scores, keep triplets with at least one corresponding rule, keep triplets that are not included in trainset".format(relation_wk_idx, relation_text))
                    valid_aligned_masks, valid_chained_masks, _,_ = load_grounding_results(grnd_save_prefix+"_valid")
                    valid_sparse = convert_arr_to_sparse_coo(train_valid_arr, n_entities, n_relations)
                    if args.compute_rule_scoring_on_cpu or not RUN_USE_CUDA:
                        rule_scoring_device="cpu"
                        rule_scoring_verbose=True
                    else:
                        rule_scoring_device=RUN_DEVICE
                        rule_scoring_verbose=args.debug
                    validset_triplets, validset_scores = get_triplets_and_scores(valid_sparse, valid_aligned_masks, valid_chained_masks, relation_wk_idx, kge_model, all_true_triple, n_entities,n_relations, kge_bsize=args.kge_bsize, cpu_num=10, device=rule_scoring_device,dataset_name=args.dataset,verbose=rule_scoring_verbose)
                    del valid_sparse, valid_aligned_masks, valid_chained_masks
                    if RUN_USE_CUDA:
                        torch.cuda.empty_cache()
                    valid_only_ids = get_unique_eval_query_ids(trainset_triplets, validset_triplets)
                    if len(valid_only_ids) == 0:
                        print("no logic rules for relation={} valid, skip modelling".format(relation))
                        relation_wo_learnable_logic.append(relation)
                        run_krea_modelling = False
                    else:
                        validset_triplets=[validset_triplets[idx] for idx in valid_only_ids]
                        validset_scores = validset_scores[torch.tensor(valid_only_ids, dtype=torch.long)]
                        validset_scores = validset_scores.to(RUN_DEVICE if not args.run_reasoner_models_on_cpu else "cpu")

                        trainset_scores = trainset_scores.to(RUN_DEVICE if not args.run_reasoner_models_on_cpu else "cpu")
                        logging.info("relation {}={}: saving rule quality scores for train/valid to disk".format(relation_wk_idx, relation_text))
                        torch.save(trainset_scores, train_scores_fname)
                        torch.save(validset_scores, valid_scores_fname)
                        run_krea_modelling = True
            if run_krea_modelling:
                if file_exists(learned_weights_fname):
                    print("read learned rule weights from disk for relation {}={}".format(relation_wk_idx, relation_text))
                    n_rules = n_rules +1
                    if args.use_reasoner_plus:
                        learned = torch.load(learned_weights_fname)
                        learned_weights = learned["logical_weights"]
                        learned_alpha = learned["logical_alpha"]
                    else:
                        learned_weights = torch.load(learned_weights_fname)
                else:
                    print("init and train the reasoner")
                    n_rules = n_rules +1
                    if args.use_reasoner_plus:
                        model = ReasonerModelPlus(n_rules) 
                    else:
                        model = ReasonerModel(n_rules) 
                    if not args.run_reasoner_models_on_cpu and RUN_USE_CUDA:
                        model.to("cuda")

                    # ---------------- EMVR-KBC: warm-start init (NEW) ----------------
                    if args.use_emvr and args.use_emvr_warmstart:
                        ev_scores_fname = os.path.join(reasoner_dir, "{}_ev_scores.json".format(relation_wk_idx))
                        if file_exists(ev_scores_fname):
                            with open(ev_scores_fname) as f:
                                aligned_ev = json.load(f)
                            # aligned_ev has one entry per logical rule (n_rules - 1);
                            # pad/truncate defensively in case checked_rules count
                            # drifted between the grounding pass and this pass.
                            n_logical = n_rules - 1
                            if len(aligned_ev) != n_logical:
                                aligned_ev = (aligned_ev + [0.0] * n_logical)[:n_logical]
                            model = apply_warmstart(model, aligned_ev)
                            print("[EMVR-KBC] warm-started {} logical-rule weights from EV_i "
                                  "(loss function L1 unchanged)".format(n_logical))
                        else:
                            print("[EMVR-KBC] no saved EV scores found for relation {}, "
                                  "falling back to random init".format(relation_wk_idx))
                    # -------------------------------------------------------------------

                    optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
                    best_valid_loss = float('inf')
                    best_learned_weights = torch.softmax(model.raw_weights, dim=0)
                    if args.use_reasoner_plus:
                        best_learned_alpha = 0
                    no_improvement_counter = 0
                    for epoch in range(args.num_epochs):
                        model, valid_loss = train_loop(model, epoch, optimizer, scheduler, trainset_scores, validset_scores, epsilon=epsilon,verbose=True)
                        if valid_loss < best_valid_loss:
                            best_valid_loss = valid_loss
                            best_learned_weights = torch.softmax(model.raw_weights, dim=0)
                            no_improvement_counter = 0
                            if args.use_reasoner_plus:
                                best_learned_alpha = torch.sigmoid(model.raw_alpha)
                        else:
                            no_improvement_counter += 1
                        if no_improvement_counter >= args.early_stop_patience:
                            print("Early stopping as validation loss did not improve for {} epochs.".format(args.early_stop_patience))
                            break
                    print("[{}] Model trained for {} epoches".format(time.strftime("%Y-%m-%d %H:%M"), epoch))                    
                    learned_weights = best_learned_weights
                    torch.save(learned_weights, learned_weights_fname)
                    del model, optimizer, scheduler, trainset_scores, validset_scores
                    if args.use_reasoner_plus:
                        learned_alpha = best_learned_alpha
                        torch.save({"logical_weights":learned_weights, "logical_alpha":learned_alpha}, learned_weights_fname)
                if args.use_reasoner_plus:
                    print("Learned alpha: {}".format(learned_alpha))
                print("Learned weights wj: {}".format(learned_weights))
                if file_exists(test_save_prefix+"_ranks.pt"): 
                    ranks = torch.load(test_save_prefix+"_ranks.pt")
                    print("[{}] loaded inference result on test set query with relation {}={} from disk".format(time.strftime("%Y-%m-%d %H:%M"), relation_wk_idx, relation_text))
                else:
                    eval_quries = test_arr[test_arr[:, 1]==relation_wk_idx]
                    eval_labels = eval_quries[:, 2]
                    eval_cmasks = load_grounding_results(grnd_save_prefix+"_test")[1]
                    logging.info("load test set info from disk for relation {}={}".format(relation_wk_idx, relation_text))
                    print("[{}] perform inference on test set query with relation {}={}".format(time.strftime("%Y-%m-%d %H:%M"), relation_wk_idx, relation_text))
                    if len(eval_labels) == 0:
                        relation_notin_test_set.append(relation)
                        print("[{}] relation=({}){} did not appeared in test set".format(time.strftime("%Y-%m-%d %H:%M"), relation_wk_idx, relation))
                        continue
                    if args.use_reasoner_plus:
                        eval_preds = test_loop_plus(eval_quries, learned_weights, learned_alpha, n_rules, n_entities, n_relations, all_true_triple, eval_cmasks, kge_model, args.kge_bsize, cpu_num=10, dataset_name=args.dataset,device=RUN_DEVICE)
                    else:
                        eval_preds = test_loop(eval_quries, learned_weights, n_rules, n_entities, n_relations, all_true_triple, eval_cmasks, kge_model, args.kge_bsize, cpu_num=10, dataset_name=args.dataset, device=RUN_DEVICE)
                    logging.info("finish making test set predictions for relation {}={}".format(relation_wk_idx, relation_text))
                    argsort = torch.argsort(eval_preds, dim=1, descending=True) 
                    answers = argsort[:, 0]
                    ranks = (argsort == torch.Tensor(eval_labels).unsqueeze(1).to(RUN_DEVICE)).nonzero(as_tuple=False)[:, 1] 
                    ranks = ranks + 1 
                    torch.save(eval_preds, test_save_prefix+"_predicts.pt")
                    torch.save(eval_quries, test_save_prefix+"_queries.pt")
                    torch.save(answers, test_save_prefix+"_answers.pt")
                    torch.save(ranks, test_save_prefix+"_ranks.pt")
                    logging.info("save test set predictions to disk for relation {}={}".format(relation_wk_idx, relation_text))
                mr, mrr, hit_ks = compute_metrics(ranks, k_vals=hit_k_vals)
                print("[{}] Relation=({}){}, N_Query={} | KREA | MR={:.4f}, MRR={:.4f}".format(time.strftime("%Y-%m-%d %H:%M"), relation_wk_idx, relation_text, ranks.shape[0], mr, mrr), end="")
                for i in range(len(hit_k_vals)):
                    print(", Hit@{}={:.4f}".format(hit_k_vals[i], hit_ks[i]), end="")
                print("")    
        if not run_krea_modelling: 
            relation_wk_idx = relation2id_dict[relation]
            relation_wk_txt = relation
            test_save_prefix = os.path.join(reasoner_dir, "{}_test".format(relation_wk_idx))
            eval_quries = test_arr[test_arr[:, 1]==relation_wk_idx]
            eval_labels = eval_quries[:, 2]
            if len(eval_labels) == 0:
                relation_notin_test_set.append(relation)
                continue
            eval_triple = [tuple(row) for row in eval_quries]
            eval_metrics, _, _, eval_preds, answers = kge_model.test_step(kge_model, eval_triple, all_true_triple,n_entities,n_relations,cpu_num=10,test_batch_size=args.kge_bsize, use_cuda=RUN_USE_CUDA,tail_only=True)
            argsort = torch.argsort(eval_preds, dim=1, descending=True)
            ranks = (argsort == torch.from_numpy(eval_quries)[:, 2].unsqueeze(1).to(argsort.device)).nonzero(as_tuple=False)[:, 1] + 1
            logging.info("finish making test set predictions for relation {}={}".format(relation_wk_idx, relation_text))
            torch.save(eval_preds, test_save_prefix+"_predicts.pt")
            torch.save(eval_quries, test_save_prefix+"_queries.pt")
            torch.save(answers, test_save_prefix+"_answers.pt")
            torch.save(ranks, test_save_prefix+"_ranks.pt")
            logging.info("save test set predictions to disk for relation {}={}".format(relation_wk_idx, relation_text))
            mr, mrr, hit_ks = compute_metrics(ranks, k_vals=hit_k_vals)
            print("[{}] Relation=({}){}, N_Query={} | KGE | MR={:.4f}, MRR={:.4f}".format(time.strftime("%Y-%m-%d %H:%M"), relation_wk_idx, relation_text, ranks.shape[0], mr, mrr), end="")
            for i in range(len(hit_k_vals)):
                print(", Hit@{}={:.4f}".format(hit_k_vals[i], hit_ks[i]), end="")
            print("")     
    print("")
    print("Compute eval metrics on the entire test set instead of per relation")
    all_ranks = []
    for relation in all_relations_:
        if relation in relation_notin_test_set:
            print("relation={} did not appeared in test set".format(relation))
            continue
        relation_wk_idx = relation2id_dict[relation]
        relation_wk_txt = relation
        relation_text = convert_kgtxt_to_natlang_plus(relation_wk_txt)
        test_save_prefix = os.path.join(reasoner_dir, "{}_test".format(relation_wk_idx))
        ranks = torch.load(test_save_prefix+"_ranks.pt")
        logging.info("load test set query ansering ranks for relation {}={}".format(relation_wk_idx, relation_text))
        all_ranks.append(ranks)
    if all_ranks:
        ranks = torch.cat(all_ranks)
        mr, mrr, hit_ks = compute_metrics(ranks, k_vals=hit_k_vals)
        print("[{}] N_Query={} | KREA | MR={:.4f}, MRR={:.4f}".format(time.strftime("%Y-%m-%d %H:%M"), ranks.shape[0], mr, mrr), end="")
        for i in range(len(hit_k_vals)):
            print(", Hit@{}={:.4f}".format(hit_k_vals[i], hit_ks[i]), end="")
        print("")
    else:
        print("no learnable logical rules for the given relations!")
    print("\nall done... sleep tight\n")
    return 0



if __name__ == "__main__":
    args = parse_args()
    if not args.run_dir:
        args.run_dir = os.path.join("runs", args.run_name)
    make_dir(args.run_dir)
    setup_logging(args.verbose, os.path.join("runs", args.run_name, "logging.txt"))
    set_random_seed(args.rand_seed)
    logging.info(f"Setting pandas and numpy random seed to: {args.rand_seed}")
    if args.config_file:
        config = load_config(args.config_file)
        logging.info(f"Run configuration loaded from: {args.config_file}")
    save_config(args, os.path.join("runs", args.run_name, "config.yaml"))
    print(f"Command-line arguments: {vars(args)}")
    logging.info("Running main function")
    result = main(args)
    if result == 0:
        logging.info("Python script finished successfully")
    else:
        logging.error("Python script encountered errors")
    sys.exit(result)