from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader



import numpy as np
import torch

from torch.utils.data import Dataset

# ================== KGE Model =======================

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        raise Exception("One should not train KGE during KREA")

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, nentity, nrelation, cpu_num, test_batch_size, use_cuda=True, tail_only=True):
        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()
        
        #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        #Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                nentity, 
                nrelation, 
                'head-batch'
            ), 
            batch_size=test_batch_size,
            num_workers=max(1, cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                nentity, 
                nrelation, 
                'tail-batch'
            ), 
            batch_size=test_batch_size,
            num_workers=max(1, cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )
        
        if tail_only:
            test_dataset_list = [test_dataloader_tail]
        else:
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        
        logs = []
        head_scores = []
        tail_scores = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    use_cuda = use_cuda and torch.cuda.is_available()

                    if use_cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()
                    # if use_cuda:
                    #     positive_sample = positive_sample.cuda()
                    #     negative_sample = negative_sample.cuda()
                    #     filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score = model((positive_sample, negative_sample), mode)
                    score += filter_bias


                    #Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim = 1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                        head_scores.append(score)
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                        tail_scores.append(score)
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        #Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        #ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0/ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        tail_scores = torch.cat(tail_scores, dim=0)
        tail_preds= torch.argmax(tail_scores, dim=1)

        if tail_only:
            head_scores = None
            head_preds = None
        else:
            head_scores = torch.cat(head_scores, dim=0)
            head_preds= torch.argmax(head_scores, dim=1)

        return metrics, head_scores, head_preds, tail_scores, tail_preds


# ================================ KGE Dataloader ===========
class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

    
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


# ======================= Above code adopt from RotatE with modification ==================

import os
import json

def _load_rotate_from_npy(kge_dir, use_cuda, fallback_nentity, fallback_nrelation,
                           fallback_hidden_dim, fallback_gamma):
    """
    Load a RotatE model from the original RotatE-authors' / RNNLogic save format:
        config.json, entity_embedding.npy, relation_embedding.npy
    (no torch 'checkpoint' file, no model_state_dict — just raw numpy arrays plus
    the training config that produced them). This is a real, distinct, valid
    serialization format used by that release — not a substitute checkpoint.

    config.json is the dumped argparse Namespace from that repo's training script,
    so nentity/nrelation/hidden_dim/gamma/double_*_embedding are read from it
    directly whenever present, since those are the exact values the embeddings
    were trained with. The per-dataset fallback_* args are only used for any key
    genuinely missing from an older/trimmed config.json.
    """
    config_path = os.path.join(kge_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    model_name = config.get("model", "RotatE")
    nentity = config.get("nentity", fallback_nentity)
    nrelation = config.get("nrelation", fallback_nrelation)
    hidden_dim = config.get("hidden_dim", fallback_hidden_dim)
    gamma = config.get("gamma", fallback_gamma)
    double_entity_embedding = config.get("double_entity_embedding", True)
    double_relation_embedding = config.get("double_relation_embedding", False)

    kge_model = KGEModel(
        model_name=model_name,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=hidden_dim,
        gamma=gamma,
        double_entity_embedding=double_entity_embedding,
        double_relation_embedding=double_relation_embedding
    )

    entity_embedding = np.load(os.path.join(kge_dir, "entity_embedding.npy"))
    relation_embedding = np.load(os.path.join(kge_dir, "relation_embedding.npy"))

    if tuple(entity_embedding.shape) != tuple(kge_model.entity_embedding.shape):
        raise ValueError(
            "entity_embedding.npy shape {} does not match model shape {} built from "
            "config.json (nentity={}, hidden_dim={}, double_entity_embedding={}). "
            "The .npy file and config.json in {} may not belong together.".format(
                entity_embedding.shape, tuple(kge_model.entity_embedding.shape),
                nentity, hidden_dim, double_entity_embedding, kge_dir))
    if tuple(relation_embedding.shape) != tuple(kge_model.relation_embedding.shape):
        raise ValueError(
            "relation_embedding.npy shape {} does not match model shape {} built from "
            "config.json (nrelation={}, hidden_dim={}, double_relation_embedding={}). "
            "The .npy file and config.json in {} may not belong together.".format(
                relation_embedding.shape, tuple(kge_model.relation_embedding.shape),
                nrelation, hidden_dim, double_relation_embedding, kge_dir))

    with torch.no_grad():
        kge_model.entity_embedding.copy_(torch.from_numpy(entity_embedding))
        kge_model.relation_embedding.copy_(torch.from_numpy(relation_embedding))

    if use_cuda:
        kge_model = kge_model.cuda()
    return kge_model


def get_KGE(dataset, use_cuda=True):
    if dataset == "UMLs":
        kge_dir = "RNNLogic/data/umls/RotatE_1000"
        nentity = 135
        nrelation = 46
        gamma = 6.0
        hidden_dim = 1000

    elif dataset == "WN18RR":
        kge_dir = "RNNLogic/data/wn18rr/RotatE_500"
        nentity = 40943
        nrelation = 11
        gamma = 6.0
        hidden_dim = 500

    elif dataset == "FB15K":
        kge_dir = "RNNLogic/data/FB15K237/RotatE_1000"
        nentity = 14541
        nrelation = 237
        gamma = 9.0
        hidden_dim = 1000

    elif dataset == "WD15K":
        kge_dir = "RNNLogic/data/wd15k/RotatE_1000"
        nentity = 15812
        nrelation = 179
        gamma = 9.0
        hidden_dim = 1000

    elif dataset == "ConceptNet":
        kge_dir = "RNNLogic/data/cn100/RotatE_1000"
        nentity = 78339
        nrelation = 34
        gamma = 9.0
        hidden_dim = 1000
    else:
        raise Exception("invalid dataset option {}".format(dataset))

    ckpt_path = os.path.join(kge_dir, "checkpoint")
    npy_path = os.path.join(kge_dir, "entity_embedding.npy")

    # ---------------- format auto-detect (NEW) ----------------
    if os.path.exists(ckpt_path):
        # original torch-checkpoint format, behaviour unchanged
        model_name = "RotatE"
        double_entity_embedding = True
        double_relation_embedding = False
        kge_model = KGEModel(
                model_name=model_name,
                nentity=nentity,
                nrelation=nrelation,
                hidden_dim=hidden_dim,
                gamma=gamma,
                double_entity_embedding=double_entity_embedding,
                double_relation_embedding=double_relation_embedding
            )
        if use_cuda:
            kge_model = kge_model.cuda()
        checkpoint = torch.load(ckpt_path)
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        return kge_model
    elif os.path.exists(npy_path):
        # RNNLogic / original-RotatE-authors save format: config.json +
        # entity_embedding.npy + relation_embedding.npy, no checkpoint file.
        print("[kge.py] no 'checkpoint' file found in {}; found entity_embedding.npy "
              "instead — loading RotatE from the .npy/config.json format.".format(kge_dir))
        return _load_rotate_from_npy(kge_dir, use_cuda, nentity, nrelation, hidden_dim, gamma)
    else:
        raise Exception(
            "no usable RotatE artifacts found in '{}'. Expected either a torch "
            "'checkpoint' file (with a 'model_state_dict' key), or the "
            "'entity_embedding.npy' + 'relation_embedding.npy' + 'config.json' "
            "trio (RNNLogic / original RotatE release format). Found neither.".format(kge_dir)
        )
    # ------------------------------------------------------------


def kge_inference(
    kge_model,
    eval_arr,
    all_true_triple,
    nentity,
    nrelation,
    cpu_num,
    test_batch_size,
    use_cuda=True,
    tail_only=True,
):
    use_cuda = use_cuda and torch.cuda.is_available()

    eval_triple = [tuple(row) for row in eval_arr]

    metrics, head_scores, head_preds, tail_scores, tail_preds = (
        kge_model.test_step(
            kge_model,
            eval_triple,
            all_true_triple,
            nentity,
            nrelation,
            cpu_num,
            test_batch_size,
            use_cuda=use_cuda,
            tail_only=tail_only,
        )
    )

    return metrics, head_scores, head_preds, tail_scores, tail_preds