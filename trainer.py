from collections import defaultdict
import time
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score as ari
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn import svm
from math import ceil
import torch
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from nltk.corpus import stopwords
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertConfig, BertModel, AutoModel, AutoTokenizer
import numpy as np
import torch.nn.functional as F
import os
import shutil
import sys
import random
from tqdm import tqdm
from model import TCDWAModel, Full_layer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from metrics import cluster_accuracy
from isda import EstimatorCV
import nlpaug.augmenter.word as naw
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

device_ids = [0,1]
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print('device: '.format(device))

class TCDWATrainer(object):
    def __init__(self, args):
        self.args = args
        self.max_len = args.max_len
        self.dataset_dir = args.dataset_dir
        self.embedding_file = args.embedding_file
        self.dataset_name = args.dataset_name
        self.bow_data_file = args.bow_data_file
        self.num_cpus = min(10, cpu_count() - 1) if cpu_count() > 1 else 1
        self.world_size = args.gpus
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.shuffle_attentions = args.shuffle_attentions
        self.attention_threshold = args.attention_threshold
        self.pretrained_lm = 'bert-base-uncased'
        # self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_lm)
        # self.pretrained_lm = args.model_name_or_path
        # self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_lm, do_lower_case=True)
        # self.pretrained_lm = "unsup-simcse-bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_lm)
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        self.sep_id = self.vocab['[SEP]']
        self.cls_id = self.vocab['[CLS]']
        print('[CLS]_id is: {} and [SEP]_id is: {}'.format(self.cls_id, self.sep_id))
        self.mask_id = self.vocab[self.tokenizer.mask_token]
        self.inv_vocab = {k: v for v, k in self.vocab.items()}
        # self.read_label_names(args.dataset_dir, args.label_names_file)
        self.num_class = args.n_clusters
        self.is_dual = args.is_dual
        self.is_updated = args.is_updated
        self.beta = args.beta
        self.read_data(args.dataset_dir, args.train_file, args.train_label_file, args.train_data_file, args.test_file, args.test_label_file)
        # self.num_train_data = len(self.train_data["input_ids"])
        # self.model = LOTClassModel.from_pretrained(self.pretrained_lm,
        #                                            output_attentions=False,
        #                                            output_hidden_states=False,
        #                                            num_labels=self.num_class,
        #                                            num_train_data=self.num_train_data)

        # self.config.nu
        # m_train_data = self.num_train_data
        self.model = TCDWAModel(self.pretrained_lm, self.num_class, self.is_dual, self.beta)
        # self.model = TCDWAModel.from_pretrained(self.pretrained_lm)
        # self.config = AutoConfig.from_pretrained(self.pretrained_lm, output_attentions=False,
        #                                          output_hidden_states=False, num_labels=self.num_class)
        # self.bert = AutoModelForMaskedLM.from_pretrained(self.pretrained_lm, config=self.config)
        # self.model = TCDWAModel(config=self.config, bert=self.bert)
        self.classifier = Full_layer(768, self.num_class)
        # self.estimator = EstimatorCV(768, self.num_class)
        self.with_test_label = True if args.test_label_file is not None else False
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        # self.Hessian = torch.autograd.functional.hessian
        self.lamda = args.lamda
        self.early_stop = args.early_stop

    # set up distributed training
    def set_up_dist(self, model, classifier):
        # create local model
        model = torch.nn.DataParallel(model).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()

        return model, classifier

    def create_stop_ids(self, vocab):
        stop_words = stopwords.words('english')
        self.stop_words_ids = [vocab[stop_word] for stop_word in stop_words]

    # get document truncation statistics with the defined max length
    def corpus_trunc_stats(self, docs):
        doc_len = []
        for doc in docs:
            input_ids = self.tokenizer.encode(doc, add_special_tokens=True)
            doc_len.append(len(input_ids))
        print(f"Document max length: {np.max(doc_len)}, avg length: {np.mean(doc_len)}, std length: {np.std(doc_len)}")
        trunc_frac = np.sum(np.array(doc_len) > self.max_len) / len(doc_len)
        print(f"Truncated fraction of all documents: {trunc_frac}")

    # convert a list of strings to token ids
    def encode(self, docs):
        encoded_dict = self.tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=self.max_len,
                                                        padding='max_length',
                                                        return_attention_mask=True, truncation=True,
                                                        return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return input_ids, attention_masks

    # convert list of token ids to list of strings
    def decode(self, ids):
        strings = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return strings

    # find label name indices and replace out-of-vocab label names with [MASK]
    def label_name_in_doc(self, doc):
        doc = self.tokenizer.tokenize(doc)
        label_idx = -1 * torch.ones(self.max_len, dtype=torch.long)
        new_doc = []
        wordpcs = []
        idx = 1  # index starts at 1 due to [CLS] token
        for i, wordpc in enumerate(doc):
            wordpcs.append(wordpc[2:] if wordpc.startswith("##") else wordpc)
            if idx >= self.max_len - 1:  # last index will be [SEP] token
                break
            if i == len(doc) - 1 or not doc[i + 1].startswith("##"):
                word = ''.join(wordpcs)
                # self.label2class: {replaceable label name: category_index}
                # contain all the replaceable label names that don't appear in multiple categories(4 for agnews)
                if word in self.label2class:
                    label_idx[idx] = self.label2class[word]
                    # replace label names that are not in tokenizer's vocabulary with the [MASK] token
                    if word not in self.vocab:
                        wordpcs = [self.tokenizer.mask_token]
                new_word = ''.join(wordpcs)
                if new_word != self.tokenizer.unk_token:
                    idx += len(wordpcs)
                    new_doc.append(new_word)
                wordpcs = []
        if (label_idx >= 0).any():
            return ' '.join(new_doc), label_idx
        else:
            return None

    # find label name occurrences in the corpus
    def label_name_occurrence(self, docs):
        text_with_label = []
        label_name_idx = []
        for doc in docs:
            # find label name indices and replace out-of-vocab label names with [MASK]
            result = self.label_name_in_doc(doc)
            if result is not None:
                text_with_label.append(result[0])
                label_name_idx.append(result[1].unsqueeze(0))
        if len(text_with_label) > 0:
            encoded_dict = self.tokenizer.batch_encode_plus(text_with_label, add_special_tokens=True,
                                                            max_length=self.max_len,
                                                            padding='max_length', return_attention_mask=True,
                                                            truncation=True, return_tensors='pt')
            input_ids_with_label_name = encoded_dict['input_ids']
            attention_masks_with_label_name = encoded_dict['attention_mask']
            label_name_idx = torch.cat(label_name_idx, dim=0)
        else:
            input_ids_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            attention_masks_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            label_name_idx = torch.ones(0, self.max_len, dtype=torch.long)
        return input_ids_with_label_name, attention_masks_with_label_name, label_name_idx

    # convert dataset into tensors
    def create_dataset(self, dataset_dir, text_file, label_file, loader_name, type="train", find_label_name=False,
                       label_name_loader_name=None):
        loader_file = os.path.join(dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"Loading encoded texts from {loader_file}")
            data = torch.load(loader_file)
        else:
            print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
            corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
            docs = [doc.strip() for doc in corpus.readlines()]
            # docs.extend([doc.strip() for doc in corpus.readlines()[20000:25000]])
            print('{} docs for {}'.format(len(docs), type))
            print(f"Converting texts into tensors.")
            chunk_size = ceil(len(docs) / self.num_cpus)
            chunks = [docs[x:x + chunk_size] for x in range(0, len(docs), chunk_size)]
            results = Parallel(n_jobs=self.num_cpus)(delayed(self.encode)(docs=chunk) for chunk in chunks)
            input_ids = torch.cat([result[0] for result in results])
            attention_masks = torch.cat([result[1] for result in results])
            print(f"Saving encoded texts into {loader_file}")
            if label_file is not None:
                print(f"Reading labels from {os.path.join(dataset_dir, label_file)}")
                truth = open(os.path.join(dataset_dir, label_file))
                labels = [int(label.strip()) for label in truth.readlines()]
                # labels.extend([int(label.strip()) for label in truth.readlines()[20000:25000]])
                print('{} labels for {}'.format(len(labels), type))
                labels = torch.tensor(labels)
                data = {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}
            else:
                data = {"input_ids": input_ids, "attention_masks": attention_masks}
            torch.save(data, loader_file)
        if find_label_name:
            loader_file = os.path.join(dataset_dir, label_name_loader_name)
            if os.path.exists(loader_file):
                print(f"Loading texts with label names from {loader_file}")
                label_name_data = torch.load(loader_file)
            else:
                print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
                corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
                docs = [doc.strip() for doc in corpus.readlines()]
                # docs.extend([doc.strip() for doc in corpus.readlines()[20000:25000]])
                print("Locating label names in the corpus.")
                chunk_size = ceil(len(docs) / self.num_cpus)
                chunks = [docs[x:x + chunk_size] for x in range(0, len(docs), chunk_size)]
                results = Parallel(n_jobs=self.num_cpus)(
                    delayed(self.label_name_occurrence)(docs=chunk) for chunk in chunks)
                input_ids_with_label_name = torch.cat([result[0] for result in results])
                attention_masks_with_label_name = torch.cat([result[1] for result in results])
                label_name_idx = torch.cat([result[2] for result in results])
                assert len(input_ids_with_label_name) > 0, "No label names appear in corpus!"
                label_name_data = {"input_ids": input_ids_with_label_name,
                                   "attention_masks": attention_masks_with_label_name, "labels": label_name_idx}
                loader_file = os.path.join(dataset_dir, label_name_loader_name)
                print(f"Saving texts with label names into {loader_file}")
                torch.save(label_name_data, loader_file)
            return data, label_name_data
        else:
            return data

    # convert dataset into tensors
    def create_csv_dataset(self, dataset_dir, data_file, loader_name):
        loader_file = os.path.join(dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"Loading encoded texts from {loader_file}")
            data = torch.load(loader_file)
        else:
            print(f"Reading data from {os.path.join(dataset_dir, data_file)}")
            train_df = pd.read_csv(os.path.join(dataset_dir, data_file), header=None)
            docs = [doc for doc in train_df[1]]
            print('{} docs'.format(len(docs)))

            print(f"Converting texts into tensors.")
            chunk_size = ceil(len(docs) / self.num_cpus)
            chunks = [docs[x:x + chunk_size] for x in range(0, len(docs), chunk_size)]
            results = Parallel(n_jobs=self.num_cpus)(delayed(self.encode)(docs=chunk) for chunk in chunks)
            input_ids = torch.cat([result[0] for result in results])
            attention_masks = torch.cat([result[1] for result in results])
            print(f"Saving encoded texts into {loader_file}")

            labels = [int(label) - 1 for label in train_df[0]]
            print('{} labels'.format(len(labels)))
            labels = torch.tensor(labels)

            data = {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}
            torch.save(data, loader_file)

        return data

    # create augmented data
    def create_augmented_dataset(self, dataset_dir, text_file, label_file, augmented_loader_name, aug_type="bert"):
        aug_loader_file = os.path.join(dataset_dir, augmented_loader_name)
        if os.path.exists(aug_loader_file):
            print(f"Loading encoded aug texts from {aug_loader_file}")
            aug_data = torch.load(aug_loader_file)
        else:
            print(f"Reading original texts from {os.path.join(dataset_dir, text_file)}")
            corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
            original_docs = [doc.strip() for doc in corpus.readlines()]
            if aug_type == 'bert':
                aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
            elif aug_type == 'roberta':
                aug = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute")
            else:
                print('No such aug type: {}'.format(aug_type))
            augmented_docs = []
            for doc_index in tqdm(range(len(original_docs))):
                augmented_doc = aug.augment(original_docs[doc_index])
                augmented_docs.append(augmented_doc)
            assert len(augmented_docs) == len(original_docs)
            print('Gain {} augmented docs by aug_{}'.format(len(augmented_docs), aug_type))

            print(f"Converting augmented texts into tensors.")
            chunk_size = ceil(len(augmented_docs) / self.num_cpus)
            chunks = [augmented_docs[x:x + chunk_size] for x in range(0, len(augmented_docs), chunk_size)]
            results = Parallel(n_jobs=self.num_cpus)(delayed(self.encode)(docs=chunk) for chunk in chunks)
            aug_input_ids = torch.cat([result[0] for result in results])
            aug_attention_masks = torch.cat([result[1] for result in results])
            print(f"Saving encoded augmented texts into {aug_loader_file}")
            if label_file is not None:
                print(f"Reading labels from {os.path.join(dataset_dir, label_file)}")
                truth = open(os.path.join(dataset_dir, label_file))
                labels = [int(label.strip()) for label in truth.readlines()]
                # labels.extend([int(label.strip()) for label in truth.readlines()[20000:25000]])
                print('{} labels for {}'.format(len(labels), aug_type))
                labels = torch.tensor(labels)
                aug_data = {"aug_input_ids": aug_input_ids, "aug_attention_masks": aug_attention_masks,
                            "labels": labels}
            else:
                aug_data = {"aug_input_ids": aug_input_ids, "aug_attention_masks": aug_attention_masks}
            torch.save(aug_data, aug_loader_file)

        return aug_data

    
    # read text corpus and labels from files
    def read_data(self, dataset_dir, train_file, train_label_file, train_data_file, test_file, test_label_file, use_aug=False):
        self.train_data = self.create_dataset(dataset_dir, train_file, train_label_file, train_data_file, type="train")
        # self.train_data = self.create_csv_dataset(dataset_dir, data_file=train_file, loader_name=train_data_file)

        if use_aug:
            print('Getting augmented data by bert!')
            self.bert_augmented_data = self.create_augmented_dataset(dataset_dir, train_file, train_label_file,
                                                                     'bert_augmented_data.pt', aug_type='bert')
            print('Getting augmented data by roberta!')
            self.roberta_augmented_data = self.create_augmented_dataset(dataset_dir, train_file, train_label_file,
                                                                        'roberta_augmented_data.pt', aug_type='roberta')
        # if test_file is not None:
        #     self.test_data = self.create_dataset(dataset_dir, test_file, test_label_file, "test.pt", type="test")

    # read label names from file
    def read_label_names(self, dataset_dir, label_name_file):
        label_name_file = open(os.path.join(dataset_dir, label_name_file))
        label_names = label_name_file.readlines()
        self.label_name_dict = {i: [word.lower() for word in category_words.strip().split()] for i, category_words in
                                enumerate(label_names)}
        print(f"Label names used for each class are: {self.label_name_dict}")
        # {replaceable label name: category_index}:
        # contain all the replaceable label names that don't appear in multiple categories(4 for agnews)
        self.label2class = {}
        # []:  all replaceable label names contained in self.vocab
        # []:  all replaceable label name ids contained in self.vocab
        self.all_label_name_ids = [self.mask_id]
        self.all_label_names = [self.tokenizer.mask_token]
        for class_idx in self.label_name_dict:
            for word in self.label_name_dict[class_idx]:
                assert word not in self.label2class, f"\"{word}\" used as the label name by multiple classes!"
                self.label2class[word] = class_idx
                if word in self.vocab:
                    self.all_label_name_ids.append(self.vocab[word])
                    self.all_label_names.append(word)

    # create dataset loader
    def make_dataloader(self, data_dict, batch_size):
        if "q" in data_dict:
            dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"], data_dict["labels"], data_dict["q"])
        else:
            dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"], data_dict["labels"])

        dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return dataset_loader

    def compute_cov_matrix(self, q, features):
        mu_list = []
        cov_matix_list = []
        class_count_list = []
        q_star = q.argmax(dim=-1)

        for i in range(self.num_class):
            class_mask = q_star == i
            features_for_each_class = features[class_mask]
            # (feature_dims)
            mu = torch.mean(features_for_each_class, dim=0)
            mean_features = features_for_each_class - mu
            # (feature_dim, feature_dim)
            cov_matrix = torch.mm(mean_features.t(), mean_features) / (features.shape[1] - 1)
            mu_list.append(mu)
            cov_matix_list.append(cov_matrix)
            class_count_list.append(features_for_each_class.shape[0])

        return mu_list, cov_matix_list, class_count_list

    def update_cov_matrix(self, batch_q, batch_features):

        batch_mu, batch_cov_matix, batch_class_count = self.compute_cov_matrix(batch_q, batch_features)

        for i in range(self.num_class):
            updated_mu = (self.class_count[i] * self.mu_list[i] +  batch_class_count[i] * batch_mu[i]) / \
                         (self.class_count[i] + batch_class_count[i])
            updated_cov_matrix = (self.class_count[i] * self.cov_matrix_list[i] +  batch_class_count[i] * batch_cov_matix[i]) \
                                 / (self.class_count[i] + batch_class_count[i]) \
                                 + self.class_count[i] * batch_class_count[i] * (self.mu_list[i] - batch_mu[i]) * (self.mu_list[i] - batch_mu[i]).t() \
                                 / (self.class_count[i] + batch_class_count[i]) ** 2

            self.mu_list[i] = updated_mu
            self.cov_matrix_list[i] = updated_cov_matrix
            self.class_count[i] = self.class_count[i] + batch_class_count[i]

    def initial_inference(self, model, dataset_loader, k, mix_layers_set, similarity_type):
        all_features = []
        all_true_labels = []
        all_selected_token_features = []
        all_selected_weights = []
        model.eval()
        for batch in tqdm(dataset_loader):
            layer_index = np.random.choice(mix_layers_set, 1)[0]
            with torch.no_grad():
                input_ids = batch[0].cuda()
                input_mask = batch[1].cuda()
                true_labels = batch[2]
                # (batch_size, mean_feature_dims)
                mean_features, features, selected_token_features, _, selected_weights, _ = model(input_ids,
                                                  similarity_type=similarity_type,
                                                  cls_id=self.cls_id,
                                                  sep_id=self.sep_id,
                                                  shuffle_attentions=self.shuffle_attentions,
                                                  attention_threshold=self.attention_threshold,
                                                  k=k,
                                                  layer_index=layer_index,
                                                  attention_mask=input_mask)
                all_features.append(mean_features)
                all_true_labels.append(true_labels)
                all_selected_token_features.append(selected_token_features)
                all_selected_weights.append(selected_weights)
                # (batch_size, cls_feature_dims)
                # cls_features = cls_features[:, 0, :])
        all_features = torch.cat(all_features, dim=0)
        all_true_labels = torch.cat(all_true_labels, dim=0)
        all_selected_token_features = torch.cat(all_selected_token_features, dim=0)
        all_selected_weights = torch.cat(all_selected_weights, dim=0)

        return all_features, all_true_labels, all_selected_token_features, all_selected_weights
    
    # initialize assigments by kmeans
    def initialize_self_train_assignments(self, model, n_clusters, k, mix_layers_set, similarity_type):
        print('Initialize assignments...')
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    
        train_dataset_loader = self.make_dataloader(self.train_data, self.train_batch_size)

        embedding_file_path = os.path.join(self.dataset_dir, self.embedding_file)
        if os.path.exists(embedding_file_path):
            print(f"Loading embeddings from {embedding_file_path}")
            all_train_true_labels = self.train_data['labels']
            all_train_features = np.genfromtxt(embedding_file_path)
        else:
            all_train_features, all_train_true_labels, _, _ = self.initial_inference(model, train_dataset_loader, k, mix_layers_set, similarity_type)
            all_train_features = all_train_features.cpu().numpy()
            np.savetxt(embedding_file_path, all_train_features, fmt='%f')
            print('Succeed in saving embeddings into {}!'.format(embedding_file_path))
       
        predictions = 1. / kmeans.fit_transform(all_train_features)
        all_train_assignments = np.argmax(predictions, axis=1)

        _, initial_acc = cluster_accuracy(all_train_true_labels.numpy(), all_train_assignments)
        nmi_score = nmi(all_train_true_labels.numpy(), all_train_assignments)
        ari_score = ari(all_train_true_labels.numpy(), all_train_assignments)
        print('Initial Accuracy: {}\tNMI: {}\tARI: {}'.format(initial_acc, nmi_score, ari_score))

        predictions = torch.Tensor(predictions)
        print(predictions)
        # row normalization
        predictions = F.normalize(predictions, p=1, dim=1)
        print(predictions)
        self.train_data['q'] = predictions

        return initial_acc, nmi_score, ari_score

    def show_selected_tokens(self, input_ids, selected_indices):
        mask = torch.zeros_like(input_ids, dtype=torch.bool).cuda()
        mask.scatter_(1, selected_indices, 1)
        selected_ids = torch.masked_select(input_ids, mask)
        selected_tokens = self.decode(selected_ids.cpu().numpy().tolist())

        return selected_tokens

    # def Hessian_loss(self, classifier, y):
    #     def Hessian_loss2(z):
    #         q = F.softmax(classifier(z.reshape(-1, 768)), dim=-1) 
    #         loss = F.kl_div(y, q, reduction='batchmean')
    #         return loss
    #
    #     return Hessian_loss2
    def Hession_loss(self, classifier, predictions, y):
        # dense_weight: (128, 768)
        # fc_weight: (num_class, 128)
        # predictions: (num_class)
        # y: (num_class)
        # dense_weight = classifier.module.dense.weight
        # fc_weight = classifier.module.fc.weight
        # dense_weight = model.module.dense.weight
        fc_weight = classifier.module.fc.weight
        temp = torch.mul(y, predictions)
        diag_matrix = torch.diag(temp)
        temp = diag_matrix - torch.mm(temp.reshape(-1,1), predictions.reshape(1,-1))
        # temp1 = torch.mm(dense_weight.t(), fc_weight.t())
        # temp2 = torch.mm(temp1, temp)
        # temp3 = torch.mm(temp2, fc_weight)
        # hessian_matrix = torch.mm(temp3, dense_weight)
        temp = torch.mm(fc_weight.t(), temp)
        hessian_matrix = torch.mm(temp, fc_weight)

        return hessian_matrix

    def compute_average_cov_matrix(self, preds, cov_matrixes):
        average_cov_matrix = torch.Tensor([0.]).cuda()
        for i in range(self.num_class):
            average_cov_matrix = average_cov_matrix + preds[i] * cov_matrixes[i]

        return average_cov_matrix
    
    def initialize_simcse_train_assignments(self, n_clusters, dataset_dir, embedding_file):
        print('Initialize simcse assignments...')
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        true_labels = self.train_data['labels'].numpy()

        print(f"Reading embeddings from {os.path.join(dataset_dir, embedding_file)}")
        initial_embeddings = np.genfromtxt(os.path.join(dataset_dir, embedding_file))

        predictions = 1. / kmeans.fit_transform(initial_embeddings)
        all_train_assignments = np.argmax(predictions, axis=1)
        _, initial_acc = cluster_accuracy(true_labels, all_train_assignments)
        nmi_score = nmi(true_labels, all_train_assignments)
        ari_score = ari(true_labels, all_train_assignments)
        print('Initial Accuracy: {}\tNMI: {}\tARI: {}'.format(initial_acc, nmi_score, ari_score))

        predictions = torch.Tensor(predictions)
        print(predictions)
        # row normalization
        predictions = F.normalize(predictions, p=1, dim=1)
        print(predictions)
        self.train_data['q'] = predictions

        return initial_acc, nmi_score, ari_score
        

    def batch_train(self, epochs, n_clusters, k, mix_layers_set, loader_name, similarity_type):
        model, classifier = self.set_up_dist(self.model, self.classifier)
        # best_p_acc, best_p_nmi, best_p_ari = self.initialize_self_train_assignments(model, n_clusters, k, mix_layers_set, similarity_type)
        best_p_acc, best_p_nmi, best_p_ari = self.initialize_simcse_train_assignments(n_clusters, self.dataset_dir, self.embedding_file)
        if self.is_dual:
            if self.is_updated:
                print('Performing initializing CV with q!')
                initial_train_dataset_loader = self.make_dataloader(self.train_data, self.train_batch_size)
                _, _, initial_selected_token_features, initial_selected_token_weights = self.initial_inference(model, initial_train_dataset_loader, k, mix_layers_set, similarity_type)
                model.module.estimator.update_CV(initial_selected_token_features, self.train_data['q'].cuda(), initial_selected_token_weights)
            else:
                print('Performing initializing CV with zero tensors!')
        # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, eps=1e-8)
        # total_steps = int(len(self.train_data["input_ids"]) * epochs / self.train_batch_size)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
        #                                             num_training_steps=total_steps)
        optimizer = AdamW([
         {
             "params": model.module.bert.parameters(),
             "lr": self.args.lrmain
         },
         # {
         #     "params": model.module.dense.parameters(),
         #     "lr": self.args.lrlast
         # },
         {
             "params": classifier.module.parameters(),
             "lr": self.args.lrlast
         },
     ])
        scheduler = None

        if self.early_stop:
            agree_count = 0

        results_row = []
        results_row.append([0, best_p_acc, best_p_nmi, best_p_ari])
        for epoch in range(epochs):
            # current_lamba = (epoch / epochs) * self.lamda
            if epoch < 2:
                current_lamba = 0.
            else:
                current_lamba = (epoch / epochs) * self.lamda
            epoch_train_loss = 0.
            epoch_train_kl_loss = 0.
            epoch_train_hessian_loss = 0.
            all_updated_q = []

            rand_idx = torch.randperm(len(self.train_data["input_ids"]))
            self.train_data = {"input_ids": self.train_data["input_ids"][rand_idx],
                               "attention_masks": self.train_data["attention_masks"][rand_idx],
                               "labels": self.train_data["labels"][rand_idx],
                               "q": self.train_data["q"][rand_idx]}
            print("\nStart self-training: epoch_{}/{}.".format(epoch, epochs))

            train_dataset_loader = self.make_dataloader(self.train_data, self.train_batch_size)
            model.train()
            classifier.train()
            for step, batch in enumerate(tqdm(train_dataset_loader)):
                batch_input_ids = batch[0].to(device)
                batch_input_mask = batch[1].to(device)
                q = batch[3].to(device)
                # mean_features: (batch_size, feature_dims)
                # selected_token_features: (batch_size, k+1, feature_dims)
                # logits: (batch_size, k+1, l)
                # selected_weights: (batch_size, k+1)
                layer_index = np.random.choice(mix_layers_set, 1)[0]
                mean_features, pooler_output, token_features, shuffled_token_features, token_weights, token_input_ids = model(batch_input_ids,
                                                                                                           similarity_type=similarity_type,
                                                                                                           cls_id=self.cls_id,
                                                                                                           sep_id=self.sep_id,
                                                                                                           shuffle_attentions=self.shuffle_attentions,
                                                                                                           attention_threshold=self.attention_threshold,
                                                                                                           k=k,
                                                                                                           layer_index=layer_index,
                                                                                                           token_type_ids=None,
                                                                                                           attention_mask=batch_input_mask)
                # selected_tokens = self.decode(selected_input_ids.reshape(-1).cpu().numpy().tolist())
                # tokens = self.decode(batch_input_ids.reshape(-1).cpu().numpy().tolist())
                token_weights = token_weights.detach()
                # print(tokens[:len(batch_input_ids[0])])
                # print(selected_tokens[:4])
                # print(selected_weights[0])
                if self.shuffle_attentions:
                    q = torch.cat((q, q), dim=0)
                    token_features = torch.cat((token_features, shuffled_token_features), dim=0)
                    token_weights = torch.cat((token_weights, token_weights), dim=0)
                
                logits = classifier(token_features)
                preds = nn.Softmax(dim=-1)(logits)
                # with torch.no_grad():
                #     # (train_data, l)
                #     predictions = torch.matmul(selected_weights.unsqueeze(1), preds).squeeze(1)
                #     predictions = predictions / torch.sum(selected_weights, dim=1).reshape(-1, 1)
                #     weight = predictions ** 2 / torch.sum(predictions, dim=0)
                #     updated_q = (weight.t() / torch.sum(weight, dim=1)).t()
                #     all_updated_q.append(updated_q)

                loss = 0.
                sum_kl_loss = 0.
                sum_hessian_loss = 0.
                if self.is_dual:
                    cov_matrixes = model.module.estimator.CoVariance.detach()
                    for i in range(preds.shape[0]):
                        y_star = q[i].argmax()
                        cov_matrix = cov_matrixes[y_star]
                        for j in range(preds.shape[1]):
                            kl_loss = self.kl_loss(preds[i][j].log().view(-1, self.num_class), q[i].view(-1, self.num_class))
                            # (feature_dim, feature_dim)
                            hessian_matrix = self.Hession_loss(classifier,preds[i][j].view(-1), q[i].view(-1))
                            hessian_loss = (current_lamba * torch.mm(hessian_matrix, cov_matrix)).trace() / 2
                            sum_kl_loss = sum_kl_loss + kl_loss
                            sum_hessian_loss = sum_hessian_loss + hessian_loss
                            one_loss = token_weights[i][j] * (kl_loss + hessian_loss)
                            loss = loss + one_loss
                else:
                    for i in range(preds.shape[0]):
                        for j in range(preds.shape[1]):
                            one_loss = token_weights[i][j] * self.kl_loss(preds[i][j].log().view(-1, self.num_class), q[i].view(-1, self.num_class))
                            loss = loss + one_loss

                epoch_train_loss += loss.item()
                epoch_train_hessian_loss += sum_hessian_loss
                epoch_train_kl_loss += sum_kl_loss

                optimizer.zero_grad()
                # 反向传播
                loss.backward()
                # 梯度裁剪，避免出现梯度爆炸情况
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # 更新参数
                optimizer.step()
                # 更新学习率
                # scheduler.step()
                # update mu and cov_matrix
                # self.update_cov_matrix(q, mean_features)

            # self.train_data["q"] = torch.cat(all_updated_q, dim=0).cpu()
            # 平均训练误差
            avg_train_loss = epoch_train_loss / len(train_dataset_loader)
            print(f"lr: {optimizer.param_groups[0]['lr']:.4g}")

            if self.is_dual:
                print('Use dual augmentation!')
                avg_train_kl_loss = epoch_train_kl_loss / len(train_dataset_loader)
                avg_train_hessian_loss = epoch_train_hessian_loss / len(train_dataset_loader)
                print("Average training loss: {}, kl_loss: {}, hessian_loss: {} in epoch_{}".
                      format(avg_train_loss, avg_train_kl_loss, avg_train_hessian_loss, epoch+1))
            else:
                print("Average training loss: {} in epoch_{}".format(avg_train_loss, epoch+1))

            if epoch % 1 == 0:
                print("Running Validation...")
                # predictions: (train_data, l) weights: (train_data, k)
                input_ids, input_mask, predictions, _, true_labels, selected_weights, _, selected_token_features = self.inference(model, classifier, train_dataset_loader, return_type="data", k=k, mix_layers_set=mix_layers_set, similarity_type=similarity_type)
                _, p_acc = cluster_accuracy(true_labels.cpu().numpy(), predictions.argmax(dim=-1).cpu().numpy())
                p_nmi = nmi(true_labels.cpu().numpy(), predictions.argmax(dim=-1).cpu().numpy())
                p_ari = ari(true_labels.cpu().numpy(), predictions.argmax(dim=-1).cpu().numpy())
                # current_acc = (predictions.argmax(dim=-1) == true_labels).int().sum().item() / len(targets)
                print('Epoch_{}/{}: p_acc = {}, p_nmi = {} and p_ari = {}'.format(epoch+1, epochs, p_acc, p_nmi, p_ari))

                weight = predictions ** 2 / torch.sum(predictions, dim=0)
                updated_q = (weight.t() / torch.sum(weight, dim=1)).t()
                if self.is_dual:
                    model.module.estimator.update_CV(selected_token_features, updated_q, selected_weights)
                self.train_data["q"] = updated_q.cpu()

                if p_acc > best_p_acc:
                    best_p_acc = p_acc
                    best_p_nmi = p_nmi
                    best_p_ari = p_ari

                results_row.append([epoch+1, p_acc, p_nmi, p_ari])

        results_row.append([epochs+1, best_p_acc, best_p_nmi, best_p_ari])

        return results_row

    # self training
    def self_train(self, epochs, n_clusters, k, mix_layers_set, loader_name, similarity_type):
        print(f"\nStart self-training.")
        return self.batch_train(epochs, n_clusters, k, mix_layers_set, loader_name, similarity_type)

    # use a model to do inference on a dataloader
    def inference(self, model, classifier, dataset_loader, return_type, k, mix_layers_set, similarity_type):
        if return_type == "data":
            all_input_ids = []
            all_input_mask = []
            all_preds = []
            all_mean_features = []
            all_labels = []
            all_weights = []
            all_selected_input_ids = []
            all_selected_token_features = []
        elif return_type == "acc":
            pred_labels = []
            truth_labels = []
        elif return_type == "pred":
            pred_labels = []
        model.eval()
        classifier.eval()
        for batch in tqdm(dataset_loader):
            with torch.no_grad():
                input_ids = batch[0].to(device)
                input_mask = batch[1].to(device)
                # mean_features: (batch_size, feature_dims)
                # selected_token_features: (batch_size, k+1, feature_dims)
                # logits: (batch_size, k+1, l)
                # selected_weights: (batch_size, k+1)
                # selected_input_ids: (batch_size, k+1)
                layer_index = np.random.choice(mix_layers_set, 1)[0]
                mean_features, pooler_output, selected_token_features, _, selected_weights, selected_input_ids = model(input_ids,
                                                                                                           similarity_type=similarity_type,
                                                                                                           cls_id=self.cls_id,
                                                                                                           sep_id=self.sep_id,
                                                                                                           shuffle_attentions=self.shuffle_attentions,
                                                                                                           attention_threshold=self.attention_threshold,
                                                                                                           k=k,
                                                                                                           layer_index=layer_index,
                                                                                                           token_type_ids=None,
                                                                                                           attention_mask=input_mask)
                logits = classifier(selected_token_features)
                preds = nn.Softmax(dim=-1)(logits)
                # (train_data, l)
                predictions = torch.matmul(selected_weights.unsqueeze(1), preds).squeeze(1)
                predictions = predictions / torch.sum(selected_weights, dim=1).reshape(-1, 1)
                # logits = classifier(mean_features)
                # logits = logits[:, 0, :]
                # (batch_size, cls_feature_dims)
                # cls_features = cls_features[:, 0, :]
                if return_type == "data":
                    labels = batch[2]
                    all_input_ids.append(input_ids)
                    all_input_mask.append(input_mask)
                    # all_preds.append(nn.Softmax(dim=-1)(logits))
                    all_preds.append(predictions)
                    all_mean_features.append(mean_features)
                    all_labels.append(labels)
                    all_weights.append(selected_weights)
                    all_selected_input_ids.append(selected_input_ids)
                    all_selected_token_features.append(selected_token_features)
                elif return_type == "acc":
                    labels = batch[2]
                    pred_labels.append(torch.argmax(logits, dim=-1).cpu())
                    truth_labels.append(labels)
                elif return_type == "pred":
                    pred_labels.append(torch.argmax(logits, dim=-1).cpu())
        if return_type == "data":
            all_input_ids = torch.cat(all_input_ids, dim=0)
            all_input_mask = torch.cat(all_input_mask, dim=0)
            all_preds = torch.cat(all_preds, dim=0)
            all_mean_features = torch.cat(all_mean_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_weights = torch.cat(all_weights, dim=0)
            all_selected_input_ids = torch.cat(all_selected_input_ids, dim=0)
            all_selected_token_features = torch.cat(all_selected_token_features, dim=0)
            return all_input_ids, all_input_mask, all_preds, all_mean_features, all_labels, all_weights, all_selected_input_ids, all_selected_token_features
        elif return_type == "acc":
            pred_labels = torch.cat(pred_labels, dim=0)
            truth_labels = torch.cat(truth_labels, dim=0)
            samples = len(truth_labels)
            # _, acc = cluster_accuracy(truth_labels.numpy(), pred_labels.numpy())
            acc = (pred_labels == truth_labels).float().sum() / samples
            return acc
        elif return_type == "pred":
            pred_labels = torch.cat(pred_labels, dim=0)
            return pred_labels