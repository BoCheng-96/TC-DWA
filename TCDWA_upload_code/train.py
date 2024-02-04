from trainer import TCDWATrainer
import argparse
import torch
import numpy as np
import random
import pandas as pd
import time

def main():
    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # agnews/imdb
    parser.add_argument('--dataset_name', default='dbpedia',
                        help='dataset name')
    parser.add_argument('--dataset_dir', default='datasets/dbpedia/',
                        help='dataset directory')
    parser.add_argument('--model_name_or_path', default='saves/dbpedia',
                        help='saved fine-tuned model')
    parser.add_argument('--label_names_file', default='label_names.txt',
                        help='file containing label names (under dataset directory)')
    parser.add_argument('--train_file', default='train_10000.txt',
                        help='unlabeled text corpus for training (under dataset directory); one document per line')
    parser.add_argument('--train_label_file', default='train_labels_10000.txt',
                        help='train corpus ground truth label; if provided, model will be evaluated during self-training')
    # bert_embeddings_10000.txt / initial_simcse_embeddings_10000.txt
    parser.add_argument('--embedding_file', default='initial_simcse_embeddings_10000.txt',
                        help='embedding file')
    parser.add_argument('--train_data_file', default='bert_train_10000.pt',
                        help='dict for dataset training: {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}')
    parser.add_argument('--bow_data_file', default="dbpedia_features_10000.npy",
                        help='bow features and labels for kmeans: {"bow_features": _ , "labels": _ }')
    parser.add_argument('--test_file', default='processed_test.txt',
                        help='test corpus to conduct model predictions (under dataset directory); one document per line')
    parser.add_argument('--test_label_file', default='test_labels.txt',
                        help='test corpus ground truth label; if provided, model will be evaluated during self-training')
    parser.add_argument('--final_model', default='cls_dual_final_model.pt',
                        help='the name of the final classification model to save to')
    parser.add_argument('--results_dir', default='bert_dbpedia_10000',
                        help='the name of the final classification model to save to')
    parser.add_argument('--lrmain', type=float, default=5e-6)
    parser.add_argument('--lrlast', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='batch size per GPU for evaluation; bigger batch size makes training faster')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help='batch size per GPU for training')
    parser.add_argument('--max_len', type=int, default=128,
                        help='length that documents are padded/truncated to')
    parser.add_argument('--self_train_epochs', type=int, default=5,
                        help='self training epochs; 1-5 usually is good depending on dataset size (smaller dataset needs more epochs)')
    parser.add_argument('--gpus', default=2, type=int,
                        help='number of gpus to use')
    parser.add_argument('--n_clusters', default=14, type=int,
                        help='number of classes in the dataset')
    parser.add_argument('--layer_index', default=12, type=int,
                        help='the selected hidden layer')
    parser.add_argument('--mix_layers_set', nargs='+', default=[7, 9, 12], type=int,
                        help='the selected layers set')
    parser.add_argument('--lamda', type=float, default=0.1,
                        help='the positive coefficient to control the strength of semantic data augmentation')
    parser.add_argument('--early_stop', action='store_true',
                        help='whether or not to enable early stop of self-training')
    parser.add_argument('--is_dual', type=bool, default=True,   
                        help='whether or not to enable dual training')
    parser.add_argument('--is_updated', type=bool, default=False,   
                        help='initialize CoVariance with initial q or zero tensor')
    parser.add_argument('--shuffle_attentions', type=bool, default=False,
                        help='whether or not to enable triple training')
    parser.add_argument('--attention_threshold', type=float, default=0.8,
                        help='keep attentions > attention_threshold to obtain shuffled_hidden_states')
    parser.add_argument('--similarity_type', type=str, default='norm',
                        help='the type to compute similarity')
    parser.add_argument('--times', default=5, type=int) 

    args = parser.parse_args()
    print(args)

    # 'single':use kl_loss
    # 'dual':kl_loss + hessian_loss
    if args.is_dual:
        dual_type = 'dual'
    else:
        dual_type = 'single'
   
    # 'non-zero': initialize CoVariance with (initial q generated by kmeans)
    # 'zero': initialize CoVariances with zero tensors
    if args.is_updated:
        EW_type = 'non-zero'
    else:
        EW_type = 'zero'
    
    if args.shuffle_attentions:
        triple_type = 'shuffled_' + str(args.attention_threshold) 
    else:
        triple_type = 'unshuffled'

    full_results = []
    for num in [0,1,2,3,4,5]:
        args.k = num
        for i in [2021,2022,2023,2024,2025]:
            print('Training times: {}/{}'.format(i, args.times))
            torch.random.manual_seed(i)
            torch.manual_seed(i)
            torch.cuda.manual_seed(i)
            random.seed(i)
            np.random.seed(i)
            trainer = TCDWATrainer(args)
            # results_row = trainer.self_train(
            #               epochs=args.self_train_epochs,
            #               n_clusters=args.n_clusters,
            #               k=args.k,
            #               layer_index=args.layer_index,
            #               loader_name=args.final_model,
            #               similarity_type=args.similarity_type,)
            results_row = trainer.self_train(
                epochs=args.self_train_epochs,
                n_clusters=args.n_clusters,
                k=args.k,
                mix_layers_set=args.mix_layers_set,
                loader_name=args.final_model,
                similarity_type=args.similarity_type, )
            results_row = np.array(results_row)
            full_results.append(results_row[-1, 1:])
            print('Training time_{}: p_ACC = {}, p_NMI = {}, p_ARI = {}'.format(
                i, results_row[-1, 1], results_row[-1, 2], results_row[-1, 3]))

            results_log_file = 'logs/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_train_results_{}.csv'.format(args.results_dir, args.similarity_type, dual_type, triple_type, EW_type, args.k, args.layer_index, args.lamda, args.self_train_epochs, args.beta, i)
            
            results_row_df = pd.DataFrame(results_row)
            results_row_df.to_csv(results_log_file, header=None, index=False)

        ave_results = np.mean(full_results, 0)
        full_results.append(ave_results)

        results_log_file = 'logs/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_full_results.csv'.format(args.results_dir, args.similarity_type, dual_type, triple_type, EW_type, args.k, args.layer_index, args.lamda, args.self_train_epochs, args.beta)
        
        results_df = pd.DataFrame(full_results)
        results_df.to_csv(results_log_file, header=None, index=False)

        print('After {} training, ave_p_ACC = {}, ave_p_NMI = {}, ave_p_ARI = {}'.
            format(args.times, ave_results[0], ave_results[1], ave_results[2]))


if __name__ == "__main__":
    begin = time.time()
    print(begin)
    main()
    end = time.time()
    print(end)

    print('Need {}s'.format(end-begin))
