#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import *
from utils import batch_seq_generator, san_batch_generator, great_batch_generator, \
eval_metrics, eval_predicted_metrics, pad_sequences
import numpy as np
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score
import argparse
import datetime
import logging
import os, os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import json
from rnn_train_eval import *


NOTE = 'V1.0.0: Formal Version'

            
### Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervise_attention', default=True)
    parser.add_argument('--lamda', default=100, type=float)
    parser.add_argument('--n_epochs', default=40, type=int)
    parser.add_argument('--input_dim', default=73, type=int)
    parser.add_argument('--embedding_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--output_dim', default=1, type=int)
    parser.add_argument('--n_layers', default=2, type=int)
    parser.add_argument('--bidirectional', default=True)
    parser.add_argument('--dropout', default=0.8)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=300, type=int)
    parser.add_argument('--annotator', default='human_intersection')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--ham_percent', default=1.0, type=float)
    parser.add_argument('--log_dir', default='log-gru')
    parser.add_argument("--save_model", default=False, action='store_true')
    parser.add_argument("--early_stop", default=True, action='store_true')
    parser.add_argument("--model_type", default='GRU', type=str)
    parser.add_argument('--data_source', default='yelp')
    parser.add_argument("--semi_type", default='orig-base-self-label', type=str)
    parser.add_argument('--conf_thres', default=0.1, type=float)
    parser.add_argument('--y_dis_weight', default=10, type=float)
    parser.add_argument('--num_add_batch', default=None, type=int)
    parser.add_argument('--num_retrain_epoch', default=5, type=int)
    parser.add_argument('--performance_file', default='all_self_label_first_performance.txt')
    
    args = parser.parse_args()

    assert args.early_stop >= args.save_model
    assert args.model_type.split('-')[0] in ['LSTM', 'GRU']
    assert args.model_type.split('-')[1] in ['RA']
    assert args.model_type.split('-')[2] in ['CE', 'MSE']
    assert args.annotator in ['human_intersection', 'human']
    assert args.data_source in ['yelp', 'n2c2', 'movie', 'senti', 'senti_nj', 'senti_wj']
    assert args.semi_type in ['skip', 'base-self-label', 'acc-help-label', 'orig-base-self-label',
                              'orig-acc-weight-label']
                              
    if args.num_add_batch == 0:
        args.num_add_batch = None
    if args.num_retrain_epoch == 0:
        args.num_retrain_epoch = None
    
    log_directory = args.log_dir + '/' + str(args.n_epochs) + '_epoch/' + args.annotator + '/' + str(
        args.lamda) + '_lambda/' \
                    + args.semi_type + '/' + str(args.ham_percent) + '_ham/' + str(args.conf_thres) + '_thres/'
    if args.semi_type in ['orig-acc-weight-label']:
        log_directory += str(args.y_dis_weight) + '_y_dis_weight/'
    if args.num_retrain_epoch:
        log_directory += str(args.num_retrain_epoch) + '_num_retrain_epoch/'
    if args.num_add_batch:
        log_directory += str(args.num_add_batch) + '_num_add_batch/'
    log_directory += str(args.seed) + '_seed/'

    log_time = str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.', '-')
    log_filename = 'log.' + log_time + '.txt'
    model_filename = 'saved-model.' + log_time + '.pt'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logname = log_directory + log_filename
    modelname = log_directory + model_filename
    
    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    
    if args.data_source == 'yelp':
        X = np.load('./yelp_data/x.npy')
        y = np.load('./yelp_data/y.npy')
        attention_labels = np.load('./yelp_data/human_intersection.npy').astype(int)
        X_train, X_test, y_train, y_test, attention_labels_train, attention_labels_test = train_test_split(X, y,
                                                                                                           attention_labels,
                                                                                                           test_size=0.3,
                                                                                                           random_state=args.seed)
    elif args.data_source == 'n2c2':
        X_train = np.load('./medical_data/X_train.npy')
        X_test = np.load('./medical_data/X_test.npy')
        y_train = np.load('./medical_data/y_train.npy')
        y_test = np.load('./medical_data/y_test.npy')
        attention_labels_train = np.load('./medical_data/att_labels_train.npy').astype(int)
        attention_labels_test = np.load('./medical_data/att_labels_test.npy').astype(int)

    elif args.data_source == 'movie':
        X_train = np.load('./movie_data/x_train.npy')
        X_test = np.load('./movie_data/x_val_test.npy')
        y_train = np.load('./movie_data/y_train.npy')
        y_test = np.load('./movie_data/y_val_test.npy')
        attention_labels_train = np.load('./movie_data/att_labels_train.npy').astype(int)
        attention_labels_test = np.load('./movie_data/att_labels_val_test.npy').astype(int)

    elif args.data_source.startswith('senti'):
        X_train_ham = np.load(f'./{args.data_source}_data/x_with_att_train.npy')
        X_train_noham = np.load(f'./{args.data_source}_data/x_without_att_train.npy')
        X_test_ham = np.load(f'./{args.data_source}_data/x_with_att_val_test.npy')
        X_test_noham = np.load(f'./{args.data_source}_data/x_without_att_val_test.npy')
        y_train_ham = np.load(f'./{args.data_source}_data/y_with_att_train.npy')
        y_train_noham = np.load(f'./{args.data_source}_data/y_without_att_train.npy')
        y_test_ham = np.load(f'./{args.data_source}_data/y_with_att_val_test.npy')
        y_test_noham = np.load(f'./{args.data_source}_data/y_without_att_val_test.npy')
        raw_attention_labels_train_ham = np.load(f'./{args.data_source}_data/att_labels_with_att_train.npy', allow_pickle=True)
        raw_attention_labels_test_ham = np.load(f'./{args.data_source}_data/att_labels_with_att_val_test.npy', allow_pickle=True)
        attention_labels_train_ham = pad_sequences(raw_attention_labels_train_ham, maxlen=56, value=0, dtype="int", truncating="post", padding="post")
        attention_labels_test_ham = pad_sequences(raw_attention_labels_test_ham, maxlen=56, value=0, dtype="int", truncating="post", padding="post")
        attention_labels_train_noham = -np.ones((X_train_noham.shape[0], X_train_noham.shape[1]), int)
        attention_labels_test_noham = -np.ones((X_test_noham.shape[0], X_test_noham.shape[1]), int)
        attention_labels_train = np.concatenate([attention_labels_train_ham, attention_labels_train_noham])
        attention_labels_test = np.concatenate([attention_labels_test_ham, attention_labels_test_noham])
        X_train = np.concatenate([X_train_ham, X_train_noham])
        y_train = np.concatenate([y_train_ham, y_train_noham])
        X_test = np.concatenate([X_test_ham, X_test_noham])
        y_test = np.concatenate([y_test_ham, y_test_noham])

    if args.annotator in ['human', 'human_intersection'] and 0.0 < args.ham_percent < 1.0 and not args.data_source.startswith('senti'):
        (X_train_ham, X_train_noham, y_train_ham, y_train_noham, attention_labels_train_ham,
         attention_labels_train_noham) = train_test_split(X_train, y_train, attention_labels_train,
                                                          test_size=(1 - args.ham_percent), random_state=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    logging.info("Number of GPU is {}".format(n_gpu))
    for i in range(n_gpu):
        logging.info(("Device Name: {},"
                      "Device Capability: {},"
                      "Device Property: {}").format(torch.cuda.get_device_name(i),
                                                    torch.cuda.get_device_capability(i),
                                                    torch.cuda.get_device_properties(i)))

    logging.info(args)
    print(args)
    

    for re_e in range(args.num_retrain_epoch):
        if re_e > 0:
            del model, optimizer_attention, criterion, attention_criterion
            torch.cuda.empty_cache()
        if args.model_type.split('-')[1] == 'RA':
            model = RNN_RA_Attention(args.embedding_dim, args.hidden_dim, args.n_layers, args.model_type)

        criterion = nn.CrossEntropyLoss()
        if args.model_type.split('-')[2] == 'MSE':
            attention_criterion = nn.MSELoss()
        elif args.model_type.split('-')[2] == 'CE':
            attention_criterion = nn.BCELoss()

        model = model.to(device)
        criterion = criterion.to(device)
        attention_criterion = attention_criterion.to(device)
        
        # train on word-level supervision
        if args.early_stop:
            early_stop_sign = 0
        best_human_likeness = 0
        train_att_losses = []
        model.out.weight.requires_grad = False
        optimizer_attention = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0.01)
        for epoch in range(args.n_epochs):

            # train
            if args.annotator in ['human', 'human_intersection']:
                ham_batch_size = min(X_train_ham.shape[0], args.batch_size)
                train_batch_ham_generator = great_batch_generator(X_train_ham, None, attention_labels_train_ham,
                                                                  None, ham_batch_size)
                num_batches = X_train_ham.shape[0] // ham_batch_size
                
                
            if args.annotator in ['human', 'human_intersection']:
                train_attention_loss_ham = train_att_only(model, optimizer_attention, attention_criterion,
                                                          train_batch_ham_generator, num_batches, device)
            train_att_losses.append(train_attention_loss_ham)
            
            # eval
            if args.data_source.startswith('senti'):
                human_likeness, valid_att_pred = eval_human_likeness(model, X_test_ham, y_test_ham,
                                                                     attention_labels_test_ham, device)
            else:
                human_likeness, valid_att_pred = eval_human_likeness(model, X_test, y_test, attention_labels_test,
                                                                     device)
            
            content = f'human_likeness: {human_likeness}'
            print(content)
            logging.info(content)
            
            if best_human_likeness < human_likeness:
                best_human_likeness = human_likeness
                torch.save(model.state_dict(), modelname)
                if args.early_stop:
                    early_stop_sign = 0
            else:
                if args.early_stop:
                    early_stop_sign += 1
                    
                if args.early_stop and early_stop_sign > 3:
                    break
                    
        del model
        torch.cuda.empty_cache()
        if args.model_type.split('-')[1] == 'RA':
            model = RNN_RA_Attention(args.embedding_dim, args.hidden_dim, args.n_layers, args.model_type)
    
        model.load_state_dict(torch.load(modelname))
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(device)
        
        if (args.semi_type != 'skip') and (args.ham_percent < 1.0) and (
                args.annotator in ['human', 'human_intersection']) \
                and (re_e < args.num_retrain_epoch - 1) and (X_train_noham is not None):
            if X_train_noham.size > 0:
                if args.num_add_batch:
                    num_add = args.num_add_batch
                else:
                    num_add = X_train_noham.shape[0] // (args.num_retrain_epoch - 1 - re_e)
                if args.semi_type in ['base-self-label', 'acc-help-label']:
                    pseudo_att, conf_order, acc_list = get_pseudo_attention(model, X_train_noham, y_train_noham,
                                                                            args.conf_thres, device)
                    if args.semi_type == 'base-self-label':
                        pseudo_ix = conf_order[0:num_add]
                    elif args.semi_type == 'acc-help-label':
                        pseudo_ix = conf_order[acc_list][0:num_add]
                elif args.semi_type in ['orig-base-self-label', 'orig-acc-weight-label']:
                    pseudo_att, conf_list, y_dis, acc_list = get_pseudo_attention_orig(model, X_train_noham,
                                                                                       y_train_noham, args.conf_thres,
                                                                                       device)
                    if args.semi_type == 'orig-base-self-label':
                        conf_order = np.argsort(conf_list)[::-1]
                        pseudo_ix = conf_order[0:num_add]
                    elif args.semi_type == 'orig-acc-weight-label':
                        new_conf_list = conf_list - args.y_dis_weight * y_dis
                        conf_order = np.argsort(new_conf_list)[::-1]
                        pseudo_ix = conf_order[0:num_add]
                pseudo_mask = np.isin(np.arange(X_train_noham.shape[0]), pseudo_ix)
                X_train_pseudo_ham = X_train_noham[pseudo_mask]
                y_train_pseudo_ham = y_train_noham[pseudo_mask]
                attention_labels_train_pseudo_ham = pseudo_att[pseudo_mask]
                if args.ham_percent > 0.0 or re_e > 0:
                    X_train_ham = np.append(X_train_ham, X_train_pseudo_ham, axis=0)
                    y_train_ham = np.append(y_train_ham, y_train_pseudo_ham, axis=0)
                    attention_labels_train_ham = np.append(attention_labels_train_ham, attention_labels_train_pseudo_ham,
                                                           axis=0)
                else:
                    X_train_ham = X_train_pseudo_ham
                    y_train_ham = y_train_pseudo_ham
                    attention_labels_train_ham = attention_labels_train_pseudo_ham
                if pseudo_mask.sum() < X_train_ham.shape[0]:
                    X_train_noham = X_train_noham[~pseudo_mask]
                    y_train_noham = y_train_noham[~pseudo_mask]
                    attention_labels_train_noham = attention_labels_train_noham[~pseudo_mask]
                else:
                    X_train_noham, y_train_noham, masks_train_noham, attention_labels_train_noham = None, None, None, None
                print(f"Add {pseudo_mask.sum()} instances to ham data")
                logging.info(f"Add {pseudo_mask.sum()} instances to ham data")
    
    # train on word-level supervision
    if args.early_stop:
        early_stop_sign = 0
    best_human_likeness = 0
    train_att_losses = []
    model.out.weight.requires_grad = False
    optimizer_attention = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0.01)
    for epoch in range(args.n_epochs):

        # train
        if args.annotator in ['human', 'human_intersection']:
            ham_batch_size = min(X_train_ham.shape[0], args.batch_size)
            train_batch_ham_generator = great_batch_generator(X_train_ham, None, attention_labels_train_ham,
                                                              None, ham_batch_size)
            num_batches = X_train_ham.shape[0] // ham_batch_size
            
            
        if args.annotator in ['human', 'human_intersection']:
            train_attention_loss_ham = train_att_only(model, optimizer_attention, attention_criterion,
                                                      train_batch_ham_generator, num_batches, device)
        train_att_losses.append(train_attention_loss_ham)
        
        # eval
        if args.data_source.startswith('senti'):
            human_likeness, valid_att_pred = eval_human_likeness(model, X_test_ham, y_test_ham, attention_labels_test_ham, device)
        else:
            human_likeness, valid_att_pred = eval_human_likeness(model, X_test, y_test, attention_labels_test, device)
        
        content = f'human_likeness: {human_likeness}'
        print(content)
        logging.info(content)
        
        if best_human_likeness < human_likeness:
            best_human_likeness = human_likeness
            torch.save(model.state_dict(), modelname)
            if args.early_stop:
                early_stop_sign = 0
        else:
            if args.early_stop:
                early_stop_sign += 1
                
            if args.early_stop and early_stop_sign > 3:
                break
                
    del model
    torch.cuda.empty_cache()
    if args.model_type.split('-')[1] == 'RA':
        model = RNN_RA_Attention(args.embedding_dim, args.hidden_dim, args.n_layers, args.model_type)

    model.load_state_dict(torch.load(modelname))
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    # train on document-level supervision
    best_valid_acc, human_likeness_best_acc, = 0, 0,
    train_cls_losses = []
    eval_losses = []
    if args.early_stop:
        early_stop_sign = 0
    model.out.weight.requires_grad = True
    model.word.weight.requires_grad = False
    optimizer_classification = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0.01)
    for epoch in range(args.n_epochs):

        # train
        if args.annotator in ['human', 'human_intersection']:
            ham_batch_size = min(X_train_ham.shape[0], args.batch_size)
            num_batches = X_train_ham.shape[0] // ham_batch_size
            
            train_batch_noham_generator = great_batch_generator(X_train_ham, y_train_ham, None, None,
                                                                            ham_batch_size)
            
        if args.annotator in ['human', 'human_intersection']:
            train_classification_loss_ham, train_acc, train_f1 = train_no_ham(model, optimizer_classification, criterion,
                                                                    train_batch_noham_generator, num_batches, device)
            train_cls_losses.append(train_classification_loss_ham)
            
        # eval
        test_batch_generator = batch_seq_generator(X_test, y_test, args.eval_batch_size)
        num_batches = X_test.shape[0] // args.eval_batch_size
        valid_loss, valid_performance_dict, valid_y_pred_values, valid_y_pred = evaluate_acc(model, criterion, test_batch_generator, num_batches, device)
        eval_losses.append(valid_loss)
        valid_acc = valid_performance_dict['acc']
        valid_f1 = valid_performance_dict['mac_f1']

        if args.data_source.startswith('senti'):
            human_likeness, valid_att_pred = eval_human_likeness(model, X_test_ham, y_test_ham, attention_labels_test_ham, device)
        else:
            human_likeness, valid_att_pred = eval_human_likeness(model, X_test, y_test, attention_labels_test, device)
        if best_valid_acc < valid_acc:
            best_valid_acc = valid_acc
            human_likeness_best_acc = human_likeness
            torch.save(model.state_dict(), modelname)
            if args.early_stop:
                early_stop_sign = 0
        else:
            if best_valid_acc - valid_acc < 0.03 and human_likeness - human_likeness_best_acc > 0.01 and args.early_stop:
                pass
            elif args.early_stop:
                early_stop_sign += 1
        if epoch == 0 or epoch == args.n_epochs - 1 or (args.early_stop and early_stop_sign > 3):
            test_batch_generator = batch_seq_generator(X_test, y_test, args.eval_batch_size)
            num_batches = X_test.shape[0] // args.eval_batch_size
            valid_loss, valid_performance_dict, valid_y_pred_values, valid_y_pred = evaluate(model, criterion,
                                                                                             test_batch_generator,
                                                                                             num_batches, device)
        print(f'Train Acc: {train_acc * 100:.2f}%, Train Macro F1: {train_f1 * 100:.2f}%')
        logging.info(f'Train Acc: {train_acc * 100:.2f}%, Train Macro F1: {train_f1 * 100:.2f}%')
        print(f'Val. Acc: {valid_acc * 100:.2f}%, Val. Macro F1: {valid_f1 * 100:.2f}%')
        logging.info(f'Val. Acc: {valid_acc * 100:.2f}%, Val. Macro F1: {valid_f1 * 100:.2f}%')
        content = f'human_likeness: {human_likeness}'
        print(content)
        logging.info(content)
        
        if args.early_stop and early_stop_sign > 3:
            break
    
    del model
    torch.cuda.empty_cache()
    if args.model_type.split('-')[1] == 'RA':
        model = RNN_RA_Attention(args.embedding_dim, args.hidden_dim, args.n_layers, args.model_type)
    model.load_state_dict(torch.load(modelname))
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    test_batch_generator = batch_seq_generator(X_test, y_test, args.eval_batch_size)
    num_batches = X_test.shape[0] // args.eval_batch_size
    best_valid_loss, best_valid_performance_dict, valid_y_pred_values, valid_y_pred = evaluate(model, criterion,
                                                                                               test_batch_generator,
                                                                                               num_batches, device)
    if args.data_source.startswith('senti'):
        human_likeness_best_acc, valid_att_pred = eval_human_likeness(model, X_test_ham, y_test_ham,
                                                                      attention_labels_test_ham,
                                                                      device)
    else:
        human_likeness_best_acc, valid_att_pred = eval_human_likeness(model, X_test, y_test, attention_labels_test,
                                                                      device)
    content = (f"Best valid accuracy: {best_valid_performance_dict['acc']}, Human Likeness: {human_likeness_best_acc},"
               f" F1: {best_valid_performance_dict['mac_f1']}")
    print(content)
    logging.info(content)
    
    att_epoch_count = np.arange(1, len(train_att_losses)+1)
    cls_epoch_count = np.arange(1, len(eval_losses)+1)
    fig1, (ax1, ax2) = plt.subplots(2, figsize=(10, 6),sharex=True,gridspec_kw={'hspace': 0.})
    ax1.plot(cls_epoch_count, train_cls_losses, 'r--')
    ax1.plot(cls_epoch_count, eval_losses, 'b-')
    ax1.legend(['Training Classification Loss', 'Test Loss'], fontsize=14)
    ax2.plot(att_epoch_count, train_att_losses, 'y--')
    ax2.legend(['Training Attention Loss'], fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=16)
    ax2.set_ylabel('Loss', fontsize=16)
    ax1.set_ylabel('Loss', fontsize=16)
    figure_filename = 'fig.' + str(datetime.datetime.now()).replace(' ', '--') + '.png'
    figfullname = log_directory + figure_filename
    plt.savefig(figfullname, dpi=fig1.dpi)

    att_pred_dir = log_directory + 'att_prediction.npy'
    np.save(att_pred_dir, valid_att_pred)
    y_pred_dir = log_directory + 'y_prediction.npy'
    np.save(y_pred_dir, valid_y_pred)
    y_prob_pred_dir = log_directory + 'y_prob_prediction.npy'
    np.save(y_prob_pred_dir, valid_y_pred_values)
    
    performance_dict = vars(args)
    performance_dict['Accuracy'] = best_valid_acc
    for k, v in best_valid_performance_dict.items():
        performance_dict['Best_' + k] = v
    performance_dict['Human_likeness_best_acc'] = human_likeness_best_acc
    performance_dict['Time'] = str(datetime.datetime.now())
    performance_dict['Last_Accuracy'] = valid_acc
    performance_dict['Last_Human_likeness'] = human_likeness
    for k, v in valid_performance_dict.items():
        performance_dict['Last_' + k] = v
    performance_dict['log_directory'] = log_directory
    performance_dict['log_filename'] = log_filename
    performance_dict['note'] = NOTE
    for key, value in performance_dict.items():
        if type(value) is np.int64:
            performance_dict[key] = int(value)
    with open(args.performance_file, 'a+') as outfile:
        outfile.write(json.dumps(performance_dict) + '\n')
    if (not args.save_model) and os.path.exists(modelname):
        os.remove(modelname)

        
if __name__ == '__main__':
    main()