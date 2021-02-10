#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import argparse
import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import random
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from model import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from tqdm import tqdm
from transformers import BertTokenizer, AdamW
from utils import great_batch_generator, mask_batch_seq_generator, pad_sequences
import json

NOTE = 'V1.0.0: Formal Version'


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8
    """
    m = nn.Softmax(dim=1)
    probabilities = m(preds)
    values, indices = torch.max(probabilities, 1)
    y_pred = indices
    acc = accuracy_score(y, y_pred)
    return acc


def token_align(orig_tokens, orig_attentions, tokenizer):
    """
    tokenize a sentence and generate corresponding attention labels
    """
    if type(orig_tokens) is list:
        orig_tokens = orig_tokens[0:200]
    else:
        orig_tokens = orig_tokens.split()
    bert_tokens = []
    new_attentions = []
    bert_tokens.append("[CLS]")
    new_attentions.append(0)
    for orig_token, orig_attent in zip(orig_tokens, orig_attentions):
        token = tokenizer.tokenize(orig_token)
        bert_tokens.extend(token)
        new_attentions.extend([orig_attent for i in token])
    bert_tokens.append("[SEP]")
    new_attentions.append(0)
    return bert_tokens, new_attentions


def token_align_float(orig_tokens, orig_attentions, tokenizer):
    """
    tokenize a sentence and generate corresponding attention labels (float)
    """
    if type(orig_tokens) is list:
        orig_tokens = orig_tokens[0:200]
    else:
        orig_tokens = orig_tokens.split()
    bert_tokens = []
    new_attentions = []
    bert_tokens.append("[CLS]")
    new_attentions.append(0.0)
    for orig_token, orig_attent in zip(orig_tokens, orig_attentions):
        token = tokenizer.tokenize(orig_token)
        bert_tokens.extend(token)
        new_attentions.extend([orig_attent for i in token])
    bert_tokens.append("[SEP]")
    new_attentions.append(0.0)
    return bert_tokens, new_attentions


def token_align_two(orig_tokens, orig_attentions, orig_attentions_for_val, tokenizer):
    """
    tokenize a sentence and generate two corresponding attention labels
    """
    if type(orig_tokens) is list:
        orig_tokens = orig_tokens[0:200]
    else:
        orig_tokens = orig_tokens.split()
    bert_tokens = []
    new_attentions = []
    new_attentions_val = []
    bert_tokens.append("[CLS]")
    new_attentions.append(0)
    new_attentions_val.append(0)
    for orig_token, orig_attent, orig_attent_val in zip(orig_tokens, orig_attentions, orig_attentions_for_val):
        token = tokenizer.tokenize(orig_token)
        bert_tokens.extend(token)
        new_attentions.extend([orig_attent for i in token])
        new_attentions_val.extend([orig_attent_val for i in token])
    bert_tokens.append("[SEP]")
    new_attentions.append(0)
    new_attentions_val.append(0)
    return bert_tokens, new_attentions, new_attentions_val


def tokenize_with_new_attentions(orig_text, orig_attention_list, max_length, tokenizer, if_float=False):
    """
    tokenize a array of raw text and generate corresponding
    attention labels array and attention masks array
    """
    if if_float == True:
        tokens_attents = [token_align_float(r, a, tokenizer) for r, a in zip(orig_text, orig_attention_list)]
    else:
        tokens_attents = [token_align(r, a, tokenizer) for r, a in zip(orig_text, orig_attention_list)]
    bert_tokens = [i[0] for i in tokens_attents]
    attent_labels = [i[1] for i in tokens_attents]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in bert_tokens]
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    if if_float == True:
        attent_labels = pad_sequences(attent_labels, maxlen=max_length, dtype="float", truncating="post",
                                      padding="post")
    else:
        attent_labels = pad_sequences(attent_labels, maxlen=max_length, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    attention_masks = np.array(attention_masks)
    return input_ids, attent_labels, attention_masks


def tokenize_with_two_new_attentions(orig_text, orig_attention_list, orig_attentions_list_for_val, max_length,
                                     tokenizer):
    """
    tokenize a array of raw text and generate two corresponding
    attention labels arraies and attention masks array
    """
    tokens_attents = [token_align_two(r, a, av, tokenizer) for r, a, av in
                      zip(orig_text, orig_attention_list, orig_attentions_list_for_val)]
    bert_tokens = [i[0] for i in tokens_attents]
    attent_labels = [i[1] for i in tokens_attents]
    attent_labels_for_val = [i[2] for i in tokens_attents]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in bert_tokens]
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    attent_labels = pad_sequences(attent_labels, maxlen=max_length, dtype="long", truncating="post", padding="post")
    attent_labels_for_val = pad_sequences(attent_labels_for_val, maxlen=max_length, dtype="long", truncating="post",
                                          padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    attention_masks = np.array(attention_masks)
    return input_ids, attent_labels, attent_labels_for_val, attention_masks


def train(model, optimizer, criterion, attention_criterion, train_batch_generator, num_batches, device, args):
    """
    Main training routine
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_cl = 0
    epoch_al = 0

    model.train()

    # Training
    for b in tqdm(range(num_batches)):
        x_batch, y_batch, att_labels_batch, masks_batch = next(train_batch_generator)
        x_batch = Variable(torch.LongTensor(x_batch)).to(device)
        y_batch = y_batch.astype(np.float)
        y_batch = Variable(torch.LongTensor(y_batch)).to(device)
        att_labels_batch = Variable(torch.FloatTensor(att_labels_batch)).to(device)
        masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)

        optimizer.zero_grad()
        predictions, machine_attention = model(x_batch, masks_batch)

        loss_classification = criterion(predictions, y_batch)

        if args.supervise_attention:
            loss_attention = attention_criterion(machine_attention, att_labels_batch)
            loss = loss_classification + args.lamda * loss_attention

        else:
            loss = loss_classification

        loss.backward()
        optimizer.step()

        acc = binary_accuracy(predictions.detach().cpu(),
                              y_batch.detach().cpu())
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_cl += loss_classification.item()
    if args.supervise_attention:
        epoch_al += loss_attention.item()
        print(
            f'\tClassification Loss: {epoch_cl / num_batches:.3f} | Attention Loss: {epoch_al / num_batches:.3f} | Total: {epoch_loss / num_batches:.3f}')
    else:
        print(f'\tClassification Loss: {epoch_cl / num_batches:.3f}')
    return epoch_loss / num_batches, epoch_acc / num_batches


def train_att_only(model, optimizer, attention_criterion, train_batch_generator, num_batches, device):
    """
    Main training routine
    """
    epoch_loss = 0

    model.train()

    # Training
    for b in tqdm(range(num_batches)):
        a_x_batch, a_att_labels_batch, a_masks_batch = next(train_batch_generator)
        a_x_batch = torch.LongTensor(a_x_batch).to(device)
        a_att_labels_batch = torch.FloatTensor(a_att_labels_batch).to(device)
        a_masks_batch = torch.FloatTensor(a_masks_batch).to(device)

        optimizer.zero_grad()
        predictions, machine_attention = model(a_x_batch, a_masks_batch)
        loss = attention_criterion(machine_attention, a_att_labels_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'\tAttention Loss: {epoch_loss / num_batches:.3f}')
    return epoch_loss / num_batches
    

def train_no_ham(model, optimizer, criterion, train_batch_generator, num_batches, device, args):
    """
    Main training routine without human attention labels
    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    # Training
    for b in tqdm(range(num_batches)):
        x_batch, y_batch, masks_batch = next(train_batch_generator)
        x_batch = Variable(torch.LongTensor(x_batch)).to(device)
        y_batch = y_batch.astype(np.float)
        y_batch = Variable(torch.LongTensor(y_batch)).to(device)
        masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)

        optimizer.zero_grad()
        predictions, machine_attention = model(x_batch, masks_batch)

        loss = criterion(predictions, y_batch)

        loss.backward()
        optimizer.step()

        acc = binary_accuracy(predictions.detach().cpu(),
                              y_batch.detach().cpu())
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    print(f'\tClassification Loss: {epoch_loss / num_batches:.3f}')
    return epoch_loss / num_batches, epoch_acc / num_batches


def external_att_train(model, optimizer, criterion, attention_criterion,
                       train_batch_generator, attention_train_batch_generator,
                       num_batches, device, args):
    """
    Main training routine, with two different data source
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_cl = 0
    epoch_al = 0

    model.train()

    # Training
    for b in tqdm(range(num_batches)):
        x_batch, y_batch, att_labels_batch, masks_batch = next(train_batch_generator)
        a_x_batch, a_att_labels_batch, a_masks_batch = next(attention_train_batch_generator)
        x_batch = torch.LongTensor(x_batch).to(device)
        y_batch = y_batch.astype(np.float)
        y_batch = torch.LongTensor(y_batch).to(device)
        masks_batch = torch.FloatTensor(masks_batch).to(device)
        optimizer.zero_grad()
        predictions, machine_attention = model(x_batch, masks_batch)
        loss_classification = criterion(predictions, y_batch)
        loss = loss_classification
        loss.backward()
        optimizer.step()
        epoch_cl += loss_classification.item()
        acc = binary_accuracy(predictions.detach().cpu(),
                              y_batch.detach().cpu())
        epoch_acc += acc.item()

        optimizer.zero_grad()
        a_x_batch = torch.LongTensor(a_x_batch).to(device)
        a_att_labels_batch = torch.FloatTensor(a_att_labels_batch).to(device)
        a_masks_batch = torch.FloatTensor(a_masks_batch).to(device)
        predictions, machine_attention = model(a_x_batch, a_masks_batch)
        loss_attention = attention_criterion(machine_attention, a_att_labels_batch)
        loss = args.lamda * loss_attention
        loss.backward()
        optimizer.step()
        epoch_al += loss_attention.item()
    print(f'\tClassification Loss: {epoch_cl / (b + 1):.3f} | Attention Loss: {epoch_al / (b + 1):.3f}')
    return epoch_cl / num_batches, epoch_al / num_batches, epoch_acc / num_batches


def get_pseudo_attention(model, X_train_noham, y_train_noham, masks_train_noham, conf_thres, device):
    '''
    Get pseudo attention for noham data
    :param model:
    :param X_train_noham:
    :param y_train_noham:
    :param masks_train_noham:
    :param conf_thres:
    :param device:
    :return:
    '''
    model.eval()
    m = nn.Softmax(dim=1)
    with torch.no_grad():
        x_batch = torch.LongTensor(X_train_noham).to(device)
        y_batch = y_train_noham.astype(np.float)
        y_batch = torch.LongTensor(y_batch).to(device)
        masks_batch = torch.FloatTensor(masks_train_noham).to(device)

        predictions, machine_attention = model(x_batch, masks_batch)

        probabilities = m(predictions)
        values, indices = torch.max(probabilities, 1)
        y_pred = indices
        acc_list = (y_pred == y_batch).detach().cpu().numpy()
        pseudo_att = (machine_attention > conf_thres).detach().cpu().numpy().astype(np.int64)
        conf_list = pseudo_att.sum(axis=1)
        conf_order = np.argsort(conf_list)[::-1]

        return pseudo_att, conf_order, acc_list


def get_pseudo_attention_orig(model, X_train_noham, y_train_noham, masks_train_noham, conf_thres, device):
    '''
    Get pseudo attention for noham data
    :param model:
    :param X_train_noham:
    :param y_train_noham:
    :param masks_train_noham:
    :param conf_thres:
    :param device:
    :return:
    '''
    model.eval()
    m = nn.Softmax(dim=1)
    with torch.no_grad():
        x_batch = torch.LongTensor(X_train_noham).to(device)
        y_batch = y_train_noham.astype(np.float)
        y_batch = torch.LongTensor(y_batch).to(device)
        masks_batch = torch.FloatTensor(masks_train_noham).to(device)

        predictions, machine_attention = model(x_batch, masks_batch)

        probabilities = m(predictions)
        values, indices = torch.max(probabilities, 1)
        y_pred = indices
        acc_list = (y_pred == y_batch).detach().cpu().numpy()
        pseudo_att = (machine_attention > conf_thres).detach().cpu().numpy().astype(np.int64)
        conf_list = (machine_attention - conf_thres).abs().sum(axis=1).detach().cpu().numpy()
        y_dis = (values - y_batch).abs().detach().cpu().numpy()

        return pseudo_att, conf_list, y_dis, acc_list


def get_pseudo_attention_one_only(model, X_train_noham, y_train_noham, masks_train_noham, conf_thres, device):
    '''
    Get pseudo attention for noham data
    :param model:
    :param X_train_noham:
    :param y_train_noham:
    :param masks_train_noham:
    :param conf_thres:
    :param device:
    :return:
    '''
    model.eval()
    m = nn.Softmax(dim=1)
    with torch.no_grad():
        x_batch = torch.LongTensor(X_train_noham).to(device)
        y_batch = y_train_noham.astype(np.float)
        y_batch = torch.LongTensor(y_batch).to(device)
        masks_batch = torch.FloatTensor(masks_train_noham).to(device)

        predictions, machine_attention = model(x_batch, masks_batch)

        probabilities = m(predictions)
        values, indices = torch.max(probabilities, 1)
        y_pred = indices
        acc_list = (y_pred == y_batch).detach().cpu().numpy()
        pseudo_att = (machine_attention > conf_thres).detach().cpu().numpy().astype(np.int64)
        conf_list = np.abs(machine_attention.detach().cpu().numpy()*pseudo_att).sum(axis=1)
        y_dis = (values - y_batch).abs().detach().cpu().numpy()

        return pseudo_att, conf_list, y_dis, acc_list


def evaluate_acc(model, criterion, test_batch_generator, num_batches, device):
    """
    Main evaluation routine
    """
    m = nn.Softmax(dim=1)
    epoch_loss, epoch_acc = 0, 0

    model.eval()

    with torch.no_grad():
        for b in tqdm(range(num_batches)):
            x_batch, y_batch, masks_batch = next(test_batch_generator)
            x_batch = torch.LongTensor(x_batch).to(device)
            y_batch = y_batch.astype(np.float)
            y_batch = torch.LongTensor(y_batch).to(device)
            masks_batch = torch.FloatTensor(masks_batch).to(device)

            predictions, machine_attention = model(x_batch, masks_batch)
            probs = m(predictions)
            y_pred_values, y_pred = torch.max(probs, 1)
            
            loss = criterion(predictions, y_batch)
            acc = accuracy_score(y_batch.detach().cpu(), y_pred.detach().cpu())
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / num_batches, epoch_acc / num_batches


def evaluate(model, criterion, test_batch_generator, num_batches, device):
    """
    Main evaluation routine
    """
    m = nn.Softmax(dim=1)
    epoch_loss, epoch_acc = 0, 0
    epoch_suff, epoch_comp = 0, 0

    model.eval()

    with torch.no_grad():
        for b in tqdm(range(num_batches)):
            x_batch, y_batch, masks_batch = next(test_batch_generator)
            x_batch = torch.LongTensor(x_batch).to(device)
            y_batch = y_batch.astype(np.float)
            y_batch = torch.LongTensor(y_batch).to(device)
            masks_batch = torch.FloatTensor(masks_batch).to(device)

            predictions, machine_attention = model(x_batch, masks_batch)
            probs = m(predictions)
            y_pred_values, y_pred = torch.max(probs, 1)
            
            loss = criterion(predictions, y_batch)
            acc = accuracy_score(y_batch.detach().cpu(), y_pred.detach().cpu())
            
            prop_list = [0.01, 0.05, 0.10, 0.20, 0.50]
            suff_sum = torch.zeros(x_batch.shape[0]).to(device)
            comp_sum = torch.zeros(x_batch.shape[0]).to(device)
            for prop in prop_list:
                suff_mask = machine_attention > torch.topk(machine_attention,
                            max(1,int(machine_attention.shape[1]*prop)), dim=1, largest=False)[0][:,-1].unsqueeze(1)
                comp_mask = machine_attention < torch.topk(machine_attention,
                                max(1,int(machine_attention.shape[1]*prop)), dim=1)[0][:,-1].unsqueeze(1)
                suff_attention = machine_attention * suff_mask
                comp_attention = machine_attention * comp_mask
                suff_predictions, suff_attention = model(x_batch, suff_attention)
                comp_predictions, comp_attention = model(x_batch, comp_attention)
                suff_probs =  m(suff_predictions)
                comp_probs = m(comp_predictions)
                suff_diff_p = probs - suff_probs
                comp_diff_p = probs - comp_probs
                suff_p = torch.masked_select(suff_diff_p, (nn.functional.one_hot(y_pred, num_classes=2) == 1))
                comp_p = torch.masked_select(comp_diff_p, (nn.functional.one_hot(y_pred, num_classes=2) == 1))
                suff_sum += suff_p
                comp_sum += comp_p
            suff_aopc = torch.mean(suff_sum * 1/(len(prop_list)+1))
            comp_aopc = torch.mean(comp_sum * 1/(len(prop_list)+1))
            epoch_suff += suff_aopc.item()
            epoch_comp += comp_aopc.item()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / num_batches, epoch_acc / num_batches, epoch_suff / num_batches, epoch_comp / num_batches


def single_evaluate(model, x_item, y_item, mask_item, device):
    """
    Evaluates a single instance and ALSO returns the attention scores for this instance.
    Use this routine for attention evaluation
    """
    model.eval()

    with torch.no_grad():
        x_item = Variable(torch.LongTensor(x_item)).to(device)
        y_item = y_item.astype(np.float)
        y_item = Variable(torch.LongTensor(y_item)).to(device)
        mask_item = Variable(torch.FloatTensor(mask_item)).to(device)

        predictions, machine_attention = model(x_item, mask_item)

        acc = binary_accuracy(predictions.detach().cpu(),
                              y_item.detach().cpu())
        epoch_acc = acc.item()

    return epoch_acc, machine_attention


def eval_human_likeness(model, X_test, y_test, mask_test, attention_labels_test, device):
    total_auc = 0
    num_instances = 0

    for i in range(0, X_test.shape[0]):
        item_acc, att_weights = single_evaluate(model, X_test[i:i + 1], y_test[i:i + 1], mask_test[i:i + 1], device)
        att_weights = torch.squeeze(att_weights)
        human_att = attention_labels_test[i]
        if np.sum(human_att) > 0:
            auc = roc_auc_score(human_att, att_weights.detach().cpu())
            total_auc += auc
            num_instances += 1

    mean_auc = total_auc / num_instances
    print("Human-likeness score: ", mean_auc)
    return mean_auc

            
            
### Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervise_attention', default=True, action='store_false')
    parser.add_argument('--lamda', default=1000, type=float)
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str)
    parser.add_argument('--n_epochs', default=20, type=int)
    parser.add_argument('--max_length', default=133, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=300, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--annotator', default='human_intersection', type=str)
    parser.add_argument('--ham_percent', default=1.0, type=float)
    parser.add_argument('--log_dir', default='log-BERT', type=str)
    parser.add_argument("--save_model", default=False, action='store_true')
    parser.add_argument("--early_stop", default=True, action='store_true')
    parser.add_argument("--model_type", default='Bert-RA-CE', type=str)
    parser.add_argument("--semi_type", default='orig-base-self-label', type=str)
    parser.add_argument('--conf_thres', default=0.1, type=float)
    parser.add_argument('--y_dis_weight', default=10, type=float)
    parser.add_argument('--num_add', default=None, type=int)
    parser.add_argument('--num_add_batch', default=None, type=int)
    parser.add_argument('--num_retrain_epoch', default=5, type=int)
    parser.add_argument('--data_source', default='yelp')
    
    args = parser.parse_args()
    
    assert args.model_type.split('-')[0] in ['Bert']
    assert args.model_type.split('-')[1] in ['RA']
    assert args.model_type.split('-')[2] in ['CE', 'MSE']
    assert args.annotator in ['human_intersection', 'human', 'eye_tracking']
    assert args.data_source in ['yelp', 'n2c2', 'movie']
    
    
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

    log_time = str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.', '-')
    log_filename = 'log.' + log_time + '.txt'
    per_filename = 'performance.csv'
    model_filename = 'saved-model.' + log_time + '.pt'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logname = log_directory + log_filename
    modelname = log_directory + model_filename
    perfilename = log_directory + per_filename

    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

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
                      "Device Properties: {}").format(torch.cuda.get_device_name(i),
                                                      torch.cuda.get_device_capability(i),
                                                      torch.cuda.get_device_properties(i)))
                                                      
    if args.data_source == 'yelp':
        raw_reviews = np.load('./yelp_data/raw_reviews.npy')
        y = np.load('./yelp_data/y.npy')
        raw_attention_labels = None
        if args.annotator == 'human_intersection':
            raw_attention_labels = np.load('./yelp_data/human_intersection.npy')
        elif args.annotator == 'eye_tracking':
            raw_eye_text = np.load('./data/zuco_new_text.npy')
            raw_eye_attention_labels = np.load('./eye_data/mfd_scores.npy')
            raw_eye_attention_labels_std = (raw_eye_attention_labels - raw_eye_attention_labels.min(axis=0)) / (
                    raw_eye_attention_labels.max(axis=0) - raw_eye_attention_labels.min(axis=0))
            raw_eye_attention_labels = raw_eye_attention_labels_std * (1. - 0.) + 0.
            raw_eye_attention_labels = (raw_eye_attention_labels > 0.5).astype(int)
            raw_attention_labels_for_val = np.load('./yelp_data/human_intersection.npy')
            raw_attention_labels_for_val = raw_attention_labels_for_val.astype(int)
        else:
            pass
        if args.annotator != 'eye_tracking':
            raw_attention_labels = raw_attention_labels.astype(int)
    elif args.data_source == 'movie':
        raw_review_train = np.load("./movie_data/raw_text_train.npy", allow_pickle=True)
        raw_review_test = np.load("./movie_data/raw_text_val_test.npy", allow_pickle=True)
        y_train = np.load('./movie_data/y_train.npy')
        y_test = np.load('./movie_data/y_val_test.npy')
        if args.annotator == 'human':
            raw_attention_labels_train = np.load('./movie_data/att_labels_train.npy').astype(int)
            raw_attention_labels_test = np.load('./movie_data/att_labels_val_test.npy').astype(int)
        elif args.annotator == 'eye_tracking':
            raw_eye_text = np.load('./eye_data/zuco_new_text.npy')
            raw_eye_attention_labels = np.load('./eye_data/mfd_scores.npy')
            raw_eye_attention_labels_std = (raw_eye_attention_labels - raw_eye_attention_labels.min(axis=0)) / (
                    raw_eye_attention_labels.max(axis=0) - raw_eye_attention_labels.min(axis=0))
            raw_eye_attention_labels = raw_eye_attention_labels_std * (1. - 0.) + 0.
            raw_eye_attention_labels = (raw_eye_attention_labels > 0.5).astype(int)
            raw_attention_labels_for_val_train = np.load('./movie_data/att_labels_train.npy').astype(int)
            raw_attention_labels_for_val_test = np.load('./movie_data/att_labels_val_test.npy').astype(int)
        else:
            pass
    elif args.data_source == 'n2c2':
        raw_review_train = np.load("./medical_data/raw_notes_train.npy", allow_pickle=True)
        raw_review_test = np.load("./medical_data/raw_notes_test.npy", allow_pickle=True)
        y_train = np.load('./medical_data/y_train.npy')
        y_test = np.load('./medical_data/y_test.npy')
        if args.annotator == 'human':
            raw_attention_labels_train = np.load('./medical_data/att_labels_train.npy').astype(int)
            raw_attention_labels_test = np.load('./medical_data/att_labels_test.npy').astype(int)
        elif args.annotator == 'eye_tracking':
            raw_eye_text = np.load('./eye_data/zuco_new_text.npy')
            raw_eye_attention_labels = np.load('./eye_data/mfd_scores.npy')
            raw_eye_attention_labels_std = (raw_eye_attention_labels - raw_eye_attention_labels.min(axis=0)) / (
                    raw_eye_attention_labels.max(axis=0) - raw_eye_attention_labels.min(axis=0))
            raw_eye_attention_labels = raw_eye_attention_labels_std * (1. - 0.) + 0.
            raw_eye_attention_labels = (raw_eye_attention_labels > 0.5).astype(int)
            raw_attention_labels_for_val_train = np.load('./medical_data/att_labels_train.npy').astype(int)
            raw_attention_labels_for_val_test = np.load('./medical_data/att_labels_test.npy').astype(int)
        else:
            pass
    else:
        pass
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    
    if args.data_source in ['yelp']:
        if args.annotator == 'eye_tracking':
            eye_input_ids, eye_attention_labels, eye_attention_masks = tokenize_with_new_attentions(
                raw_eye_text, raw_eye_attention_labels, args.max_length, tokenizer, if_float=True)
            input_ids, attention_labels_for_val, attention_masks = tokenize_with_new_attentions(
                raw_reviews, raw_attention_labels_for_val, args.max_length, tokenizer)
        else:
            input_ids, attention_labels, attention_masks = tokenize_with_new_attentions(
                raw_reviews, raw_attention_labels, args.max_length, tokenizer)

        if args.annotator == 'eye_tracking':
            (X_train, X_test, y_train, y_test,
             attention_labels_for_val_train,
             attention_labels_for_val_test,
             masks_train, masks_test) = train_test_split(input_ids, y, attention_labels_for_val, attention_masks,
                                                         test_size=0.3, random_state=args.seed)
        else:
            (X_train, X_test, y_train, y_test,
             attention_labels_train, attention_labels_test,
             masks_train, masks_test) = train_test_split(input_ids, y, attention_labels, attention_masks, test_size=0.3,
                                                         random_state=args.seed)
    else:
        if args.annotator == 'human':
            X_train, attention_labels_train, masks_train = tokenize_with_new_attentions(
                raw_review_train, raw_attention_labels_train, args.max_length, tokenizer)
            X_test, attention_labels_test, masks_test = tokenize_with_new_attentions(
                raw_review_test, raw_attention_labels_test, args.max_length, tokenizer)
        elif args.annotator == 'eye_tracking':
            eye_input_ids, eye_attention_labels, eye_attention_masks = tokenize_with_new_attentions(
                raw_eye_text, raw_eye_attention_labels, args.max_length, tokenizer, if_float=True)
            X_train, attention_labels_for_val_train, masks_train = tokenize_with_new_attentions(
                raw_review_train, raw_attention_labels_for_val_train, args.max_length, tokenizer)
            X_test, attention_labels_for_val_test, masks_test = tokenize_with_new_attentions(
                raw_review_test, raw_attention_labels_for_val_test, args.max_length, tokenizer)
        else:
            pass

    if args.annotator in ['human', 'human_intersection'] and 0.0 < args.ham_percent < 1.0:
        (X_train_ham, X_train_noham, y_train_ham, y_train_noham,
         attention_labels_train_ham, attention_labels_train_noham,
         masks_train_ham, masks_train_noham) = train_test_split(X_train, y_train, attention_labels_train, masks_train,
                                                                test_size=(1 - args.ham_percent), random_state=args.seed)
    
    logging.info(args)
    print(args)
    
    for re_e in range(args.num_retrain_epoch):
        if re_e > 0:
            del model, optimizer_attention, criterion, attention_criterion
            torch.cuda.empty_cache()
        if args.model_type.split('-')[1] == 'RA':
            model = Bert_RA_Attention.from_pretrained(args.bert_model)
        else:
            pass
            
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        criterion = nn.CrossEntropyLoss()
        attention_criterion = None
        if args.model_type.split('-')[2] == 'CE':
            attention_criterion = nn.BCELoss()
        else:
            attention_criterion = nn.MSELoss()
        model = model.to(device)
        criterion = criterion.to(device)
        attention_criterion = attention_criterion.to(device)
        
        # train on word-level supervision
        if args.early_stop:
            early_stop_sign = 0
        best_human_likeness = 0
        train_att_losses = []
        no_decay = ['bias', 'LayerNorm.weight']
        model.out.weight.requires_grad = False
        param_optimizer = filter(lambda p: p[1].requires_grad, list(model.named_parameters()))
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer_attention = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        for epoch in range(args.n_epochs):

            # train
            if args.annotator in ['human', 'human_intersection']:
                ham_batch_size = min(X_train_ham.shape[0], args.batch_size)
                train_batch_ham_generator = great_batch_generator(X_train_ham, None, attention_labels_train_ham,
                                                                  masks_train_ham, ham_batch_size)
                num_batches = X_train_ham.shape[0] // ham_batch_size
                
                
            if args.annotator in ['human', 'human_intersection']:
                train_attention_loss_ham = train_att_only(model, optimizer_attention, attention_criterion,
                                                          train_batch_ham_generator, num_batches, device)
            train_att_losses.append(train_attention_loss_ham)
            
            # eval
            human_likeness = eval_human_likeness(model, X_test, y_test, masks_test, attention_labels_test, device)
            
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
        
        # get pseudo labels
        if args.early_stop or args.save_model:
            del model
            torch.cuda.empty_cache()
            if args.model_type.split('-')[1] == 'RA':
                model = Bert_RA_Attention.from_pretrained(args.bert_model)
            else:
                pass
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
                                                                            masks_train_noham, args.conf_thres, device)

                    if args.semi_type == 'base-self-label':
                        pseudo_ix = conf_order[0:num_add]
                    elif args.semi_type == 'acc-help-label':
                        pseudo_ix = conf_order[acc_list][0:num_add]
                elif args.semi_type in ['orig-base-self-label', 'orig-acc-weight-label']:
                    pseudo_att, conf_list, y_dis, acc_list = get_pseudo_attention_orig(model, X_train_noham,
                                                                                       y_train_noham, masks_train_noham,
                                                                                       args.conf_thres, device)
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
                masks_train_pseudo_ham = masks_train_noham[pseudo_mask]
                attention_labels_train_pseudo_ham = pseudo_att[pseudo_mask]
                if args.ham_percent > 0.0 or re_e > 0:
                    X_train_ham = np.append(X_train_ham, X_train_pseudo_ham, axis=0)
                    y_train_ham = np.append(y_train_ham, y_train_pseudo_ham, axis=0)
                    masks_train_ham = np.append(masks_train_ham, masks_train_pseudo_ham, axis=0)
                    attention_labels_train_ham = np.append(attention_labels_train_ham, attention_labels_train_pseudo_ham,
                                                           axis=0)
                else:
                    X_train_ham = X_train_pseudo_ham
                    y_train_ham = y_train_pseudo_ham
                    masks_train_ham = masks_train_pseudo_ham
                    attention_labels_train_ham = attention_labels_train_pseudo_ham                                           
                if pseudo_mask.sum() < X_train_ham.shape[0]:
                    X_train_noham = X_train_noham[~pseudo_mask]
                    y_train_noham = y_train_noham[~pseudo_mask]
                    masks_train_noham = masks_train_noham[~pseudo_mask]
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
    no_decay = ['bias', 'LayerNorm.weight']
    model.out.weight.requires_grad = False
    param_optimizer = filter(lambda p: p[1].requires_grad, list(model.named_parameters()))
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer_attention = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    for epoch in range(args.n_epochs):

        # train
        if args.annotator in ['human', 'human_intersection']:
            ham_batch_size = min(X_train_ham.shape[0], args.batch_size)
            train_batch_ham_generator = great_batch_generator(X_train_ham, None, attention_labels_train_ham,
                                                              masks_train_ham, ham_batch_size)
            num_batches = X_train_ham.shape[0] // ham_batch_size
            
            
        if args.annotator in ['human', 'human_intersection']:
            train_attention_loss_ham = train_att_only(model, optimizer_attention, attention_criterion,
                                                      train_batch_ham_generator, num_batches, device)
        train_att_losses.append(train_attention_loss_ham)
        
        # eval
        human_likeness = eval_human_likeness(model, X_test, y_test, masks_test, attention_labels_test, device)
        
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
        model = Bert_RA_Attention.from_pretrained(args.bert_model)
    else:
        pass
    model.load_state_dict(torch.load(modelname))
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    # train on document-level supervision
    best_valid_acc, human_likeness_best_acc, suff_best_acc, comp_best_acc  = 0, 0, 0, 0
    train_cls_losses = []
    eval_losses = []
    if args.early_stop:
        early_stop_sign = 0
    model.out.weight.requires_grad = True
    model.word.weight.requires_grad = False
    param_optimizer = filter(lambda p: p[1].requires_grad, list(model.named_parameters()))
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer_classification = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    for epoch in range(args.n_epochs):

        # train
        if args.annotator in ['human', 'human_intersection']:
            ham_batch_size = min(X_train_ham.shape[0], args.batch_size)
            num_batches = X_train_ham.shape[0] // ham_batch_size
            train_batch_noham_generator = great_batch_generator(X_train_ham, y_train_ham, None,
                                                                    masks_train_ham,
                                                                    ham_batch_size)
            train_classification_loss_ham, train_acc = train_no_ham(model, optimizer_classification, criterion,
                                                                 train_batch_noham_generator,
                                                                 num_batches, device, args)
            train_cls_losses.append(train_classification_loss_ham)
        
        # eval
        test_batch_generator = mask_batch_seq_generator(X_test, y_test, masks_test, args.eval_batch_size)
        num_batches = X_test.shape[0] // args.eval_batch_size
        valid_loss, valid_acc = evaluate_acc(model, criterion, test_batch_generator, num_batches, device)
        eval_losses.append(valid_loss)
        
        if args.annotator in ['eye_tracking']:
            human_likeness = eval_human_likeness(model, X_test, y_test, masks_test, attention_labels_for_val_test,
                                                 device)
        else:
            human_likeness = eval_human_likeness(model, X_test, y_test, masks_test, attention_labels_test, device)
            
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
            test_batch_generator = mask_batch_seq_generator(X_test, y_test, masks_test, args.eval_batch_size)
            num_batches = X_test.shape[0] // args.eval_batch_size
            valid_loss, valid_acc, valid_suff, valid_comp = evaluate(model, criterion, test_batch_generator, num_batches, device)
        
        print(f'Train Acc: {train_acc * 100:.2f}%')
        logging.info(f'Train Acc: {train_acc * 100:.2f}%')
        print(f'Val. Acc: {valid_acc * 100:.2f}%')
        logging.info(f'Val. Acc: {valid_acc * 100:.2f}%')
        content = f'human_likeness: {human_likeness}'
        print(content)
        logging.info(content)
        
    del model
    torch.cuda.empty_cache()
    if args.model_type.split('-')[1] == 'RA':
        model = Bert_RA_Attention.from_pretrained(args.bert_model)
    else:
        pass
    model.load_state_dict(torch.load(modelname))
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    test_batch_generator = mask_batch_seq_generator(X_test, y_test, masks_test, args.eval_batch_size)
    num_batches = X_test.shape[0] // args.eval_batch_size
    best_valid_loss, best_valid_acc, suff_best_acc, comp_best_acc = evaluate(model, criterion, test_batch_generator, num_batches, device)
    content = (f"Best valid accuracy: {best_valid_acc}, Human Likeness: {human_likeness_best_acc},"
    f" Sufficiency: {suff_best_acc}, Comprehensiveness: {comp_best_acc}")
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
    

    performance_dict = vars(args)
    performance_dict['Accuracy'] = best_valid_acc
    performance_dict['Time'] = str(datetime.datetime.now())
    performance_dict['Last_Accuracy'] = valid_acc
    performance_dict['Last_Human_likeness'] = human_likeness
    performance_dict['Last_Sufficiency'] = valid_suff
    performance_dict['Last_Comprehensiveness'] = valid_comp
    performance_dict['Human_likeness_best_acc'] = human_likeness_best_acc
    performance_dict['suff_best_acc'] = suff_best_acc
    performance_dict['comp_best_acc'] = comp_best_acc
    performance_dict['log_directory'] = log_directory
    performance_dict['log_filename'] = log_filename
    performance_dict['note'] = NOTE
    with open('all_test_performance.txt', 'a+') as outfile:
        outfile.write(json.dumps(performance_dict) + '\n')
    if (not args.save_model) and os.path.exists(modelname):
        os.remove(modelname)
        

if __name__ == '__main__':
    main()