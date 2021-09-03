#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import *
from utils import batch_seq_generator, san_batch_generator, great_batch_generator, \
eval_metrics, eval_predicted_metrics
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


def train(model, optimizer, criterion, attention_criterion, train_batch_generator, num_batches, device, args):
    """
    Main training routine
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    epoch_cl = 0
    epoch_al = 0

    model.train()

    # Training
    for b in tqdm(range(num_batches)):
        x_batch, y_batch, att_labels = next(train_batch_generator)
        x_batch = Variable(torch.FloatTensor(x_batch)).to(device)
        y_batch = y_batch.astype(np.float)
        y_batch = Variable(torch.LongTensor(y_batch)).to(device)
        att_labels = Variable(torch.FloatTensor(att_labels)).to(device)
        # att_labels = f.normalize(att_labels, p=1, dim=1)

        optimizer.zero_grad()
        predictions, machine_attention = model(x_batch)

        loss_classification = criterion(predictions, y_batch)

        if args.supervise_attention:
            # loss_attention = torch.sum((machine_attention - att_labels) ** 2)
            # loss = loss_classification + args.lamda * loss_attention.type('torch.FloatTensor')
            # loss_attention = attention_criterion(machine_attention.view(-1), att_labels.view(-1))
            loss_attention = attention_criterion(machine_attention, att_labels)
            loss = loss_classification + args.lamda * loss_attention
            # loss = loss_attention

        else:
            loss = loss_classification

        loss.backward()
        optimizer.step()

        performance_dict = eval_metrics(predictions.detach().cpu(), y_batch.detach().cpu())
        epoch_loss += loss.item()
        epoch_acc += performance_dict['acc'].item()
        epoch_f1 += performance_dict['mac_f1'].item()
        epoch_cl += loss_classification.item()
        epoch_al += loss_attention.item()

    print(
        f'\tClassification Loss: {epoch_cl / num_batches:.3f} | Attention Loss: {epoch_al / num_batches:.3f} | Total: {epoch_loss / num_batches:.3f}')
    return epoch_loss / num_batches, epoch_acc / num_batches, epoch_f1 / num_batches


def train_no_ham(model, optimizer, criterion, train_batch_generator, num_batches, device):
    """
    Main training routine
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    model.train()

    # Training
    for b in tqdm(range(num_batches)):
        x_batch, y_batch = next(train_batch_generator)
        x_batch = Variable(torch.FloatTensor(x_batch)).to(device)
        y_batch = y_batch.astype(np.float)
        y_batch = Variable(torch.LongTensor(y_batch)).to(device)

        optimizer.zero_grad()
        predictions, machine_attention = model(x_batch)

        loss = criterion(predictions, y_batch)

        loss.backward()
        optimizer.step()

        performance_dict = eval_metrics(predictions.detach().cpu(),
                                        y_batch.detach().cpu())
        epoch_loss += loss.item()
        epoch_acc += performance_dict['acc'].item()
        epoch_f1 += performance_dict['mac_f1'].item()

    print(f'\tClassification Loss: {epoch_loss / num_batches:.3f}')
    return epoch_loss / num_batches, epoch_acc / num_batches, epoch_f1 / num_batches


def train_att_only(model, optimizer, attention_criterion, train_batch_generator, num_batches, device):
    """
    Main training routine
    """
    epoch_loss = 0

    model.train()

    # Training
    for b in tqdm(range(num_batches)):
        a_x_batch, a_att_labels_batch = next(train_batch_generator)
        a_x_batch = torch.FloatTensor(a_x_batch).to(device)
        a_att_labels_batch = torch.FloatTensor(a_att_labels_batch).to(device)

        optimizer.zero_grad()
        predictions, machine_attention = model(a_x_batch)
        loss = attention_criterion(machine_attention, a_att_labels_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'\tAttention Loss: {epoch_loss / num_batches:.3f}')
    return epoch_loss / num_batches


def external_att_train(model, optimizer, criterion, attention_criterion,
                       train_batch_generator, attention_train_batch_generator,
                       num_batches, device, args):
    """
    Main training routine, with two different data source
    """
    epoch_f1 = 0
    epoch_acc = 0
    epoch_cl = 0
    epoch_al = 0

    model.train()

    # Training
    for b in tqdm(range(num_batches)):
        x_batch, y_batch, att_labels = next(train_batch_generator)
        a_x_batch, a_att_labels_batch = next(attention_train_batch_generator)
        x_batch = torch.FloatTensor(x_batch).to(device)
        y_batch = y_batch.astype(np.float)
        y_batch = torch.LongTensor(y_batch).to(device)
        # att_labels = f.normalize(att_labels, p=1, dim=1)

        optimizer.zero_grad()
        predictions, machine_attention = model(x_batch)

        loss_classification = criterion(predictions, y_batch)
        loss = loss_classification
        loss.backward()
        optimizer.step()
        epoch_cl += loss_classification.item()
        performance_dict = eval_metrics(predictions.detach().cpu(),
                                        y_batch.detach().cpu())
        epoch_acc += performance_dict['acc'].item()
        epoch_f1 += performance_dict['mac_f1'].item()

        a_x_batch = torch.FloatTensor(a_x_batch).to(device)
        a_att_labels_batch = torch.FloatTensor(a_att_labels_batch).to(device)

        optimizer.zero_grad()
        predictions, machine_attention = model(a_x_batch)
        loss_attention = attention_criterion(machine_attention, a_att_labels_batch)

        loss = args.lamda * loss_attention
        loss.backward()
        optimizer.step()
        epoch_al += loss_attention.item()
    print(f'\tClassification Loss: {epoch_cl / (b + 1):.3f} | Attention Loss: {epoch_al / (b + 1):.3f}')
    return epoch_cl / num_batches, epoch_al / num_batches, epoch_acc / num_batches, epoch_f1 / num_batches


def evaluate_acc(model, criterion, test_batch_generator, num_batches, device):
    m = nn.Softmax(dim=1)
    epoch_loss, epoch_acc = 0, 0
    model.eval()

    prob_list, pred_list, y_list, = [], [], [],
    with torch.no_grad():
        for b in tqdm(range(num_batches)):
            x_batch, y_batch = next(test_batch_generator)
            x_batch = torch.FloatTensor(x_batch).to(device)
            y_batch = y_batch.astype(np.float)
            y_batch = torch.LongTensor(y_batch).to(device)

            predictions, machine_attention = model(x_batch)
            probs = m(predictions)
            y_pred_values, y_pred = torch.max(probs, 1)

            loss = criterion(predictions, y_batch)
            prob_list.append(y_pred_values.detach().cpu())
            pred_list.append(y_pred.detach().cpu())
            y_list.append(y_batch.detach().cpu())

            epoch_loss += loss.item()
        all_y_pred_values = np.concatenate(prob_list)
        all_y_pred = np.concatenate(pred_list)
        all_y = np.concatenate(y_list)
        performance_dict = eval_predicted_metrics(all_y_pred, all_y_pred_values, all_y)
    return epoch_loss / num_batches, performance_dict, all_y_pred_values, all_y_pred


def evaluate(model, criterion, test_batch_generator, num_batches, device):
    m = nn.Softmax(dim=1)
    epoch_loss, epoch_acc = 0, 0
    epoch_suff, epoch_comp = 0, 0
    model.eval()
    prob_list, pred_list, y_list, = [], [], [],
    with torch.no_grad():
        for b in tqdm(range(num_batches)):
            x_batch, y_batch = next(test_batch_generator)
            x_batch = torch.FloatTensor(x_batch).to(device)
            y_batch = y_batch.astype(np.float)
            y_batch = torch.LongTensor(y_batch).to(device)

            predictions, machine_attention = model(x_batch)
            probs = m(predictions)
            y_pred_values, y_pred = torch.max(probs, 1)

            loss = criterion(predictions, y_batch)
            prob_list.append(y_pred_values.detach().cpu())
            pred_list.append(y_pred.detach().cpu())
            y_list.append(y_batch.detach().cpu())

            epoch_loss += loss.item()
        all_y_pred_values = np.concatenate(prob_list)
        all_y_pred = np.concatenate(pred_list)
        all_y = np.concatenate(y_list)
        performance_dict = eval_predicted_metrics(all_y_pred, all_y_pred_values, all_y)
    return epoch_loss / num_batches, performance_dict, all_y_pred_values, all_y_pred


def single_evaluate(model, x_item, y_item, device):
    """
    Evaluates a single instance and ALSO returns the attention scores for this instance.
    Use this routine for attention evaluation
    """
    model.eval()

    with torch.no_grad():
        x_item = Variable(torch.FloatTensor(x_item)).to(device)
        y_item = y_item.astype(np.float)
        y_item = Variable(torch.LongTensor(y_item)).to(device)

        predictions, machine_attention = model(x_item)

        performance_dict = eval_metrics(predictions.detach().cpu(),
                                        y_item.detach().cpu())
        epoch_acc = performance_dict['acc'].item()

    return epoch_acc, machine_attention


def eval_human_likeness(model, X_test, y_test, attention_labels_test, device):
    total_auc = 0
    num_instances = 0

    att_list = []
    for i in range(0, X_test.shape[0]):
        item_acc, att_weights = single_evaluate(model, X_test[i:i + 1], y_test[i:i + 1], device)
        att_weights = torch.squeeze(att_weights)
        att_list.append(att_weights.detach().cpu())
        human_att = attention_labels_test[i]
        if np.sum(human_att) > 0:
            auc = roc_auc_score(human_att, att_weights.detach().cpu())
            total_auc += auc
            num_instances += 1

    all_att_pred = np.concatenate(att_list)
    mean_auc = total_auc / num_instances
    print("Human-likeness score: ", mean_auc)
    return mean_auc, all_att_pred


def get_pseudo_attention(model, X_train_noham, y_train_noham, conf_thres, device):
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
        x_batch = torch.FloatTensor(X_train_noham).to(device)
        y_batch = y_train_noham.astype(np.float)
        y_batch = torch.LongTensor(y_batch).to(device)

        predictions, machine_attention = model(x_batch)

        probabilities = m(predictions)
        values, indices = torch.max(probabilities, 1)
        y_pred = indices
        acc_list = (y_pred == y_batch).detach().cpu().numpy()
        pseudo_att = (machine_attention > conf_thres).detach().cpu().numpy().astype(np.int64)
        conf_list = pseudo_att.sum(axis=1)
        conf_order = np.argsort(conf_list)[::-1]

        return pseudo_att, conf_order, acc_list


def get_pseudo_attention_orig(model, X_train_noham, y_train_noham, conf_thres, device):
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
    batch_size = 500
    i = 0
    patt_list, cl_list, yd_list, al_list = [], [], [], []
    with torch.no_grad():
        while True:
            x_batch = torch.FloatTensor(X_train_noham[i:i+batch_size]).to(device)
            y_batch = y_train_noham[i:i+batch_size].astype(np.float)
            y_batch = torch.LongTensor(y_batch).to(device)

            predictions, machine_attention = model(x_batch)

            probabilities = m(predictions)
            values, indices = torch.max(probabilities, 1)
            y_pred = indices
            acc_list = (y_pred == y_batch).detach().cpu().numpy()
            pseudo_att = (machine_attention > conf_thres).detach().cpu().numpy().astype(np.int64)
            conf_list = (machine_attention - conf_thres).abs().sum(axis=1).detach().cpu().numpy()
            y_dis = (values - y_batch).abs().detach().cpu().numpy()
            al_list.append(acc_list)
            patt_list.append(pseudo_att)
            cl_list.append(conf_list)
            yd_list.append(y_dis)
            if i + batch_size < X_train_noham.shape[0]:
                i += batch_size
            else:
                break
        pseudo_att = np.concatenate(patt_list)
        conf_list = np.concatenate(cl_list)
        y_dis = np.concatenate(yd_list)
        acc_list = np.concatenate(al_list)
        return pseudo_att, conf_list, y_dis, acc_list


def get_pseudo_attention_one_only(model, X_train_noham, y_train_noham, conf_thres, device):
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
        x_batch = torch.FloatTensor(X_train_noham).to(device)
        y_batch = y_train_noham.astype(np.float)
        y_batch = torch.LongTensor(y_batch).to(device)

        predictions, machine_attention = model(x_batch)

        probabilities = m(predictions)
        values, indices = torch.max(probabilities, 1)
        y_pred = indices
        acc_list = (y_pred == y_batch).detach().cpu().numpy()
        pseudo_att = (machine_attention > conf_thres).detach().cpu().numpy().astype(np.int64)
        conf_list = np.abs(machine_attention.detach().cpu().numpy() * pseudo_att).sum(axis=1)
        y_dis = (values - y_batch).abs().detach().cpu().numpy()

        return pseudo_att, conf_list, y_dis, acc_list