#!/usr/bin/env python
# -*- coding: UTF-8 -*-

########################################################################
# GNU General Public License v3.0
# GNU GPLv3
# Copyright (c) 2019, Noureldien Hussein
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
########################################################################

"""
Evaluation functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from config_v import params
import numpy as np
from sklearn.metrics import average_precision_score

def map_charades(y_true, y_pred):
    """ Returns mAP """
    m_aps = []
    n_classes = y_pred.shape[1]
    y_true_one_hot = np.zeros_like(y_pred)
    y_true_one_hot[np.arange(len(y_true)), y_true] = 1

    for oc_i in range(n_classes):
        # Extract predictions and true values for class `oc_i`
        pred_row = y_pred[:, oc_i]
        true_row = y_true_one_hot[:, oc_i]

        # Sort predictions in descending order
        sorted_idxs = np.argsort(-pred_row)

        # True positives and false positives
        tp = true_row[sorted_idxs] == 1
        fp = ~tp
        n_pos = tp.sum()

        # If no positive samples, skip this class
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue

        # Compute precision at each threshold
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs + t_pcs).astype(float)

        # Compute average precision
        avg_prec = 0
        for i in range(len(tp)):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos)

    # Compute mean average precision
    m_aps = np.array(m_aps)
    m_ap = np.nanmean(m_aps)  # Ignore NaN values
    return m_ap

def map_sklearn(y_true, y_pred):
    # """ Returns mAP """
    n_classes = y_true.shape[1]
    map = [average_precision_score(y_true[:, i], y_pred[:, i]) for i in range(n_classes)]
    map = np.nan_to_num(map)
    map = np.mean(map)
    return map

def accuracy(y_true, y_pred):
    idx = np.argmax(y_pred, axis=1)
    n_items = len(y_true)
    accuracy = np.sum(idx == y_true) / float(n_items)
    return accuracy

def acuracy_top_n(n_top, y_true, y_pred):
    n_corrects = 0
    for gt, pr in zip(y_true, y_pred):
        idx = np.argsort(pr)[::-1]
        idx = idx[0:n_top]
        gt = np.where(gt == 1)[0][0]
        if gt in idx:
            n_corrects += 1
    n = len(y_true)
    score = n_corrects / float(n)
    return score

