#!/usr/bin/env python3
"""
script including
functions for easy usage in main scripts
"""

import os
import sys
import numpy as np
from scipy.stats import entropy

from global_defs import CONFIG
from in_out import metrics_load, metrics_dump, probs_load, grads_load


def concat_metrics(loader):
    print('concat metrics')

    if os.path.isfile(CONFIG.METRICS_DIR+'metrics_all.p'):
        metrics =  metrics_load('all') 

    else:
        metrics =  metrics_load(loader[0][2]) 

        for item,i in zip(loader,range(len(loader))):
            if i == 0:
                continue
            sys.stdout.write("\t concatenated file number {} / {}\r".format(i,len(loader)))
            m = metrics_load(item[2]) 
            for j in metrics:
                metrics[j] += m[j]

        metrics_dump( metrics, 'all' )

    print(" ")
    print("components:", len(metrics['iou0']) )
    print("connected components:", np.sum( np.asarray(metrics['S']) != 0) )
    print("non-empty connected components:", np.sum( np.asarray(metrics['S_in']) != 0) )
    print("IoU = 0:", np.sum( np.asarray(metrics['iou0']) == 1) )
    print("IoU > 0:", np.sum( np.asarray(metrics['iou0']) == 0) )
    return metrics


def metrics_to_nparray( metrics, names, normalize=False, non_empty=True ):

    I = range(len(metrics['S_in']))
    if non_empty == True:
        I = np.asarray(metrics['S_in']) > 0
    M = np.asarray( [ np.asarray(metrics[ m ])[I] for m in names ] )
    MM = M.copy()
    if normalize == True:
        for i in range(M.shape[0]):
            if names[i] != "class":
                M[i] = ( np.asarray(M[i]) - np.mean(MM[i], axis=-1 ) ) / ( np.std(MM[i], axis=-1 ) + 1e-10 )
    M = np.squeeze(M.T)
    return M


def label_as_onehot(label, num_classes, shift_range=0):
    y = np.zeros((num_classes, label.shape[0], label.shape[1]))
    for c in range(shift_range,num_classes+shift_range):
        y[c-shift_range][label==c] = 1
    y = np.transpose(y,(1,2,0)) 
    return y.astype('uint8')


def classes_to_categorical( classes, nc = None ):
    classes = np.squeeze( np.asarray(classes) )
    if nc == None:
        nc      = np.max(classes)
    classes = label_as_onehot( classes.reshape( (classes.shape[0],1) ), nc ).reshape( (classes.shape[0], nc) )
    names   = [ "C_"+str(i) for i in range(nc) ]
    return classes, names


def metrics_to_dataset( metrics, non_empty=True, probs=True, layer_ntl=False ):

    if layer_ntl:
        X_names = sorted([ m for m in metrics if m not in ["index","class","iou0","iou"] and "cprob" not in m and ('G' not in m or 'ntl_G' in m)])
    else:
        X_names = sorted([ m for m in metrics if m not in ["index","class","iou0","iou"] and "cprob" not in m and 'ntl_G' not in m])

    nclasses = np.max( metrics["class"] ) + 1
    if probs:
        class_names = [ "cprob"+str(i) for i in range(nclasses) if "cprob"+str(i) in metrics ]
    else:
        class_names = ["class"]

    Xa      = metrics_to_nparray( metrics, X_names    , normalize=True , non_empty=non_empty )
    classes = metrics_to_nparray( metrics, class_names, normalize=True, non_empty=non_empty )
    ya      = metrics_to_nparray( metrics, ["iou" ]   , normalize=False, non_empty=non_empty )
    y0a     = metrics_to_nparray( metrics, ["iou0"]   , normalize=False, non_empty=non_empty )
    
    if not probs:
        classes, class_names = classes_to_categorical( classes, nclasses )
  
    Xa = np.concatenate( (Xa,classes), axis=-1 )
    X_names += class_names
    print(X_names)
    return Xa, np.squeeze(ya), np.squeeze(y0a), X_names  


def init_stats(X_names, baselines):
    """
    initialize dataframe for storing results
    """
    num_baselines = len(baselines)

    stats = dict({})
    single_stats = ['val_acc','val_auroc','train_acc','train_auroc','val_mse', 'val_r2', 'train_mse', 'train_r2']

    for s in single_stats:
        stats[s] = np.zeros((num_baselines,CONFIG.NUM_RUNS,))
            
    stats["coefs_classif"] = np.zeros((num_baselines,CONFIG.NUM_RUNS,len(X_names) ))
    stats["coefs_regress"] = np.zeros((num_baselines,CONFIG.NUM_RUNS,len(X_names) ))
    stats["n_av"]         = CONFIG.NUM_RUNS
    stats["n_metrics"]    = len(X_names) 
    stats["metric_names"] = X_names
    stats["methods"] = baselines
            
    return stats


def select_metrics(bl, X_names):
    """
    returns indizes of selected metrics
    """
    idx = np.zeros((len(X_names)),dtype='bool')

    if bl == 'max_softmax':
        for i in range(len(X_names)):
            if 'cprob' in X_names[i]:
                idx[i] = 1
    elif bl == 'entropy': 
        for i in range(len(X_names)):
            if X_names[i] == "E":
                idx[i] = 1
    elif bl == 'grads_01': 
        for i in range(len(X_names)):
            if 'G0' in X_names[i]:
                idx[i] = 1
    elif bl == 'grads_03': 
        for i in range(len(X_names)):
            if 'G1' in X_names[i]:
                idx[i] = 1
    elif bl == 'grads_05': 
        for i in range(len(X_names)):
            if 'G2' in X_names[i]:
                idx[i] = 1
    elif bl == 'grads_1': 
        for i in range(len(X_names)):
            if 'G3' in X_names[i]:
                idx[i] = 1
    elif bl == 'grads_2': 
        for i in range(len(X_names)):
            if 'G4' in X_names[i]:
                idx[i] = 1
    elif bl == 'grads_01u': 
        for i in range(len(X_names)):
            if 'G5' in X_names[i]:
                idx[i] = 1
    elif bl == 'grads_03u': 
        for i in range(len(X_names)):
            if 'G6' in X_names[i]:
                idx[i] = 1
    elif bl == 'grads_05u': 
        for i in range(len(X_names)):
            if 'G7' in X_names[i]:
                idx[i] = 1
    elif bl == 'grads_1u': 
        for i in range(len(X_names)):
            if 'G8' in X_names[i]:
                idx[i] = 1
    elif bl == 'grads_2u': 
        for i in range(len(X_names)):
            if 'G9' in X_names[i]:
                idx[i] = 1
    elif bl == 'grads_oh': 
        for i in range(len(X_names)):
            if 'G0' in X_names[i] or 'G1' in X_names[i] or 'G2' in X_names[i] or 'G3' in X_names[i] or 'G4' in X_names[i]:
                idx[i] = 1
    elif bl == 'grads_uni': 
        for i in range(len(X_names)):
            if 'G5' in X_names[i] or 'G6' in X_names[i] or 'G7' in X_names[i] or 'G8' in X_names[i] or 'G9' in X_names[i]:
                idx[i] = 1
    elif bl == 'grads': 
        for i in range(len(X_names)):
            if 'G' in X_names[i]:
                idx[i] = 1
    elif bl == 'MetaSeg':
        for i in range(len(X_names)):
            if 'G' not in X_names[i]:
                idx[i] = 1
    elif bl == 'all':
        idx = ~idx
    return idx 


def split_data_randomly( Xa, ya, y0a, seed ):
    """
    create random data split 80/20 for training and validation
    """

    np.random.seed( seed )
    val_mask = np.random.rand(len(ya)) < 0.2
    Xa_val = Xa[val_mask]
    ya_val = ya[val_mask]
    y0a_val = y0a[val_mask]
    Xa_train = Xa[np.logical_not(val_mask)]
    ya_train = ya[np.logical_not(val_mask)]
    y0a_train = y0a[np.logical_not(val_mask)]
    return Xa_val, ya_val, y0a_val, Xa_train, ya_train, y0a_train


def concat_unc_metrics(loader):

    nclasses = probs_load(loader[0][3]).shape[0]
    num_grads = grads_load(loader[0][3]).shape[0]

    pred = []
    gt = []
    squared_error = []
    unc_max_soft = []
    unc_entropy = []
    unc_grad = {}
    for i in range(num_grads):  
        unc_grad['grad'+str(i)] = list([])

    for item in loader:
        print(item[2])

        probs = probs_load(item[3])
        seg = np.asarray( np.argmax(probs, axis=0)) 
        grads = grads_load(item[3])

        pred.append(seg)
        gt.append(item[1])
        unc_max_soft.append(np.max(probs,0))
        unc_entropy.append(entropy(probs, axis=0))
        for i in range(num_grads):  
            unc_grad['grad'+str(i)].append(grads[i])

        gt_se = (item[1].copy()).reshape(-1)
        gt_se[gt_se==255] = 0
        probs = probs.reshape(nclasses,-1)
        one_hot_targets = np.eye(nclasses)[gt_se]
        one_hot_targets = np.swapaxes(one_hot_targets, 0, 1)
        se = np.sum(np.power(one_hot_targets-probs, 2), 0)
        se = se.reshape(seg.shape[0],seg.shape[1])
        squared_error.append(se)
        
    pred = np.stack(pred, 0)
    gt = np.stack(gt, 0)
    squared_error = np.stack(squared_error, 0)
    unc_max_soft = np.stack(unc_max_soft, 0)
    unc_entropy = np.stack(unc_entropy, 0)
    for i in range(num_grads):
        unc_grad['grad'+str(i)] = np.stack(unc_grad['grad'+str(i)], 0)

    np.save(CONFIG.EVAL_UNC_DIR+'pred.npy', pred)
    np.save(CONFIG.EVAL_UNC_DIR+'gt.npy', gt)
    np.save(CONFIG.EVAL_UNC_DIR+'squared_error.npy', squared_error)
    np.save(CONFIG.EVAL_UNC_DIR+'unc_max_soft.npy', unc_max_soft)
    np.save(CONFIG.EVAL_UNC_DIR+'unc_entropy.npy', unc_entropy)
    for i in range(num_grads):
        np.save(CONFIG.EVAL_UNC_DIR+'unc_grad'+str(i)+'.npy', unc_grad['grad'+str(i)])
    
    return num_grads

    
def name_to_latex( name ):
  
    for i in range(100):
        if name == "cprob"+str(i):
            return "$C_{"+str(i)+"}$"

    mapping =  {'E': '$\\bar E$',
                'E_bd': '${\\bar E}_{bd}$',
                'E_in': '${\\bar E}_{in}$',
                'E_rel_in': '$\\tilde{\\bar E}_{in}$',
                'E_rel': '$\\tilde{\\bar E}$',
                'M': '$\\bar M$',
                'M_bd': '${\\bar M}_{bd}$',
                'M_in': '${\\bar M}_{in}$',
                'M_rel_in': '$\\tilde{\\bar M}_{in}$',
                'M_rel': '$\\tilde{\\bar M}$',
                'S': '$S$',
                'S_bd': '${S}_{bd}$',
                'S_in': '${S}_{in}$',
                'S_rel_in': '$\\tilde{S}_{in}$',
                'S_rel': '$\\tilde{S}$',
                'C_p' : '${C}_{p}$',
                'iou' : '$IoU_{adj}$'}        
    if str(name) in mapping:
        return mapping[str(name)]
    else:
        return str(name)

