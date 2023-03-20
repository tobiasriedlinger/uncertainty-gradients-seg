#!/usr/bin/env python3
"""
script including
functions that do calculations
"""

import time
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import entropy
from sklearn import linear_model
from skimage import measure as ms
from sklearn.metrics import r2_score, roc_curve, auc, mean_squared_error
from skimage.segmentation import find_boundaries

from global_defs import CONFIG
from in_out import probs_load, grads_load, components_dump, metrics_dump
from plot import plot_regression_scatter


def comp_entropy( probs ):
    return entropy(probs, axis=0)
  

def variation_ratio( probs ):
    output = np.ones((probs.shape[1],probs.shape[2]))
    return output - np.sort(probs, axis=0)[-1,:,:]


def probdist( probs ):
    output = np.ones((probs.shape[1],probs.shape[2]))
    return output - np.sort(probs, axis=0)[-1,:,:] + np.sort(probs, axis=0)[-2,:,:]


def comp_metrics_i(item, idx):

    start = time.time()

    probs = probs_load(item[3])
    imc = probs.shape[0]

    seg = np.asarray( np.argmax(probs, axis=0)) 
    seg[item[1]==255] = 255
    segments = ms.label(seg, background=255) 
    segments_bd = segments.copy()
    tmp = find_boundaries(segments, connectivity=segments.ndim, mode='inner')
    segments_bd[tmp==1] *= -1
    components_dump( segments_bd, item[2] ) 
    gt_segments = ms.label(item[1], background=255)

    heatmaps = { "E": comp_entropy( probs ), "M": probdist( probs ) } 
    grads = grads_load(item[3])
    grad_half = int(grads.shape[0]/2)
    for i in range(grad_half):
        heatmaps['G'+str(i)] = grads[i]
    for i in range(grad_half,grads.shape[0]):
        heatmaps['ntl_G'+str(i-grad_half)] = grads[i]
    metrics = { "index": list([]), "iou": list([]), "iou0": list([]), "class": list([]) } 

    for m in list(heatmaps)+["S"]:
        metrics[m          ] = list([])
        metrics[m+"_in"    ] = list([])
        metrics[m+"_bd"    ] = list([])
        metrics[m+"_rel"   ] = list([])
        metrics[m+"_rel_in"] = list([])
        if m != "S" and m != "E" and m != "M":
            metrics[m+"_var"] = list([])
            metrics[m+"_var_in"] = list([])
            metrics[m+"_var_bd"] = list([])
            metrics[m+"_var_rel"] = list([])
            metrics[m+"_var_rel_in"] = list([])
    for c in range(imc):
        metrics['cprob'+str(c)] = list([])
    
    for i in range(1,np.unique(segments)[-1]+1):

        if np.sum(segments_bd==-i) == 0:
            segments_bd[segments_bd==i] = -i

        for m in metrics:
            metrics[m].append( 0 )
        
        metrics["index"][-1] = idx
    
        metrics["S_in"][-1] = np.sum(segments_bd==i)
        metrics["S_bd"][-1] = np.sum(segments_bd==-i)
        metrics["S"][-1] = metrics["S_in"][-1] + metrics["S_bd"][-1]
        metrics["S_rel"][-1] = metrics["S"][-1] / metrics["S_bd"][-1]
        metrics["S_rel_in"][-1] = metrics["S_in"][-1] / metrics["S_bd"][-1]

        for m in heatmaps:
            metrics[m+"_in"][-1] = np.sum(heatmaps[m][segments_bd==i])
            metrics[m+"_bd"][-1] = np.sum(heatmaps[m][segments_bd==-i])
            metrics[m][-1] = (metrics[m+"_in"][-1] + metrics[m+"_bd"][-1]) / metrics["S"][-1]
            if metrics["S_in"][-1] > 0:
                metrics[m+"_in"][-1] /= metrics["S_in"][-1]
            metrics[m+"_bd"][-1] /= metrics["S_bd"][-1]
            metrics[m+"_rel"][-1] = metrics[m][-1] * metrics["S_rel"][-1]
            metrics[m+"_rel_in"][-1] = metrics[m+"_in"][-1] * metrics["S_rel_in"][-1]

            if m != "E" and m != "M":
                metrics[m+"_var_in"][-1] = np.sum(heatmaps[m][segments_bd==i]**2)
                metrics[m+"_var_bd"][-1] = np.sum(heatmaps[m][segments_bd==i]**2)
                metrics[m+"_var"][-1] = (metrics[m+"_var_in"][-1] + metrics[m+"_var_bd"][-1]) / metrics["S"][-1] - metrics[m][-1]**2
                if metrics["S_in"][-1] > 0:
                    metrics[m+"_var_in"][-1] = metrics[m+"_var_in"][-1] / metrics["S_in"][-1] - metrics[m+"_in"][-1]**2
                metrics[m+"_var_bd"][-1] = metrics[m+"_var_bd"][-1] / metrics["S_bd"][-1] - metrics[m+"_bd"][-1]**2
                metrics[m+"_var_rel"][-1] = metrics[m+"_var"][-1] * metrics["S_rel"][-1]
                metrics[m+"_var_rel_in"][-1] = metrics[m+"_var_in"][-1] * metrics["S_rel_in"][-1]

        metrics["class"][-1] = seg[segments==i].max()

        for c in range(imc):
            metrics["cprob"+str(c)][-1] = np.sum( probs[c,segments==i] ) / metrics["S"][-1]

        gt_segments_class = gt_segments.copy()
        gt_segments_class[item[1]!=metrics["class"][-1]] = 0
        if item[1] != []:
            tp_loc = gt_segments_class[segments == i]
            gt_ind = np.unique(tp_loc[tp_loc != 0])
            intersection = len(tp_loc[np.isin(tp_loc, gt_ind)])
            adjustment = len(gt_segments_class[np.logical_and(~np.isin(segments, [0, i]), np.isin(gt_segments_class, gt_ind))])
            adjusted_union = np.sum(np.isin(gt_segments_class, gt_ind)) + np.sum(segments == i) - intersection - adjustment
            metrics["iou"][-1] = float(intersection / adjusted_union)
            metrics["iou0"][-1] = int(intersection == 0)
        else:
            metrics["iou"][-1] = None
            metrics["iou0"][-1] = None

    metrics_dump( metrics, item[2] )
    print('image', item[2], 'processed in {}s\r'.format( round(time.time()-start,4) ) )


def regression_fit_and_predict( X_train, y_train, X_test, y_test=[] ): 

    print('regresssion input:', np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test))
    if CONFIG.META_MODEL == 'gradientboosting':
        model = xgb.XGBRegressor(max_depth=3, colsample_bytree=0.5, n_estimators=50, reg_alpha=0.4, reg_lambda=0.4) 
    elif CONFIG.META_MODEL == 'linear':
        model = linear_model.LinearRegression() 
    model.fit(X_train,y_train)
    y_train_pred = np.clip( model.predict(X_train), 0, 1 )
    y_test_pred = np.clip( model.predict(X_test), 0, 1 )
    print("model train r2 score:", r2_score(y_train,y_train_pred) )
    print("model test r2 score:", r2_score(y_test,y_test_pred) )
    print(" ")
    return y_test_pred, y_train_pred, model


def classification_fit_and_predict( X_train, y_train, X_test, y_test=[] ):

    print('classification input:', np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test))
    if CONFIG.META_MODEL == 'gradientboosting':
        model = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, subsample=0.5, reg_alpha=0.5, reg_lambda=0.5) 
    elif CONFIG.META_MODEL == 'linear':
        model = linear_model.LogisticRegression(penalty='none', solver='lbfgs', max_iter=3000, tol=1e-3) 
    model.fit( X_train, y_train )

    y_train_pred = model.predict_proba(X_train)
    fpr, tpr, _ = roc_curve(y_train.astype(int),y_train_pred[:,1])
    print("model train auroc score:", auc(fpr, tpr) )
    y_test_pred = model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test.astype(int),y_test_pred[:,1])
    print("model test auroc score:", auc(fpr, tpr) )
    print(" ")
    return y_test_pred, y_train_pred, model


def compute_correlations( metrics ):
  
    pd.options.display.float_format = '{:,.5f}'.format
    df_full = pd.DataFrame( data=metrics )
    df_full = df_full.copy().drop(["class","iou0","index"], axis=1) 
    df_all  = df_full.copy()
    df_full = df_full.copy().loc[df_full['S_in'].to_numpy().nonzero()[0]]
    return df_all, df_full


def fit_model_run(Xa_val, ya_val, y0a_val, Xa_train, ya_train, y0a_train, single_run_stats, run, bl, sel_metrics, X_names ):
    """
    fit meta model for one random data split and store results in dataframe
    """

    # classification:
    y0a_val_pred, y0a_train_pred, model_classif = classification_fit_and_predict( Xa_train, y0a_train, Xa_val, y0a_val )

    single_run_stats['train_acc'][bl,run] = np.mean( np.argmax(y0a_train_pred,axis=-1)==y0a_train )
    single_run_stats['val_acc'][bl,run] = np.mean( np.argmax(y0a_val_pred,axis=-1)==y0a_val )

    fpr, tpr, _ = roc_curve(y0a_train, y0a_train_pred[:,1])
    single_run_stats['train_auroc'][bl,run] = auc(fpr, tpr)
    fpr, tpr, _ = roc_curve(y0a_val, y0a_val_pred[:,1])
    single_run_stats['val_auroc'][bl,run] = auc(fpr, tpr)

    if sel_metrics == 'all': 
        if CONFIG.META_MODEL == 'gradientboosting':
            single_run_stats['coefs_classif'][bl,run] = np.array(model_classif.feature_importances_)
        elif CONFIG.META_MODEL == 'linear':
            single_run_stats['coefs_classif'][bl,run] = np.array(model_classif.coef_)

    # regression:
    ya_val_pred, ya_train_pred, model_regress = regression_fit_and_predict( Xa_train, ya_train, Xa_val, ya_val )

    single_run_stats['train_mse'][bl,run] = np.sqrt( mean_squared_error(ya_train, ya_train_pred) )
    single_run_stats['val_mse'][bl,run] = np.sqrt( mean_squared_error(ya_val, ya_val_pred) )

    single_run_stats['train_r2'][bl,run]  = r2_score(ya_train, ya_train_pred)
    single_run_stats['val_r2'][bl,run]  = r2_score(ya_val, ya_val_pred)

    if sel_metrics == 'all': 
        if CONFIG.META_MODEL == 'gradientboosting':
            single_run_stats['coefs_regress'][bl,run] = np.array(model_regress.feature_importances_)
        elif CONFIG.META_MODEL == 'linear':
            single_run_stats['coefs_regress'][bl,run] = np.array(model_regress.coef_)
            
    if run == 0:
        plot_regression_scatter( Xa_val, ya_val, ya_val_pred, X_names, sel_metrics )
            
    return single_run_stats


def metric_ece(pred, gt, conf, num_bins=10):

    th_acc = []
    th_avg_conf = []
    th_items = []

    step = int(100/num_bins)
    ths = np.arange(step,100+step,step) 
    for t in ths:
        idx = np.logical_and(conf>=(t-step)/100, conf<t/100)
        th_acc.append( np.sum(np.logical_and(pred==gt,idx)) / np.sum(np.logical_and(gt!=255,idx)) )
        th_avg_conf.append( np.mean(conf[np.logical_and(gt!=255,idx)]) ) 
        th_items.append(np.sum(np.logical_and(gt!=255,idx)))

    th_acc = np.nan_to_num(th_acc)
    th_avg_conf = np.nan_to_num(th_avg_conf)

    diff = np.abs(th_acc-th_avg_conf)
    ece = np.sum(th_items * diff) / np.sum(th_items)
    return ece


def metric_ause(gt, squared_error, unc):

    gt = gt.reshape(-1)
    squared_error = squared_error.reshape(-1)
    unc = unc.reshape(-1)

    squared_error = squared_error[gt!=255]
    unc = unc[gt!=255]
    num_samples = unc.shape[0]

    sorted_idx_error = np.argsort(squared_error)
    sorted_idx_unc = np.argsort(unc)

    error_brier = []
    unc_brier = []

    steps = list(np.arange(0, 1, 0.01))
    for step in steps:
        error_brier.append( np.mean( squared_error[sorted_idx_error[0:int((1-step)*num_samples)]] ) )
        unc_brier.append( np.mean( unc[sorted_idx_unc[0:int((1-step)*num_samples)]] ) )
    
    error_brier_norm = error_brier/error_brier[0]
    unc_brier_norm = unc_brier/unc_brier[0]

    sparsi_error = unc_brier_norm - error_brier_norm
    ause = np.trapz(y=sparsi_error, x=steps)
    return ause


def comp_metrics_cali(num_grads):
    """
    metrics from Gustafsson2020
    """

    pred = np.load(CONFIG.EVAL_UNC_DIR+'pred.npy')
    gt = np.load(CONFIG.EVAL_UNC_DIR+'gt.npy')
    squared_error = np.load(CONFIG.EVAL_UNC_DIR+'squared_error.npy')
    unc_max_soft = np.load(CONFIG.EVAL_UNC_DIR+'unc_max_soft.npy')
    unc_entropy = np.load(CONFIG.EVAL_UNC_DIR+'unc_entropy.npy')
    unc_grad = {}
    for i in range(num_grads):
        unc_grad['grad'+str(i)] = np.load(CONFIG.EVAL_UNC_DIR+'unc_grad'+str(i)+'.npy', )

    unc_max_soft = (unc_max_soft - np.min(unc_max_soft)) / (np.max(unc_max_soft) - np.min(unc_max_soft))
    unc_entropy = (unc_entropy - np.min(unc_entropy)) / (np.max(unc_entropy) - np.min(unc_entropy))
    for i in range(num_grads):
        unc_grad['grad'+str(i)] = (unc_grad['grad'+str(i)] - np.min(unc_grad['grad'+str(i)])) / (np.max(unc_grad['grad'+str(i)]) - np.min(unc_grad['grad'+str(i)]))

    print('ECE:')
    print('maximum softmax', metric_ece(pred, gt, unc_max_soft))
    print('entropy', metric_ece(pred, gt, 1-unc_entropy))
    for i in range(num_grads):
        print('Gradients', str(i), metric_ece(pred, gt, 1-unc_grad['grad'+str(i)]))

    print('AUSE:')
    print('maximum softmax', metric_ause(gt, squared_error, 1-unc_max_soft))
    print('entropy', metric_ause(gt, squared_error, unc_entropy))
    for i in range(num_grads):
        print('Gradients', str(i), metric_ause(gt, squared_error, unc_grad['grad'+str(i)]))
