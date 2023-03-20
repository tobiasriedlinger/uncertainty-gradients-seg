#!/usr/bin/env python3
"""
script including
functions for handling input/output like loading/saving
"""

import os
import pickle
import numpy as np

from global_defs import CONFIG
import labels as labels

trainId2label = { label.trainId : label for label in reversed(labels.cs_labels) }
category_id2label = { label.category_id : label for label in reversed(labels.cs_labels) }


def probs_load( path ):
    preds = np.load(path) 
    sem_seg = np.asarray(preds[:19,:,:],dtype=np.float32)
    return sem_seg

def grads_load( path ):
    preds = np.load(path) 
    grads = np.asarray(preds[19:,:,:],dtype=np.float32)
    return grads

def get_save_path_components_i( i ):
    return CONFIG.COMPONENTS_DIR + "components_" + str(i) +".p"

def components_dump( components, i ):
    dump_path = get_save_path_components_i( i )
    pickle.dump( components, open( dump_path, "wb" ) )

def components_load( i ):
    read_path = get_save_path_components_i( i )
    components = pickle.load( open( read_path, "rb" ) )
    return components

def get_save_path_metrics_i( i ):
    return CONFIG.METRICS_DIR + "metrics_" + str(i) +".p"

def metrics_dump( metrics, i ):
    dump_path = get_save_path_metrics_i( i )
    pickle.dump( metrics, open( dump_path, "wb" ) )

def metrics_load( i ):
    read_path = get_save_path_metrics_i( i )
    metrics = pickle.load( open( read_path, "rb" ) )
    return metrics


def stats_dump( stats, df_all, y0a, baselines ):

    pickle.dump( stats, open( os.path.join(CONFIG.META_PRED_DIR, CONFIG.META_MODEL, 'stats.p'), "wb" ) )

    df_full = df_all.copy().loc[df_all['S_in'].to_numpy().nonzero()[0]]
    iou_corrs = df_full.corr()["iou"]

    # dump stats latex ready
    with open(os.path.join(CONFIG.META_PRED_DIR, CONFIG.META_MODEL, 'av_results_.txt'), 'wt') as f:
    
        mean_stats_all = []
        std_stats_all = []

        for bl in range(len(baselines)):

            mean_stats = dict({})
            std_stats = dict({})
            for s in stats:
                if s not in ["n_av", "n_metrics", "metric_names", "methods"]:
                    mean_stats[s] = np.mean(stats[s][bl], axis=0)
                    std_stats[s]  = np.std( stats[s][bl], axis=0)
            
            mean_stats_all.append(mean_stats)
            std_stats_all.append(std_stats)
  
            print('metrics:', baselines[bl], file=f)
            print(" ", file=f )
            print( iou_corrs, file=f )
            print(" ", file=f )
            
            print("classification", file=f )
            print( "                             & train                &  val  \\\\ ", file= f)
            M = sorted([ s for s in mean_stats if 'acc' in s ])
            print( "ACC       ", end=" & ", file= f )
            for s in M: print( "${:.2f}\%".format(100*mean_stats[s])+"(\pm{:.2f}\%)$".format(100*std_stats[s]), end=" & ", file=f )
            print("   \\\\ ", file=f )
            
            M = sorted([ s for s in mean_stats if 'auroc' in s ])
            print( "AUROC     ", end=" & ", file= f )
            for s in M: print( "${:.2f}\%".format(100*mean_stats[s])+"(\pm{:.2f}\%)$".format(100*std_stats[s]), end=" & ", file=f )
            print("   \\\\ ", file=f )
            
            print(" ", file=f)
            print("regression", file=f)
            
            M = sorted([ s for s in mean_stats if 'mse' in s and 'entropy' not in s ])
            print( "$\sigma$  ", end=" & ", file= f )
            for s in M: print( "${:.3f}".format(mean_stats[s])+"(\pm{:.3f})$".format(std_stats[s]), end="    & ", file=f )
            print("   \\\\ ", file=f )
            
            M = sorted([ s for s in mean_stats if 'r2' in s and 'entropy' not in s ])
            print( "$R^2$     ", end=" & ", file= f )
            for s in M: print( "${:.2f}\%".format(100*mean_stats[s])+"(\pm{:.2f}\%)$".format(100*std_stats[s]), end=" & ", file=f )
            print("   \\\\ ", file=f )
            
            print(" ", file=f )
            M = sorted([ s for s in mean_stats if 'iou' in s ])          
            for s in M: print( s, ": {:.0f}".format(mean_stats[s])+"($\pm${:.0f})".format(std_stats[s]), file=f )
            print("IoU=0:", np.sum(y0a==1), "of", y0a.shape[0], "non-empty components", file=f )
            print("IoU>0:", np.sum(y0a==0), "of", y0a.shape[0], "non-empty components", file=f )
            print("total number of components: ", len(df_all), file=f )
            print(" ", file=f )
    
    return mean_stats_all, std_stats_all


  