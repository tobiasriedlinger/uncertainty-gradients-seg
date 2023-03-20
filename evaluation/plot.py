#!/usr/bin/env python3
"""
script including
functions for visualizations
"""

import os
import numpy as np 
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.stats import pearsonr

from global_defs import CONFIG
import labels as labels
from in_out import probs_load, grads_load, components_load
from helper import name_to_latex


trainId2label = { label.trainId : label for label in reversed(labels.cs_labels) }
category_id2label = { label.category_id : label for label in reversed(labels.cs_labels) }

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


def vis_pred_i(item, grad_idx = 0): 

    gt = item[1].copy()
    gtc   = np.asarray([ trainId2label[ gt[p,q]   ].color for p in range(gt.shape[0]) for q in range(gt.shape[1]) ])
    gtc   = gtc.reshape(item[0].shape)

    seg = np.asarray( np.argmax(probs_load(item[3]), axis=0), dtype='int' )
    segc = np.asarray([ trainId2label[ seg[p,q] ].color for p in range(seg.shape[0]) for q in range(seg.shape[1]) ])
    segc = segc.reshape(item[0].shape)
    
    I1 = gtc*0.6 + item[0]*0.4
    I2 = segc*0.6 + item[0]*0.4
    
    plt.imsave(CONFIG.VIS_PRED_DIR + item[2] + str(grad_idx) + '_tmp1.png', entropy(probs_load(item[3]), axis=0), cmap='inferno')
    I3 = np.asarray( Image.open(CONFIG.VIS_PRED_DIR + item[2] + str(grad_idx) + '_tmp1.png').convert('RGB') )
    os.remove(CONFIG.VIS_PRED_DIR + item[2] + str(grad_idx) + '_tmp1.png')

    plt.imsave(CONFIG.VIS_PRED_DIR + item[2] + str(grad_idx) + '_tmp2.png', grads_load(item[3])[grad_idx,:,:], cmap='inferno') # choose norm
    I4 = np.asarray( Image.open(CONFIG.VIS_PRED_DIR + item[2] + str(grad_idx) + '_tmp2.png').convert('RGB') )
    os.remove(CONFIG.VIS_PRED_DIR + item[2] + str(grad_idx) + '_tmp2.png')

    img12   = np.concatenate( (I1,I2), axis=1 )
    img34  = np.concatenate( (I3,I4), axis=1 )
    img   = np.concatenate( (img12,img34), axis=0 )
    image = Image.fromarray(img.astype('uint8'), 'RGB')
    image = image.resize((int(item[0].shape[1]/2),int(item[0].shape[0]/2)))
    image.save(CONFIG.VIS_PRED_DIR + item[2] + '_grad' + str(grad_idx) + '.png')
    plt.close()

    print('stored:', item[2]+'.png')


def visualize_segments(comp, metric):

    R = np.asarray(metric)
    R = 1-0.5*R
    G = np.asarray(metric)
    B = 0.3+0.35*np.asarray(metric)

    R = np.concatenate((R, np.asarray([0, 1])))
    G = np.concatenate((G, np.asarray([0, 1])))
    B = np.concatenate((B, np.asarray([0, 1])))

    components = np.asarray(comp.copy(), dtype='int16')
    components[components < 0] = len(R)-1
    components[components == 0] = len(R)

    img = np.zeros(components.shape+(3,))

    for v in range(len(R)):
        img[components==(v+1), 0] = R[v]
        img[components==(v+1), 1] = G[v]
        img[components==(v+1), 2] = B[v]

    img = np.asarray(255*img).astype('uint8')

    return img


def vis_meta_pred_i(item, ya_i, ya_pred_i):

    rgb_img = item[0]
    gt = item[1].copy()
    seg = np.asarray( np.argmax(probs_load(item[3]), axis=0), dtype='int' )

    I1 = rgb_img.copy()
    I2 = rgb_img.copy()

    for c in range(19):
        I1[gt==c,:] = np.asarray(trainId2label[c].color)
        I2[seg==c,:] = np.asarray(trainId2label[c].color)
    I1[gt==255,:] = np.asarray(trainId2label[255].color)
    I2[seg==255,:] = np.asarray(trainId2label[255].color)
    I1 = I1 * 0.6 + rgb_img * 0.4
    I2 = I2 * 0.6 + rgb_img * 0.4

    components = components_load(item[2])
    I3 = visualize_segments(components, ya_i)
    I3[gt == 255] = [255,255,255]
    I4 = visualize_segments(components, ya_pred_i)
    I4[gt == 255] = [255,255,255]

    img12   = np.concatenate( (I1,I2), axis=1 )
    img34  = np.concatenate( (I3,I4), axis=1 )
    img   = np.concatenate( (img12,img34), axis=0 )
    image = Image.fromarray(img.astype('uint8'), 'RGB')
    image = image.resize((rgb_img.shape[1],rgb_img.shape[0]))
    image.save(os.path.join(CONFIG.VIS_META_PRED_DIR, CONFIG.META_MODEL + '_' + item[2] + '.png'))
    print('stored:', item[2]+'.png')


def add_scatterplot_vs_iou(ious, sizes, dataset, shortname, size_fac, scale, setylim=True):
  
    rho = pearsonr(ious,dataset)
    plt.title(r"$\rho = {:.05f}$".format(rho[0]))
    plt.scatter(ious, dataset, s = sizes/np.max(sizes)*size_fac, linewidth=.5, c='cornflowerblue', edgecolors='royalblue', alpha=.25)
    plt.xlabel('$\mathit{IoU}_\mathrm{adj}$', labelpad=-10)
    plt.ylabel(name_to_latex(shortname), labelpad=-8)
    plt.xticks((0,1),fontsize=10*scale)
    plt.yticks((dataset.min(),dataset.max()),fontsize=10*scale)


def add_scatterplot_vs_entro(entros, sizes, dataset, shortname, size_fac, scale, setylim=True):
  
    rho = pearsonr(entros,dataset)
    plt.title(r"$\rho = {:.05f}$".format(rho[0]))
    plt.scatter(entros, dataset, s = sizes/np.max(sizes)*size_fac, linewidth=.5, c='cornflowerblue', edgecolors='royalblue', alpha=.25)
    plt.xlabel(name_to_latex(shortname.replace(shortname.split('_')[0],'E')), labelpad=-10)
    plt.ylabel(name_to_latex(shortname), labelpad=-8)
    plt.xticks((entros.min(),entros.max()),fontsize=10*scale)
    plt.yticks((dataset.min(),dataset.max()),fontsize=10*scale)
    

def plot_scatter( df_full, m='E' ):

    scale = .75
    size_fac = 50*scale
    
    plt.figure(figsize=(3,3), dpi=300)
    add_scatterplot_vs_iou(df_full['iou'], df_full['S'], df_full[m], m, size_fac, scale)
    plt.tight_layout(pad=1.0*scale, w_pad=0.5*scale, h_pad=1.5*scale)
    save_path = os.path.join(CONFIG.META_PRED_DIR, CONFIG.META_MODEL, 'iou_vs_'+m+'.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
    print("scatterplots saved to " + save_path)
    plt.close()

    if 'G' in m and not '_' in m:
        plt.figure(figsize=(3,3), dpi=300)
        add_scatterplot_vs_entro(df_full[m.replace(m.split('_')[0],'E')], df_full['S'], df_full[m], m, size_fac, scale)
        plt.tight_layout(pad=1.0*scale, w_pad=0.5*scale, h_pad=1.5*scale)
        save_path = os.path.join(CONFIG.META_PRED_DIR, CONFIG.META_MODEL, m.replace(m.split('_')[0],'E')+'_vs_'+m+'.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=400)
        print("scatterplots saved to " + save_path)
        plt.close()


def plot_regression_scatter( Xa_val, ya_val, ya_pred, X_names, sel_metrics ):   
  
    plt.figure(figsize=(3,3), dpi=300)
    plt.clf()
    x = np.arange(0., 1, .01)
    plt.plot( x, x, color='black' , alpha=0.5, linestyle='dashed')
    plt.scatter( ya_val, np.clip(ya_pred,0,1), s=5, linewidth=.5, c='paleturquoise', edgecolors='turquoise', alpha=0.25 )
    plt.xlabel('$\mathit{IoU}_\mathrm{adj}$')
    plt.ylabel('predicted $\mathit{IoU}_\mathrm{adj}$')
    plt.savefig(os.path.join(CONFIG.META_PRED_DIR, CONFIG.META_MODEL, 'regression_' + sel_metrics + '.png'), bbox_inches='tight', dpi=400)
    plt.close()


def plot_coefs( coefs, X_names, task, num_metrics=12 ):

    print('plot coefficients for', task)
    coefs_neg = coefs.copy()
    coefs_neg[coefs>0] *= -1
    idx = np.argsort(coefs_neg)
    plot_coefs = coefs[idx]
    plot_coefs = plot_coefs[:num_metrics]

    metric_names = []
    for i in range(num_metrics):
        metric_names.append(name_to_latex(X_names[idx[i]]))

    size_text = 12
    f1 = plt.figure(1) 
    plt.clf()        
    plt.scatter(np.arange(num_metrics), plot_coefs, color='purple', marker='o', alpha=0.7)  
    plt.ylabel('coefficients', fontsize=size_text)
    plt.xticks(np.arange(num_metrics), metric_names, fontsize=size_text, rotation = 90)
    plt.yticks(fontsize=size_text)
    f1.savefig(os.path.join(CONFIG.META_PRED_DIR, CONFIG.META_MODEL, 'coefs_' + task + '.png'), bbox_inches='tight', dpi=400) 
    plt.close()


def plot_baseline(mean_stats, std_stats, baselines):
    
    print('Plot baselines')
    size_font = 16
    num_approaches = np.arange(len(baselines))   
    dist = 0.05
    color_map = ['tab:blue', 'tab:olive', 'tab:cyan', 'tab:pink']
    baselines1 = np.zeros((len(baselines),2,2))
    for bl in range(len(baselines)):
        baselines1[bl,0,0] = mean_stats[bl]['val_auroc']
        baselines1[bl,0,1] = std_stats[bl]['val_auroc']
        baselines1[bl,1,0] = mean_stats[bl]['val_r2']
        baselines1[bl,1,1] = std_stats[bl]['val_r2']

    f1 = f1 = plt.figure(figsize=(7.2,5.8))
    plt.clf()
    label_tmp = '$\mathit{AUROC}$ '
    plt.errorbar(num_approaches-dist, baselines1[:,0,0], baselines1[:,0,1], color=color_map[0], linestyle='', marker='o', capsize=3, label=label_tmp, alpha=1)
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['legend.handlelength'] = 0
    plt.yticks(fontsize = size_font)
    plt.xticks(num_approaches, (baselines), fontsize=size_font, rotation = 90)
    plt.grid(True)
    plt.legend(prop={'size': 16})
    f1.savefig(os.path.join(CONFIG.META_PRED_DIR, CONFIG.META_MODEL, 'baseline_auroc.png'), bbox_inches='tight', dpi=400)
    plt.close()

    f1 = f1 = plt.figure(figsize=(7.2,5.8))
    plt.clf()
    label_tmp = '$R^2$ '
    plt.errorbar(num_approaches-dist, baselines1[:,1,0], baselines1[:,1,1], color=color_map[2], linestyle='', marker='o', capsize=3, label=label_tmp, alpha=1)
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['legend.handlelength'] = 0
    plt.yticks(fontsize = size_font)
    plt.xticks(num_approaches, (baselines), fontsize=size_font, rotation = 90)
    plt.grid(True)
    plt.legend(prop={'size': 16})
    f1.savefig(os.path.join(CONFIG.META_PRED_DIR, CONFIG.META_MODEL, 'baseline_r2.png'), bbox_inches='tight', dpi=400)
    plt.close()

