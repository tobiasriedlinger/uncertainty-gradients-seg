#!/usr/bin/env python3
"""
script including
class objects called in main
"""

import os
import numpy as np
import concurrent.futures
from sklearn.metrics import r2_score

from global_defs import CONFIG
from prepare_data import Cityscapes
from plot import vis_pred_i, vis_meta_pred_i, plot_scatter, plot_coefs, plot_baseline
from calculate import comp_metrics_i, regression_fit_and_predict, compute_correlations, fit_model_run, comp_metrics_cali
from helper import concat_metrics, metrics_to_dataset, init_stats, select_metrics, split_data_randomly, concat_unc_metrics
from in_out import stats_dump


def main():
    """
    From this line on, it is assumed that:
    - IMG_DIR defined in "global_defs.py" contains 3D input images
    - GT_DIR contains 2D arrays with semantic segmentation ground truth class 
    - PREDS_DIR contains 3D semantic segmentation softmax predictions + 1&2 norm of gradient calculation (in the two last dimensions)
    """

    """
    Load data
    """
    print('load dataset')

    if CONFIG.DATASET == 'cityscapes':
        loader = Cityscapes()

    print('dataset:', CONFIG.DATASET)
    print('number of images: ', len(loader))
    print('semantic segmentation network:', CONFIG.MODEL_NAME)

    """
    For visualizing the input data and predictions.
    """
    if CONFIG.VISUALIZE_PRED:
        print("visualize input data and predictions")

        if not os.path.exists(CONFIG.VIS_PRED_DIR):
            os.makedirs(CONFIG.VIS_PRED_DIR)

        for i in range(len(loader)):
            vis_pred_i(loader[i])

    """
    Calculation of segments and metrics
    """
    if CONFIG.COMPUTE_METRICS:
        print("compute segments and metrics")

        if not os.path.exists(CONFIG.COMPONENTS_DIR):
            os.makedirs(CONFIG.COMPONENTS_DIR)
        if not os.path.exists(CONFIG.METRICS_DIR):
            os.makedirs(CONFIG.METRICS_DIR)

        if CONFIG.NUM_CORES == 1:
            for i in range(len(loader)):
                comp_metrics_i(loader[i], i)
        else:
            p_args = [(loader[i], i) for i in range(len(loader))]
            with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG.NUM_CORES) as executor:
                executor.map(comp_metrics_i, *zip(*p_args))

    """
    For visualizing the input data and predictions
    """
    if CONFIG.VISUALIZE_META_PRED:
        print("visualize meta regression")

        if not os.path.exists(CONFIG.VIS_META_PRED_DIR):
            os.makedirs(CONFIG.VIS_META_PRED_DIR)

        metrics = concat_metrics(loader)
        Xa, ya, _, _ = metrics_to_dataset(metrics, non_empty=False)
        indizes = np.asarray(metrics['index'])

        print('train meta model')
        runs = 5  # train/val splitting of 80/20
        ya_pred = np.zeros((len(ya)))

        split = np.random.random_integers(0, runs-1, len(ya))
        for i in range(runs):
            print('run:', i)

            ya_pred_i, _, _ = regression_fit_and_predict(
                Xa[split != i, :], ya[split != i], Xa[split == i, :], ya[split == i])
            ya_pred[split == i] = ya_pred_i

        print("model overall test r2 score:", r2_score(ya, ya_pred))

        if CONFIG.NUM_CORES == 1:
            for i in range(len(loader)):
                vis_meta_pred_i(
                    loader[i], ya[indizes == i], ya_pred[indizes == i])
                if i == 10:
                    exit()
        else:
            p_args = [(loader[i], ya[indizes == i], ya_pred[indizes == i])
                      for i in range(len(loader))]
            with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG.NUM_CORES) as executor:
                executor.map(vis_meta_pred_i, *zip(*p_args))

    """
    Perform meta prediction
    """
    if CONFIG.META_PRED:
        print("compute meta predictions")

        baselines = ['max_softmax', 'entropy', 'grads_01', 'grads_03', 'grads_05', 'grads_1', 'grads_2', 'grads_01u',
                     'grads_03u', 'grads_05u', 'grads_1u', 'grads_2u', 'grads_oh', 'grads_uni', 'grads', 'MetaSeg', 'all']

        if not os.path.exists(os.path.join(CONFIG.META_PRED_DIR, CONFIG.META_MODEL)):
            os.makedirs(os.path.join(CONFIG.META_PRED_DIR, CONFIG.META_MODEL))

        metrics = concat_metrics(loader)
        Xa, ya, y0a, X_names = metrics_to_dataset(
            metrics, non_empty=False, layer_ntl=True)
        print(np.shape(Xa), np.shape(ya))

        print("making iou scatterplot(s) ...")
        df_all, df_full = compute_correlations(metrics)
        for name in X_names:
            if '_' not in name:
                plot_scatter(df_full, name)

        print('start runs')
        stats = init_stats(X_names, baselines)
        for bl in range(len(baselines)):
            print(baselines[bl])
            idx = select_metrics(baselines[bl], X_names)
            Xa_sel = Xa[:, idx]
            if baselines[bl] == 'max_softmax':
                Xa_sel = np.expand_dims(np.max(Xa_sel, -1), -1)

            for run in range(CONFIG.NUM_RUNS):
                print("run", run)
                Xa_val, ya_val, y0a_val, Xa_train, ya_train, y0a_train = split_data_randomly(
                    Xa_sel, ya, y0a, seed=run)
                stats = fit_model_run(Xa_val, ya_val, y0a_val, Xa_train,
                                      ya_train, y0a_train, stats, run, bl, baselines[bl], X_names)

        mean_stats_all, std_stats_all = stats_dump(
            stats, df_all, y0a, baselines)

        plot_coefs(mean_stats_all[-1]["coefs_classif"],
                   X_names, 'classification')
        plot_coefs(mean_stats_all[-1]["coefs_regress"], X_names, 'regression')
        plot_baseline(mean_stats_all, std_stats_all, baselines)

    """
    Evalute uncertainty
    """
    if CONFIG.EVAL_UNC:
        print("evalute uncertainty")

        if not os.path.exists(CONFIG.EVAL_UNC_DIR):
            os.makedirs(CONFIG.EVAL_UNC_DIR)

        num_grads = concat_unc_metrics(loader)
        comp_metrics_cali(num_grads)


if __name__ == '__main__':

    print("===== START =====")
    main()
    print("===== DONE! =====")
