#!/usr/bin/env python3
'''
script including
class object with global settings
'''

class CONFIG:
  
    #---------------------#
    # set necessary paths #
    #---------------------#
  
    io_path   = '/home/user/'   # directory with inputs and outputs, i.e. saving and loading data
  
    #----------------------------#
    # paths for data preparation #
    #----------------------------#
    
    IMG_DIR   = '/home/user/images'  
    GT_DIR    = '/home/user/gt' 
    PREDS_DIR = '/home/user/preds/' 
    
    #------------------#
    # select or define #
    #------------------#
  
    datasets    = ['cityscapes'] 
    model_names = ['DeepLabV3Plus_WideResNet38','DeepLabV3Plus_SEResNeXt50'] 
    meta_models = ['linear','gradientboosting']
    
    DATASET    = datasets[0]    
    MODEL_NAME = model_names[0]  
    META_MODEL = meta_models[0]   

    #--------------------------------------------------------------------#
    # select tasks to be executed by setting boolean variable True/False #
    #--------------------------------------------------------------------#

    VISUALIZE_PRED      = False
    COMPUTE_METRICS     = False
    VISUALIZE_META_PRED = False
    META_PRED           = False
    EVAL_UNC            = False

    #-----------#
    # optionals #
    #-----------#
    
    NUM_CORES = 1
    NUM_RUNS = 5

    META_MODEL_DIR    = io_path + 'meta_model/'    + MODEL_NAME + '/'
    VIS_PRED_DIR      = io_path + 'vis_pred/'      + DATASET + '/' + MODEL_NAME + '/'
    COMPONENTS_DIR    = io_path + 'components/'    + DATASET + '/' + MODEL_NAME + '/'
    METRICS_DIR       = io_path + 'metrics/'       + DATASET + '/' + MODEL_NAME + '/'
    VIS_META_PRED_DIR = io_path + 'vis_meta_pred/' + DATASET + '/' + MODEL_NAME + '/' 
    META_PRED_DIR     = io_path + 'meta_pred/'     + DATASET + '/' + MODEL_NAME + '/' 
    EVAL_UNC_DIR      = io_path + 'eval_unc/'      + DATASET + '/' + MODEL_NAME + '/'


'''
In case of problems, feel free to contact
  
'''
