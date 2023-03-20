#!/usr/bin/python
#
# Cityscapes labels
#

from collections import namedtuple

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# label and all information

Label = namedtuple('Label',['name','Id','trainId','category', 'category_id','color'])

#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

cs_labels = [
    #       name                Id   trainId         category  category_id           color
    Label(  'unlabeled'     ,    0 ,     255 ,         'void' ,          0 , (255,255,255) ),
    Label(  'road'          ,    7 ,       0 ,         'flat' ,          1 , (128, 64,128) ),
    Label(  'sidewalk'      ,    8 ,       1 ,         'flat' ,          1 , (244, 35,232) ),
    Label(  'building'      ,   11 ,       2 , 'construction' ,          2 , ( 70, 70, 70) ),
    Label(  'wall'          ,   12 ,       3 , 'construction' ,          2 , (102,102,156) ),
    Label(  'fence'         ,   13 ,       4 , 'construction' ,          2 , (190,153,153) ),
    Label(  'pole'          ,   17 ,       5 ,       'object' ,          3 , (153,153,153) ),
    Label(  'traffic light' ,   19 ,       6 ,       'object' ,          3 , (250,170, 30) ),
    Label(  'traffic sign'  ,   20 ,       7 ,       'object' ,          3 , (220,220,  0) ),
    Label(  'vegetation'    ,   21 ,       8 ,       'nature' ,          4 , (107,142, 35) ),
    Label(  'terrain'       ,   22 ,       9 ,       'nature' ,          4 , (152,251,152) ),
    Label(  'sky'           ,   23 ,      10 ,          'sky' ,          5 , ( 70,130,180) ),
    Label(  'person'        ,   24 ,      11 ,        'human' ,          6 , (220, 20, 60) ),
    Label(  'rider'         ,   25 ,      12 ,        'human' ,          6 , (255,  0,  0) ),
    Label(  'car'           ,   26 ,      13 ,      'vehicle' ,          7 , (  0,  0,142) ),
    Label(  'truck'         ,   27 ,      14 ,      'vehicle' ,          7 , (  0,  0, 70) ),
    Label(  'bus'           ,   28 ,      15 ,      'vehicle' ,          7 , (  0, 60,100) ),
    Label(  'train'         ,   31 ,      16 ,      'vehicle' ,          7 , (  0, 80,100) ),
    Label(  'motorcycle'    ,   32 ,      17 ,      'vehicle' ,          7 , (  0,  0,230) ),
    Label(  'bicycle'       ,   33 ,      18 ,      'vehicle' ,          7 , (119, 11, 32) ),
    Label(  'license plate' ,   -1 ,      -1 ,      'vehicle' ,          7 , (  0,  0,142) ),
]

