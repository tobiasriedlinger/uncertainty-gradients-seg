# Pixel-wise Gradient Uncertainty for Convolutional Neural Networks applied to Out-of-Distribution Segmentation

The present code was utilized to generate the results in "[Pixel-wise Gradient Uncertainty for Convolutional Neural Networks applied to Out-of-Distribution Segmentation](https://arxiv.org/abs/2303.06920)".

## Method implementation
The implementation of the DeepLabv3+ segmentation model (forked from https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet) can be found in the subfolder `segmentation`.
Gradient norms are computed in the forward pass within `segmentation/network/deepv3.py` in the classes `DeepV3Plus` (l.253) and `DeepWV3Plus` (l.349) as the final chunk of layers is replaced by a child class `MySequential` (ll.35-124) of `nn.Sequential`.
Norm computation is implemented in the method `grad_heatmap`.

The forward pass can otherwise be called with minimal alterations (concerning the number of pixel-wise output features) to the scripts `segmentation/demo_folder.py` or `segmentation/eval_gradients.py` provided by the original framework by NVIDIA (with the provided weights of the original repository).

## Evaluation for pixel-wise and segment-wise UQ
The scripts for pixel- and segment-wise uncertainty evaluations in Sec. 4.1 and 4.2 can be found in the subfolder `evaluation` which are run by calls within `evaluation/main.py`.
In order to run evaluation, make sure to correctly define the path variables in `evaluation/global_defs.py`.

## Evaluation for OoD segmentation
See official benchmark at
http://github.com/SegmentMeIfYouCan/road-anomaly-benchmark.
