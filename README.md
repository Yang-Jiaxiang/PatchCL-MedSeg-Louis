# PatchCL-MedSeg-Louis

This code is based on the paper:
[Pseudo-Label Guided Contrastive Learning for Semi-Supervised Medical Image Segmentation](https://ieeexplore.ieee.org/document/10205303)
H. Basak and Z. Yin, "Pseudo-Label Guided Contrastive Learning for Semi-Supervised Medical Image Segmentation," 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, BC, Canada, 2023, pp. 19786-19797, doi: 10.1109/CVPR52729.2023.01895.

The training steps are implemented using the ST++ method from the paper:
[ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation](https://arxiv.org/abs/2106.05095)
Lihe Yang, Wei Zhuo, Lei Qi, Yinghuan Shi, Yang Gao, Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

This code uses the Deeplabv3+ architecture with Resnet18 as the backbone.

## Dataset

The code utilizes the binary classification Pascal VOC dataset.

## Usage

Modify the `base_path` parameter in `src/main.py` to set the execution path.

To execute the `main.py` file, use the following command:

```sh
python main.py \
--dataset_path '/home/u5169119/dataset/0_data_dataset_voc_950_kidney' \
--output_dir 'dataset/splits/kidney' \
--patch_size 7 --embedding_size 128 \
--img_size 224 --batch_size 16 --num_classes 2 \
--ContrastiveWeights 0.1 --save_interval 2
