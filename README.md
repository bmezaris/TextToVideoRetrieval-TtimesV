# Are all combinations equal? Combining textual and visual features with multiple space learning for text-based video retrieval

## Pytorch Implementation of the $T \times V$ model, presented at the CVEU workshop@ECCV 2022. Based on previous [ATT-ATV](https://github.com/bmezaris/AVS_dual_encoding_attention_network) and [SEA](https://github.com/li-xirong/sea) implementations. 

- From D. Galanopoulos, V. Mezaris, **"Are all combinations equal? Combining textual and visual features with multiple space learning for text-based video retrieval"**, Proc. European Conference on Computer Vision Workshops (ECCVW), Oct. 2022.
- Text-based video retrieval software. The datasets of the Ad-hoc Video Search (AVS) Task of NIST's TRECVID (a typical benchmarking activity for evaluating such methods) are used for evaluation.
- The software provided in the present repository can be used for training the proposed $T \times V$ model, using multiple textual and visual features.

## Main dependencies
Developed, checked, and verified on an `Ubuntu 20.04.3` PC with an `NVIDIA RTX3090` GPU. Main packages required:
|`Python` | `PyTorch` | `CUDA Version` | `cuDNN Version` | `PyTorch-Transformers` | `NumPy` 
:---:|:---:|:---:|:---:|:---:|:---:|
3.7(.13) | 1.12.0 | 11.3 | 8320 | 1.12 | 1.21 |

## Data

### Datasets
In our AVS experiments, the proposed $T \times V$ model is trained using a combination of four large-scale video captioning datasets: [MSR-VTT](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/), [TGIF](https://github.com/raingo/TGIF-Release), [ActivityNet](https://cs.stanford.edu/people/ranjaykrishna/densevid/) and [Vatex](https://eric-xw.github.io/vatex-website/index.html). For validation purposes, during training, we use the Video-to-Text
Description dataset of TRECVID 2016 (tv2016train). To evaluate the performance, the IACC.3 and V3C1 TRECVID AVS datasets are used. To download TRECVID datasets please refer to the original [TRECVID AVS](https://www-nlpir.nist.gov/projects/tv2022/avs.html) page.

### Frame-level video features
We assume that frame-level video features have been extracted. In our experiments, we utilized three different visual features:

1. ResNet-152 trained on Imagenet 11K
2. ResNeXt-101 32x16 model pre-trained on weakly-supervised data and finetuned on ImageNet
3. CLIP ViT-B/32


Please refer to the following to extract visual features.

1. ResNet-152 from the MXNet model zoo

```
wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-symbol.json
wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-0000.params
```
2. [ResNext101 WSL-Images](https://github.com/facebookresearch/WSL-Images)
3. [CLIP](https://github.com/openai/CLIP) 

Alternatively, some pre-calculated visual features for the MSRT-VTT, TGIF and tv2016train datasets (beware: these are different features from the above-described that we used!) can be downloaded from ["Ad-hoc Video Search GitHub repository"](https://github.com/li-xirong/avs).

### Data format
The features extracted as described above must be stored as a txt file of the following format (all features for all videos stored in a single txt; a separate txt file for each of the overall training, validation, evaluation datasets)):

```
<ShotID_1_frameID_01> <feat_1> <feat_2> ... <feat_N>
<ShotID_1_frameID_02> <feat_1> <feat_2> ... <feat_N>
.
.
.
<ShotID_M_frameID_K> <feat_1> <feat_2> ... <feat_N>
```

Our network reads the visual features in a binary format. In order to convert text files (e.g. file "id.feature.txt") into binary files, run the following code:

```
rootpath=$HOME/TtimesV

featname=resnext101_32x16d_wsl,flatten0_output,os
collection=tgif-msrvtt10k

dim=2048
resultdir=$rootpath/$collection/FeatureData/$featname
featurefile=${feat_dir}/id.feature.txt
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir

python util/get_frameInfo.py --collection $collection --feature $featname
```
Finally, for every dataset (training, validation or evaluation) the binary files should be stored as in the following structure example:
```
rootpath
└── TGIF_MSR_VTT_Activity_Vatex
    ├── FeatureData
    │   └── resnext101_32x16d_wsl,flatten0_output,os
    │       ├── feature.bin
    │       ├── id.txt
    │       ├── shape.txt
    │       └── video2frames.txt
    └── TextData
        └── TGIF_MSR_VTT_Activity_Vatex.caption.txt
```
## Training
To train a $T \times V$ model, please follow the steps below:

```
rootpath=$HOME/TtimesV

trainCollection=tgif-msrvtt10k
valCollection=tv2016train
testCollection=tv2016train

text_features=clip@att
visual_features=resnet152_imagenet11k,flatten0_output,os@resnext101_32x16d_wsl,flatten0_output,os@CLIP_ViT_B_32_output,os

n_caption=2
optimizer=adam
learning_rate=0.0001

CUDA_VISIBLE_DEVICES=0 python TtimesV_trainer.py $trainCollection $valCollection $testCollection --learning_rate $learning_rate --selected_text_feas $text_features --overwrite 1 --visual_feature $visual_features --n_caption $n_caption --optimizer $optimizer --num_epochs 20 --rootpath $rootpath --cv_name DG_TtimesV 
```
If training completed successfully you will see the created trained model `model_best.pth.tar` into the `logger_name` folder:

## Evaluation
To evaluate the trained model on the IACC.3 and V3C1 datasets for the TRECVID AVS 2016/2017/2018 and 2019/2020/2021 topics please follow the next steps:

```
rootpath=$HOME/TtimesV
evalpath=$rootpath
logger_name=$rootpath/

evalCollection=iacc.3
CUDA_VISIBLE_DEVICES=0 python TtimesV_iacc3_evaluation.py.py $evalCollection --evalpath $evalpath --rootpath $rootpath --logger_name $logger_name

evalCollection=v3c1
CUDA_VISIBLE_DEVICES=0 python TtimesV_V3C1_evaluation.py.py $evalCollection --evalpath $evalpath --rootpath $rootpath --logger_name $logger_name
```

The evaluation scripts produces files in the proper format for the `sample_eval.pl` evaluation script. The `sample_eval.pl` produces a file containing results according to various evaluation measures.
## Citation

If you find our work, code or models, useful in your work, please cite the following publication:

D. Galanopoulos, V. Mezaris, "<b>Are all combinations equal? Combining textual and visual features with multiple space learning for text-based video retrieval</b>", Proc. European Conference on Computer Vision Workshops (ECCVW), Oct. 2022.

BibTeX:

```
@inproceedings{gal2022eccvw,
    author    = {Galanopoulos, Damianos and Mezaris, Vasileios},
    title     = {Are all combinations equal? Combining textual and visual features with multiple space learning for text-based video retrieval},
    booktitle = {European Conference on Computer Vision Workshops},
    month     = {October},
    year      = {2022},
    organization={Springer}
}
```

## Acknowledgements
This work was supported by the EU Horizon 2020 programme under grant agreements H2020-101021866 CRiTERIA and H2020-832921 MIRROR.
