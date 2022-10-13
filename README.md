# Are all combinations equal? Combining textual and visual features with multiple space learning for text-based video retrieval

## Pytorch Implementation of the $T \times V$ model, presented at the CVEU workshop@ECCV 2022. Based on previous [ATT-ATV](https://github.com/bmezaris/AVS_dual_encoding_attention_network) and [SEA](https://github.com/li-xirong/sea) implementations. 

- From D. Galanopoulos, V. Mezaris, **"Are all combinations equal? Combining textual and visual features with multiple space learning for text-based video retrieval"**, Proc. European Conference on Computer Vision Workshops (ECCVW), Oct. 2022. https://cveu.github.io/2022/papers/0010.pdf
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

In our MSR-VTT experiments, we experimented with two versions of this dataset: MST-VTT-full and MSR-VTT-1k-A. For both versions, the proposed $T \times V$ model is trained on the training portion of the dataset and evaluated on the testing portion, respectively.

### Frame-level video features
We assume that frame-level video features have been extracted. In our experiments, we utilized three different visual features:

1. ResNet-152 trained on Imagenet 11K
2. ResNeXt-101 32x16 model pre-trained on weakly-supervised data and finetuned on ImageNet
3. CLIP ViT-B/32


Please refer to the following to extract visual features:

1. ResNet-152 from the MXNet model zoo

```
wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-symbol.json
wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-0000.params
```
2. [ResNext101 WSL-Images](https://github.com/facebookresearch/WSL-Images)
3. [CLIP](https://github.com/openai/CLIP) 

Alternatively, some pre-calculated visual features for the MSRT-VTT, TGIF and tv2016train datasets (beware: these are different features from the above-described that we used!) can be downloaded from ["Ad-hoc Video Search GitHub repository"](https://github.com/li-xirong/avs).

### Data format
Every frame-level video feature extracted as described above must be stored as a separate txt file of the following format:

```
<ShotID_1_frameID_01> <dim_1> <dim_2> ... <dim_N>
<ShotID_1_frameID_02> <dim_1> <dim_2> ... <dim_N>
.
.
.
<ShotID_M_frameID_K> <dim_1> <dim_2> ... <dim_N>
```
Also, different txt files must be created for each available dataset (e.g., the overall training collection, validation, and evaluation).

Our network reads the visual features in a binary format. The script below converts a text file into binary format. In this example the text file `id.feature.txt`, which contains the frame-level features `resnext101_32x16d_wsl,flatten0_output,os` with dimension `N=2048`  from the training dataset `tgif_msr-vtt_activity_vatex` is converted into the binary format.

```
rootpath=$HOME/TtimesV

featname=resnext101_32x16d_wsl,flatten0_output,os
collection=tgif_msr-vtt_activity_vatex

N=2048
resultdir=$rootpath/$collection/FeatureData/$featname
featurefile=${resultdir}/id.feature.txt
python simpleknn/txt2bin.py $N $featurefile 0 $resultdir

python util/get_frameInfo.py --collection $collection --feature $featname
```
The successful execution of the above script produces the following files:  `feature.bin`, `id.txt`, `shape.txt`, and `video2frames.txt`

For every dataset, a txt file `<collection>.caption.txt` with the captions of every video shot should be created and stored in the `TextData` folder in the following format:

```
<ShotID_1>#enc#<cap_id> <caption text>
<ShotID_1>#enc#<cap_id> <caption text>
.
.
.
<ShotID_M>#enc#<cap_id> <caption text>
```

Finally, for every dataset (training, validation or evaluation) the required files should be stored as in the following structure example:
```
rootpath
└── tgif_msr-vtt_activity_vatex
    ├── FeatureData
    │   └── resnext101_32x16d_wsl,flatten0_output,os
    │       ├── feature.bin
    │       ├── id.txt
    │       ├── shape.txt
    │       └── video2frames.txt
    └── TextData
        └── tgif_msr-vtt_activity_vatex.caption.txt
```

## Training
To train a $T \times V$ model, please follow the steps below:

```
rootpath=$HOME/TtimesV

trainCollection=tgif_msr-vtt_activity_vatex
valCollection=tv2016train
testCollection=tv2016train

text_features=clip@att
visual_features=resnet152_imagenet11k,flatten0_output,os@resnext101_32x16d_wsl,flatten0_output,os@CLIP_ViT_B_32_output,os

n_caption=2
optimizer=adam
learning_rate=0.0001

CUDA_VISIBLE_DEVICES=0 python TtimesV_trainer.py $trainCollection $valCollection $testCollection --learning_rate $learning_rate --selected_text_feas $text_features --overwrite 1 --visual_feature $visual_features --n_caption $n_caption --optimizer $optimizer --num_epochs 20 --rootpath $rootpath --cv_name DG_TtimesV 
```
Please refer to the arguments of the `TtimesV_trainer.py` file to change model and training parameters.

If training is completed successfully you will see the created trained model `model_best.pth.tar` into the `logger_name` folder.

To train a $T \times V$ model for the MSR-VTT datasets, please change the `trainCollection` and `testCollection` variables to match with the MSR-VTT training and testing datasets.

Please note that in [1] we train our network using six configurations of the same architecture with different training parameters, and then we combine the results of the six configurations. Specifically, each model is trained using two optimizers, i.e., Adam and RMSprop, and three learning rates ( $1\times10^4$, $5\times10^5$, $1\times10^5$ ). 

## Evaluation
To evaluate a trained model on the IACC.3 and V3C1 datasets for the TRECVID AVS 2016/2017/2018 and 2019/2020/2021 topics, you can follow the steps below:

```
rootpath=$HOME/TtimesV
evalpath=$rootpath
logger_name=$rootpath/<the path where the `model_best.pth.tar` is stored>

evalCollection=iacc.3
CUDA_VISIBLE_DEVICES=0 python TtimesV_iacc3_evaluation.py.py $evalCollection --evalpath $evalpath --rootpath $rootpath --logger_name $logger_name

evalCollection=v3c1
CUDA_VISIBLE_DEVICES=0 python TtimesV_V3C1_evaluation.py.py $evalCollection --evalpath $evalpath --rootpath $rootpath --logger_name $logger_name
```

The evaluation scripts produce results files in the correct format for subsequently processing with the `sample_eval.pl` evaluation script. The `sample_eval.pl` will produce a file reporting the overal results according to various evaluation measures.

Similarly, to evaluate a MSR-VTT-trained model on the `MSR-VTT` testing datasets, you can follow the steps below:

```
rootpath=$HOME/TtimesV
evalpath=$rootpath
logger_name=$rootpath/<the path where the `model_best.pth.tar` is stored>
n_caption=1

evalCollection=MSR_VTT_1k-A_test
CUDA_VISIBLE_DEVICES=0 python TtimesV_tester.py $evalCollection --evalpath $evalpath --rootpath $rootpath --logger_name $logger_name --n_caption $n_caption
 ```

## License

This code/materials is provided for academic, non-commercial use only, to ensure timely dissemination of scholarly and technical work. The code/materials is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this code/materials, even if advised of the possibility of such damage.

## Citation

If you find our work, code or models, useful in your work, please cite the following publication:

[1] D. Galanopoulos, V. Mezaris, "<b>Are all combinations equal? Combining textual and visual features with multiple space learning for text-based video retrieval</b>", Proc. European Conference on Computer Vision Workshops (ECCVW), Oct. 2022.

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
