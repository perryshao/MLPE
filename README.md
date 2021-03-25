# MLPE(A Multi-level Network for Human Pose Estimation)

Although multi-person 2D pose estimation has made great progress in recent years, the challenges on various scales and occlusions in complex scenes are still remained to be solved. in this paper, we propose a novel multi-level pose estimation network (MLPE) to address the challenges, while most existing single-stage networks are not able to accurately predict the key points of the human body at different scales.. We first leverage a split attention module in the feature extraction stage to achieve cross-channel interaction for multi-level feature maps. A multi-level prediction network is then introducedto accommodate multi-level features to achieve a good trade-off between the global context information and spatial resolution. Finally, the transposed convolution is used to build a high-resolution fine-tuning network to accurately locate the key points. We have conducted extensive experiments on the challenging MS COCO dataset, which has proved the effectiveness of our proposed method.
## Evaluation results on COCO minival dataset
<center>

| Method | Base Model | Input Size | Feature  retention | Receptive field | Upsample | AP | AP medium | AP large |
|:------:|:--------:|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| a | ResNeSt-50 | 256x192 | × | 3 | BI | 70.1 | 66.5 | 76.7 |
| b | ResNeSt-50 | 256x192 | √ | 3 |    BI    | 70.2 | 66.6 |   77.1   |
| c | ResNeSt-50 | 256x192 | √ | 3 | Deconv | 70.4 | 66.9 | 77.1 |
| d | ResNeSt-50 | 256x192 | v | 1 | Deconv | 70.2 | 66.7 | 76.8 |
| e | ResNeSt-50 | 384x288 | √ | 3 | Deconv | 72.4 | 68.1 | 79.9 |
| f | ResNeSt-101 | 384x288 | v | 3 | Deconv | 73.7 | 69.7 | 81.0 |


**Note**: Ablation study of our method on COCO val2017 dataset. **BI** and **Deconv** represent two upsampling methods respectively, **BI** refers to bilinear interpolation, and **Deconv** refers to upsampling using transposed convolution.

##  Comparisons on COCO test-dev dataset



|      Method      | Backbone         | Input size  | AP       | AP50     | AP75     | APM      | APL      | AR       |
| :--------------: | ---------------- | ----------- | -------- | -------- | -------- | -------- | -------- | -------- |
|   Openpose[8]    | -                | -           | 61.8     | 84.9     | 67.5     | 57.1     | 68.2     | 66.5     |
|  PersonLab[31]   | -                | -           | 68.7     | 89.0     | 75.4     | 64.1     | 75.5     | 75.4     |
| MulitPsoeNet[32] | -                | -           | 69.6     | 86.3     | 76.6     | 65.0     | 76.3     | 73.5     |
| HigherHRNet[34]  | -                | -           | 70.5     | 89.3     | 77.2     | 66.6     | 75.8     | 74.9     |
|  Mask RCNN[36]   | ResNet-50-FPN    | -           | 63.1     | 87.3     | 68.7     | 57.8     | 71.4     | -        |
|    G-RMI[42]     | ResNet-101       | 353×257     | 64.9     | 85.5     | 71.3     | 62.3     | 70.0     | 69.7     |
|    G-RMI*[42]    | ResNet-101       | 353×257     | 68.5     | 87.1     | 75.5     | 65.8     | 73.3     | 73.3     |
|     CPN[13]      | ResNet-Inception | 384×288     | 72.1     | 91.4     | 80.0     | 68.7     | 77.2     | 78.5     |
|     RPME[35]     | PyraNet          | 320×256     | 72.3     | 89.2     | 79.1     | 68.0     | 78.6     | -        |
|     **Ours**     | **ResNeSt-101**  | **384×288** | **72.8** | **90.9** | **80.5** | **69.1** | **79.3** | **79.2** |

**Note:**“*” means that the method involves extra data for training.

## Quick start
### Installation
 1. Install pytorch >= v0.4.1

 2. Clone this repo.and we'll call the directory that you cloned as ```ROOT```.

 3. Install dependencies:
    ```pip install -r requirements.txt```

 4. Download MSCOCO2017 images and annotations from [http://cocodataset.org/#download](http://cocodataset.org/#download). 
After placing data and annotation files. Please run ```label_transform.py``` at ```ROOT``` to transform the annotation fomat.

    Place COCO2017 data in this folder like this:
```
data
├── COCO2017
│   ├── annotations
│	│	├── person_keypoints_train2017.json
│	│	└── person_keypoints_val2017.json
│   ├── train2017
│   └── val2017
└── README.md
```

 5. Initialize cocoapi
```
git submodule init
git submodule update
cd cocoapi/PythonAPI
make
```

 6. Training & Testing
 - Training
 ```
    cd ROOT/model/
    python3 train.py
 ```
 - Testing
 ```
    cd ROOT/model/
    python3 test.py
 ```

