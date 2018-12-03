# CornerNet
tensorflow
# CornerNet: Training and Evaluation Code
Code for reproducing the results in the following paper:

[**CornerNet: Detecting Objects as Paired Keypoints**](https://arxiv.org/abs/1808.01244)  
Hei Law, Jia Deng  
*European Conference on Computer Vision (ECCV), 2018*

## Getting Started
environment
```
tensorflow==1.10
python3.6
```
Our current implementation only supports GPU so you need a GPU and need to have CUDA installed on your machine.

### Installing MS COCO APIs
You also need to install the MS COCO APIs.
```
cd <CornetNet dir>/data
git clone https://github.com/cocodataset/cocoapi.git 
cd <CornetNet dir>/data/coco/PythonAPI
make
```

### Downloading MS COCO Data
- Download the training/validation split we use in our paper from [here](https://drive.google.com/file/d/1dop4188xo5lXDkGtOZUzy2SHOD_COXz4/view?usp=sharing) (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/tree/master/data))
- Unzip the file and place `annotations` under `<CornetNet dir>/data/coco`
- Download the images (2014 Train, 2014 Val, 2017 Test) from [here](http://cocodataset.org/#download)
- Create 3 directories, `trainval2014`, `minival2014` and `testdev2017`, under `<CornerNet dir>/data/coco/images/`
- Copy the training/validation/testing images to the corresponding directories according to the annotation files

## Training and Evaluation
We provide the configuration file (`CornerNet.json`) and the model file (`CornerNet.py`) for CornerNet in this repo. 

To train CornerNet:
```
python train.py
```
To use the trained model:
```
python test.py 
```
##In the next few days I will provide model parameters that are trained on coco.
