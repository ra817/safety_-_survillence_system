Project plan:

### System detail
Model name:                           12th Gen Intel(R) Core(TM) i7-1255UCPU: x86_64
Physical cores: 10
Total threads: 12

### Dataset details
MS COCO 2017 dataset
(COCO = Common Objects in Context)

This includes:
118,000 training images
5,000 validation images

80 object classes
Everyday objects like people, cars, animals, bottles, etc.

link:
https://cocodataset.org/#download

PERSON
person

Vehicles
bicycle, car, motorcycle ,airplane, bus, train, truck, boat, Outdoor

TRAFFIC LIGHT
fire hydrant, stop sign, parking meter, bench

Animals
bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

Accessories
backpack, umbrella, handbag, tie, suitcase

Sports
frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket

Kitchen
bottle, wine glass, cup, fork, knife, spoon, bowl

Food
banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

Furniture
chair, couch, potted plant, bed, dining table, toilet

Electronics
tv, laptop, mouse, remote, keyboard, cell phone

Appliances
microwave, oven, toaster, sink, refrigerator

Indoor
book, clock, vase, scissors, teddy bear, hair drier, toothbrush



### Model architecture
yolo12n:
Input image: preprocessing(scale image to 640*640)
(0): Conv(
      (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
input: 640*640*3
ouput: 320*320*16

x_norm: (x-mean) / sqrt(variance + eps)
        eps: prevent division by 0

momentun: it used control how fast running mean and variance will update.
one is running mean & variant and another one is batch mean & variance which come each batch.
then this is used to update to the running part. here momentum is used to handle how running will be updated using each batch details. 

track_running_stats=True
BN keeps track of: running mean, running variance




used to learn low level features
edges
corner
color gradients
simple textures

