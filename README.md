This is a fork of the opencv_contrib (https://github.com/opencv/opencv) repo, modified to support cat face and facemarks.

## Building

1. Download OpenCV source code to local machine: https://github.com/opencv/opencv
```
cd ~
cd git
git clone https://github.com/opencv/opencv.git
```
2. Download this repo to local machine
```
cd ~
cd git
git clone https://github.com/willsong/opencv-facemark-trainer.git
```
3. Run cmake for the build:
```
cd ~
cd git
mkdir build
cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release ../opencv
```
4. Build (this can take ~30 mins depending on HW)
```
cd ~
cd git
cd build
make -j6
```

## Setup for training

1. Download kaggle dataset (https://www.kaggle.com/crawford/cat-dataset)
2. Download facemark annotator
```
cd ~
cd git
git clone https://github.com/willsong/opencv-facemark-annotator.git
```

## Create annoations

The kaggle dataset is divided into different folders.
Assume that we are annotating the images in: ~/Downloads/archive/CAT_06/

1. Create target folder - images for training will be copied here, as well as the annotation data
```
cd ~
cd git
cd opencv-facemark-annotator
cd assets
mkdir test
```
2. Run the annotator - all images will be copied, and annot.txt and images.txt will be created
```
cd ~
cd git
cd opencv-facemark-annotator
cd util
python annotate.py ~/Downloads/archive/CAT_06/ ~/git/opencv-facemark-annotator/assets/test/
```

## Run the trainer

Run the following commands
```
cd ~
cd git
cd build
bin/example_face_facemark_demo_lbf ~/git/opencv-facemark-annotator/assets/cascade/haarcascade_frontalcatface_extended.xml ~/git/opencv-facemark-annotator/assets/test/images.txt ~/git/opencv-facemark-annotator/assets/test/annot.txt ~/git/opencv-facemark-annotator/assets/test/lbfmodel.yaml
```
The model lbfmodel.yaml will be generated in ~/git/opencv-facemark-annotator/assets/test/
