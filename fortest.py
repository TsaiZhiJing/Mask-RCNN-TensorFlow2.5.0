# -*- coding: utf-8 -*-
import os
import sys
import skimage.io
import time
import cv2
from mrcnn.config import Config
# Root directory of the project
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
 
# Import Mask RCNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

time1 = time.time() 
 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
# Local path to trained weights file
MASK_RCNN_MODEL_PATH = os.path.join(MODEL_DIR ,"ResNet-50-FPN\\mask_rcnn_shapes_0030.h5")
# MASK_RCNN_MODEL_PATH = os.path.join(MODEL_DIR ,"ResNet-101-FPN\\mask_rcnn_shapes_0030.h5")
 
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "image")
 
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
 
    BACKBONE = "resnet50"  # resnet50 or resnet101
    # BACKBONE = "resnet101"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 3 shapes
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
 
#import train_tongue
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
config = InferenceConfig()
 
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
# Load weights trained on MS-COCO
model.load_weights(MASK_RCNN_MODEL_PATH, by_name=True)
 
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'scratch', 'dent']

# Load a random image from the images folder
img_name = "test.jpg"
IMAGE_PATH = os.path.join(IMAGE_DIR ,img_name)
image = skimage.io.imread(IMAGE_PATH)
 
# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]

masked_image=visualize.cv_process(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])

IMAGE_UPLOAD_PATH = os.path.join(ROOT_DIR ,"output.jpg")
cv2.imwrite(IMAGE_UPLOAD_PATH, masked_image[:,:,(2,1,0)])

time2 = time.time()
print(time2 - time1)