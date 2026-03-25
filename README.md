# Glaucoma Detection App using ResNet-50 Neural Network and Grad-CAM XAI
# Project Overview
This Matlab-powered application classifies DFIs into glaucoma positive (GON+) and healthy patient Using Hillel Yaffe Glaucoma Dataset

Dataset obtained from https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/

# Project Files
crop dataset - preprocesses data
train_model.m - trains data
generate_features - Features extraction
generate_report_metrics - displays confusion matrix, ROC-AUC snd performance parameters
app1.mlapp - Runs our detection app
glaucoma_model.mat - contains saved workspace

# Before Running the App
1. This app only work on Matlab R2024a and later version
2. Install these Toolboxes on Matlab:
- deep learning toolbox
- deep learning toolbox for resnet-50 network
- image processing toolbox

# Using the App
1. Open app1, this may take a while
2. Upload any DFI from dataset
3. EWait for model's verdict

# FAQ
1. I can't run the app
- Install the toolboxes
2. I 've installed the toolboxes but it's still not running
- Your Matlab version is incompatible

# Authors
1. Imran Fareez Azmy - imranfareez1@gmail.com
2. Ahmad Nafis Zulkiifli - ahmadnafiszulkifli@gmail.com
