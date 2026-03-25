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

# IMPORTANT (Before running the app)
Install these Toolboxes on Matlab:
- deep learning toolbox
- deep learning toolbox for resnet-50 network
- image processing toolbox

# How to open
1. Open app1
2. Upload any DFI from dataset
3. Enjoy

# If it's not working
1. unzip glaucoma_model.rar and open glaucoma_model.mat. Wait for import data prompt to show up
2. Tick trainNed and select import
3. write appdesigner on command window, then click to open app1
4. Run app1
