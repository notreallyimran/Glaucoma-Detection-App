# Glaucoma Detection App using ResNet-50 Neural Network and Grad-CAM XAI
# Project Overview
This Matlab-powered application classifies DFIs into glaucoma positive (GON+). Our ResNet-50 Neural Network is trained with DFIs from Hillel Yaffe Glaucoma Dataset - https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/


# Project Files
1. crop dataset - preprocesses data
2. train_model.m - trains data
3. generate_features - Features extraction
4. generate_report_metrics - displays confusion matrix, ROC-AUC snd performance parameters
5. app1.mlapp - Runs our detection app
6. glaucoma_model.mat - contains saved workspace, trained model

# Before Running the App
1. This app only work on Matlab R2024a and later version
2. Install these Toolboxes on Matlab:
   - deep learning toolbox
   - deep learning toolbox for resnet-50 network
   - image processing toolbox

# Using the App
1. Open app1.mlapp, this may take a while
2. Upload any DFI from dataset
3. Wait for model's verdict

# FAQ
1. I can't run the app
    - Install the toolboxes
2. I 've installed the toolboxes and it's still not running
    - Open matlab>home>import data>open glaucoma_model>tick trainedNet>click import
    - type appdesigner on command window and load app1
    - Run app1

# Authors
1. Imran Fareez Azmy - imranfareez1@gmail.com
2. Ahmad Nafis Zulkiifli - ahmadnafiszulkifli@gmail.com
