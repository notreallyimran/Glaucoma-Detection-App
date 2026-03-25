% --- Feature Extraction Visualizer ---

% 1. Load your trained model
disp('Loading model...');
load('glaucoma_model.mat', 'trainedNet');

% 2. Open a file chooser so you can pick a specific image to analyze
% (Tip: Pick one from your dataset_cropped folder!)
[file, path] = uigetfile({'*.jpg;*.png;*.jpeg', 'Image Files'}, 'Select a cropped fundus image');
if isequal(file, 0)
    disp('Canceled');
    return; 
end

% Read and resize the image for ResNet
img = imread(fullfile(path, file));
imgResized = imresize(img, [224 224]);

disp('Extracting features from the neural network...');

% 3. Look at an EARLY layer 
earlyLayer = 'conv1';
earlyFeatures = activations(trainedNet, imgResized, earlyLayer);

% 4. Look at a DEEP layer 
deepLayer = 'activation_40_relu';
deepFeatures = activations(trainedNet, imgResized, deepLayer);

% 5. Plot them side-by-side
figure('Name', 'Feature Extraction Comparison', 'Position', [100, 100, 1200, 400]);

% Panel 1: Original Image
subplot(1,3,1); 
imshow(imgResized); 
title('1. Original Input');

% Panel 2: Early Layer
subplot(1,3,2); 
% We grab the first 16 feature maps and tile them into a grid
imshow(imtile(mat2gray(earlyFeatures(:,:,1:16)))); 
title('2. Early Features (conv1\_relu)');

% Panel 3: Deep Layer
subplot(1,3,3); 
imshow(imtile(mat2gray(deepFeatures(:,:,1:16)))); 
title('3. Deep Features (activation\_40\_relu)');

disp('✅ Done! Take a screenshot of the popup window for your report.');