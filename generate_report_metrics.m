
                %%%%CONFUSION MATRIX%%%%

% Load model and validation data
load('glaucoma_model.mat', 'trainedNet');

% Get the AI's predictions on validation data
[predictedLabels, ~] = classify(trainedNet, augimdsValidation);
actualLabels = imdsValidation.Labels;

% Generate and show ONLY the Confusion Matrix
figure('Name', 'Clinical Confusion Matrix');
confusionchart(actualLabels, predictedLabels, ...
    'Title', 'Glaucoma Detection Confusion Matrix');
%% ROC AUC
% --- 1. Identify the Positive Class ---
% The 'scores' variable has two columns of probabilities (Healthy and GON+).
% We need to grab specifically the column for our positive pathology (GON+).
classes = trainedNet.Layers(end).Classes;
positiveClassIdx = find(classes == 'GON+'); 

% Extract just the confidence percentages for GON+
gonProbabilities = scores(:, positiveClassIdx);

% --- 2. Calculate the ROC Curve Math ---
% 'perfcurve' is MATLAB's built-in performance curve calculator
[X, Y, T, AUC] = perfcurve(actualLabels, gonProbabilities, 'GON+');

% --- 3. Plot the Graph ---
figure('Name', 'Clinical ROC Curve', 'Position', [100, 100, 600, 500]);
plot(X, Y, 'b-', 'LineWidth', 3); % The AI's performance line
hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1.5); % The "Random Guess" diagonal line

% Add professional labels and titles
xlabel('False Positive Rate (1 - Specificity)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('True Positive Rate (Sensitivity)', 'FontSize', 12, 'FontWeight', 'bold');
title('ROC Curve: Glaucoma Detection (ResNet-50)', 'FontSize', 14);
legend(['Model Performance (AUC = ', num2str(AUC, '%.4f'), ')'], 'Random Guess (AUC = 0.50)', 'Location', 'southeast');

grid on;
hold off;

disp(['✅ ROC Curve generated! Your AUC Score is: ', num2str(AUC)]);

%% Calculate precision, recall, accuracy & F1 score
% --- Clinical Performance Metrics Calculator ---
% This script calculates Accuracy, Precision, Recall, and F1-Score
% assuming 'actualLabels' and 'predictedLabels' are in your workspace.

% 1. Define the Positive Pathological Class
% For this project, detecting Glaucoma (GON+) is our primary target.
positiveClass = categorical({'GON+'});

% 2. Extract Confusion Matrix Variables using Logical Arrays
% True Positives: Actual is GON+, Predicted is GON+
TP = sum((actualLabels == positiveClass) & (predictedLabels == positiveClass));

% True Negatives: Actual is Healthy, Predicted is Healthy
TN = sum((actualLabels ~= positiveClass) & (predictedLabels ~= positiveClass));

% False Positives: Actual is Healthy, Predicted is GON+ (False Alarm)
FP = sum((actualLabels ~= positiveClass) & (predictedLabels == positiveClass));

% False Negatives: Actual is GON+, Predicted is Healthy (Missed Diagnosis)
FN = sum((actualLabels == positiveClass) & (predictedLabels ~= positiveClass));

% 3. Calculate the Core Metrics
accuracy = (TP + TN) / (TP + TN + FP + FN);
precision = TP / (TP + FP);
recall = TP / (TP + FN); % Clinically known as Sensitivity
f1_score = 2 * (precision * recall) / (precision + recall);

% 4. Print the Results for your Report

fprintf('\n=== Glaucoma XAI Model Performance ===\n');
fprintf('Accuracy  : %.4f \n', accuracy);
fprintf('Precision : %.4f \n', precision);
fprintf('Recall    : %.4f \n', recall);
fprintf('F1-Score  : %.4f \n', f1_score);
fprintf('======================================\n\n');
