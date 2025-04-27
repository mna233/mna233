%% Fault Detection in Three-Phase Multilevel Inverters Using Machine Learning
% This code implements various machine learning models to detect and classify
% faults in multilevel inverters for smart grid applications.

clc;
clear;
close all;

%% 1. Data Generation and Feature Extraction
% Simulate normal and fault conditions in a 3-level NPC inverter

% Parameters
fs = 20e3; % Sampling frequency (Hz)
T = 1/fs; % Sampling period
t = 0:T:0.2-T; % Time vector (0.1 second)
f = 50; % Fundamental frequency (Hz)
Vdc = 400; % DC link voltage (V)
m = 0.85; % Modulation index

% Generate reference signals
Vref = m * sin(2*pi*f*t); % Reference sine wave
triang = sawtooth(2*pi*20*t, 0.5); % Carrier wave (triangular)

% Initialize output voltage
Vout = zeros(size(t));

% Normal operation PWM generation
for i = 1:length(t)
if Vref(i) > triang(i)
Vout(i) = Vdc/2;
elseif Vref(i) < -triang(i)
Vout(i) = -Vdc/2;
else
Vout(i) = 0;
end
end

% Add noise to simulate real conditions
Vout = Vout + 0.02*Vdc*randn(size(Vout));

% Create fault conditions
% Fault 1: Lower MOSFETs isolated (Scenario I)
Vout_fault1 = Vout;
fault_start = floor(length(t)/3);
fault_end = floor(2*length(t)/3);
Vout_fault1(fault_start:fault_end) = Vout_fault1(fault_start:fault_end) + 0.5*Vdc;

% Fault 2: Lower MOSFETs short-circuited (Scenario II)
Vout_fault2 = Vout;
Vout_fault2(fault_start:fault_end) = Vout_fault2(fault_start:fault_end) - 0.7*Vdc;

% Fault 3: Upper MOSFETs isolated (Scenario III)
Vout_fault3 = Vout;
Vout_fault3(fault_start:fault_end) = Vout_fault3(fault_start:fault_end) + 0.8*Vdc;

% Fault 4: Upper MOSFETs short-circuited (Scenario IV)
Vout_fault4 = Vout;
Vout_fault4(fault_start:fault_end) = Vout_fault4(fault_start:fault_end) - 0.6*Vdc;

%% Feature Extraction using FFT and THD Calculation
% Function to calculate THD and extract harmonic features
function [thd, features] = extractFeatures(signal, fs, f)
N = length(signal);
fft_signal = abs(fft(signal)/N);
fft_signal = fft_signal(1:N/2+1);
fft_signal(2:end-1) = 2*fft_signal(2:end-1);
f_axis = fs*(0:(N/2))/N;
% Find fundamental frequency component
[~, fund_idx] = min(abs(f_axis - f));
fund_amp = fft_signal(fund_idx);
% Calculate THD
harmonic_bands = (2:40)*f; % Up to 40th harmonic
harmonic_power = 0;
for h = harmonic_bands
[~, h_idx] = min(abs(f_axis - h));
if h_idx <= length(fft_signal)
harmonic_power = harmonic_power + fft_signal(h_idx)^2;
end
end
thd = sqrt(harmonic_power)/fund_amp * 100;
% Extract harmonic features (1st to 5th harmonics)
features = zeros(1,5);
for h = 1:5
[~, h_idx] = min(abs(f_axis - h*f));
if h_idx <= length(fft_signal)
features(h) = fft_signal(h_idx)/fund_amp;
end
end
% Add statistical features
features = [features, mean(signal), std(signal), skewness(signal), kurtosis(signal)];
end

% Extract features for all conditions
[thd_normal, features_normal] = extractFeatures(Vout, fs, f);
[thd_fault1, features_fault1] = extractFeatures(Vout_fault1, fs, f);
[thd_fault2, features_fault2] = extractFeatures(Vout_fault2, fs, f);
[thd_fault3, features_fault3] = extractFeatures(Vout_fault3, fs, f);
[thd_fault4, features_fault4] = extractFeatures(Vout_fault4, fs, f);

% Display THD values
fprintf('THD Values:\n');
fprintf('Normal operation: %.2f%%\n', thd_normal);
fprintf('Lower MOSFETs isolated: %.2f%%\n', thd_fault1);
fprintf('Lower MOSFETs short-circuited: %.2f%%\n', thd_fault2);
fprintf('Upper MOSFETs isolated: %.2f%%\n', thd_fault3);
fprintf('Upper MOSFETs short-circuited: %.2f%%\n', thd_fault4);

%% 2. Create Dataset for Machine Learning
% Generate multiple samples for each condition with slight variations
num_samples = 100;
features = zeros(num_samples*5, 9); % 5 conditions, 9 features each
labels = zeros(num_samples*5, 1);

for i = 1:num_samples
% Normal operation with random variations
noise_level = 0.01 + 0.01*rand();
V_sample = Vout + noise_level*Vdc*randn(size(Vout));
[~, features((i-1)*5+1,:)] = extractFeatures(V_sample, fs, f);
labels((i-1)*5+1) = 0;
% Fault 1 with random variations
V_sample = Vout_fault1 + noise_level*Vdc*randn(size(Vout));
[~, features((i-1)*5+2,:)] = extractFeatures(V_sample, fs, f);
labels((i-1)*5+2) = 1;
% Fault 2 with random variations
V_sample = Vout_fault2 + noise_level*Vdc*randn(size(Vout));
[~, features((i-1)*5+3,:)] = extractFeatures(V_sample, fs, f);
labels((i-1)*5+3) = 2;
% Fault 3 with random variations
V_sample = Vout_fault3 + noise_level*Vdc*randn(size(Vout));
[~, features((i-1)*5+4,:)] = extractFeatures(V_sample, fs, f);
labels((i-1)*5+4) = 3;
% Fault 4 with random variations
V_sample = Vout_fault4 + noise_level*Vdc*randn(size(Vout));
[~, features((i-1)*5+5,:)] = extractFeatures(V_sample, fs, f);
labels((i-1)*5+5) = 4;
end

% Split into training and testing sets (70/30 split)
rng(42); % For reproducibility
cv = cvpartition(labels, 'HoldOut', 0.3);
X_train = features(cv.training,:);
y_train = labels(cv.training,:);
X_test = features(cv.test,:);
y_test = labels(cv.test,:);

%% 3. Train and Evaluate Multiple Machine Learning Models

% Model 1: Decision Tree
disp('Training Decision Tree...');
tree = fitctree(X_train, y_train, 'OptimizeHyperparameters', 'auto');
y_pred_tree = predict(tree, X_test);
acc_tree = sum(y_pred_tree == y_test)/length(y_test) * 100;
fprintf('Decision Tree Accuracy: %.2f%%\n', acc_tree);

% Model 2: Support Vector Machine (SVM)
disp('Training SVM...');
svm = fitcecoc(X_train, y_train, 'OptimizeHyperparameters', 'auto');
y_pred_svm = predict(svm, X_test);
acc_svm = sum(y_pred_svm == y_test)/length(y_test) * 100;
fprintf('SVM Accuracy: %.2f%%\n', acc_svm);

% Model 3: Neural Network
disp('Training Neural Network...');
net = patternnet(10); % Single hidden layer with 10 neurons
net.trainParam.showWindow = false; % Suppress training GUI
net = train(net, X_train', dummyvar(y_train+1)');
y_pred_nn = net(X_test');
[~, y_pred_nn] = max(y_pred_nn);
y_pred_nn = y_pred_nn' - 1;
acc_nn = sum(y_pred_nn == y_test)/length(y_test) * 100;
fprintf('Neural Network Accuracy: %.2f%%\n', acc_nn);

% Model 4: Random Forest
disp('Training Random Forest...');
rf = TreeBagger(50, X_train, y_train, 'Method', 'classification');
y_pred_rf = str2double(predict(rf, X_test));
acc_rf = sum(y_pred_rf == y_test)/length(y_test) * 100;
fprintf('Random Forest Accuracy: %.2f%%\n', acc_rf);

% Model 5: k-Nearest Neighbors (k-NN)
disp('Training k-NN...');
knn = fitcknn(X_train, y_train, 'OptimizeHyperparameters', 'auto');
y_pred_knn = predict(knn, X_test);
acc_knn = sum(y_pred_knn == y_test)/length(y_test) * 100;
fprintf('k-NN Accuracy: %.2f%%\n', acc_knn);

%% 4. Compare Model Performance and Select the Best One
model_names = {'Decision Tree', 'SVM', 'Neural Network', 'Random Forest', 'k-NN'};
accuracies = [acc_tree, acc_svm, acc_nn, acc_rf, acc_knn];

figure;
bar(accuracies);
title('Model Comparison - Classification Accuracy');
set(gca, 'XTickLabel', model_names);
ylabel('Accuracy (%)');
grid on;

% Find the best model
[best_acc, best_idx] = max(accuracies);
fprintf('\nBest model: %s with %.2f%% accuracy\n', model_names{best_idx}, best_acc);

%% 5. Confusion Matrix for the Best Model
switch best_idx
case 1
best_model = tree;
y_pred = y_pred_tree;
case 2
best_model = svm;
y_pred = y_pred_svm;
case 3
y_pred = y_pred_nn;
case 4
best_model = rf;
y_pred = y_pred_rf;
case 5
best_model = knn;
y_pred = y_pred_knn;
end

figure;
confusionchart(y_test, y_pred);
title(['Confusion Matrix for ', model_names{best_idx}]);
xlabel('Predicted Class');
ylabel('True Class');

%% 6. Feature Importance Analysis (for tree-based models)
if ismember(best_idx, [1,4]) % Decision Tree or Random Forest
figure;
if best_idx == 1
imp = predictorImportance(tree);
else
imp = rf.OOBPermutedPredictorDeltaError;
end
bar(imp);
title('Feature Importance');
xlabel('Features');
ylabel('Importance Score');
xticklabels({'1st Harm', '2nd Harm', '3rd Harm', '4th Harm', '5th Harm', ...
'Mean', 'Std Dev', 'Skewness', 'Kurtosis'});
xtickangle(45);
grid on;
end

%% 7. Visualize Fault Signatures
figure;
subplot(3,2,1);
plot(t, Vout);
title('Normal Operation');
xlabel('Time (s)');
ylabel('Voltage (V)');
grid on;

subplot(3,2,2);
plot(t, Vout_fault1);
title('Lower MOSFETs Isolated');
xlabel('Time (s)');
ylabel('Voltage (V)');
grid on;

subplot(3,2,3);
plot(t, Vout_fault2);
title('Lower MOSFETs Short-Circuited');
xlabel('Time (s)');
ylabel('Voltage (V)');
grid on;

subplot(3,2,4);
plot(t, Vout_fault3);
title('Upper MOSFETs Isolated');
xlabel('Time (s)');
ylabel('Voltage (V)');
grid on;

subplot(3,2,5);
plot(t, Vout_fault4);
title('Upper MOSFETs Short-Circuited');
xlabel('Time (s)');
ylabel('Voltage (V)');
grid on;

%% 8. FFT Analysis Visualization
[~, fft_normal] = extractFeatures(Vout, fs, f);
[~, fft_fault1] = extractFeatures(Vout_fault1, fs, f);
[~, fft_fault2] = extractFeatures(Vout_fault2, fs, f);
[~, fft_fault3] = extractFeatures(Vout_fault3, fs, f);
[~, fft_fault4] = extractFeatures(Vout_fault4, fs, f);

figure;
subplot(2,1,1);
bar([fft_normal(1:5); fft_fault1(1:5); fft_fault2(1:5); fft_fault3(1:5); fft_fault4(1:5)]);
title('Harmonic Components (1st-5th)');
legend('1st', '2nd', '3rd', '4th', '5th');
set(gca, 'XTickLabel', {'Normal', 'Fault1', 'Fault2', 'Fault3', 'Fault4'});
ylabel('Normalized Amplitude');
grid on;

subplot(2,1,2);
bar([thd_normal, thd_fault1, thd_fault2, thd_fault3, thd_fault4]);
title('Total Harmonic Distortion (THD)');
set(gca, 'XTickLabel', {'Normal', 'Fault1', 'Fault2', 'Fault3', 'Fault4'});
ylabel('THD (%)');
grid on;
%