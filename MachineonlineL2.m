%% Fault Detection in Three-Phase Multilevel Inverters Using Machine Learning
% Enhanced version with realistic dataset, advanced feature extraction, and
% improved machine learning pipeline

clc;
clear;
close all;

%% 1. Enhanced Data Generation with Three-Phase System
% Simulate a complete three-phase NPC inverter with more realistic faults

% Parameters
fs = 20e3;          % Sampling frequency (Hz)
T = 1/fs;           % Sampling period
t = 0:T:0.2-T;      % Time vector (0.2 second)
f = 50;             % Fundamental frequency (Hz)
Vdc = 400;          % DC link voltage (V)
m = 0.85;           % Modulation index
phase_shift = 2*pi/3; % 120 degrees phase shift for three-phase

% Generate three-phase reference signals
Vref_a = m * sin(2*pi*f*t);
Vref_b = m * sin(2*pi*f*t - phase_shift);
Vref_c = m * sin(2*pi*f*t + phase_shift);

% Carrier wave (triangular)
triang = sawtooth(2*pi*20*t, 0.5);

% Initialize output voltages for all three phases
Vout_a = zeros(size(t));
Vout_b = zeros(size(t));
Vout_c = zeros(size(t));

% PWM generation for all three phases
for i = 1:length(t)
    % Phase A
    if Vref_a(i) > triang(i)
        Vout_a(i) = Vdc/2;
    elseif Vref_a(i) < -triang(i)
        Vout_a(i) = -Vdc/2;
    else
        Vout_a(i) = 0;
    end
    
    % Phase B
    if Vref_b(i) > triang(i)
        Vout_b(i) = Vdc/2;
    elseif Vref_b(i) < -triang(i)
        Vout_b(i) = -Vdc/2;
    else
        Vout_b(i) = 0;
    end
    
    % Phase C
    if Vref_c(i) > triang(i)
        Vout_c(i) = Vdc/2;
    elseif Vref_c(i) < -triang(i)
        Vout_c(i) = -Vdc/2;
    else
        Vout_c(i) = 0;
    end
end

% Add realistic noise and disturbances
noise_level = 0.02;
Vout_a = Vout_a + noise_level*Vdc*randn(size(Vout_a));
Vout_b = Vout_b + noise_level*Vdc*randn(size(Vout_b));
Vout_c = Vout_c + noise_level*Vdc*randn(size(Vout_c));

% Create more realistic fault conditions with varying severity and duration
fault_start = floor(length(t)/3);
fault_end = floor(2*length(t)/3);

% Define fault types (now including partial faults and multiple device faults)
fault_types = {
    'Normal',               % 0
    'Single IGBT Open',     % 1 (Phase A upper)
    'Single IGBT Short',    % 2 (Phase A lower)
    'Phase Leg Open',       % 3 (Phase A)
    'Phase Leg Short',      % 4 (Phase A)
    'Cross Phase Fault',    % 5 (Phase A-B)
    'DC Link Imbalance',    % 6
    'Multiple IGBT Fault'   % 7 (Phase A upper + Phase B lower)
};

num_fault_types = length(fault_types) - 1; % Exclude normal

% Generate fault signals for all phases
[Vout_a_faults, Vout_b_faults, Vout_c_faults] = deal(zeros(num_fault_types, length(t)));

for fault = 1:num_fault_types
    Vout_a_faults(fault,:) = Vout_a;
    Vout_b_faults(fault,:) = Vout_b;
    Vout_c_faults(fault,:) = Vout_c;
    
    switch fault
        case 1 % Single IGBT Open (Phase A upper)
            for i = fault_start:fault_end
                if Vref_a(i) > triang(i)
                    Vout_a_faults(fault,i) = 0; % Upper IGBT fails to turn on
                end
            end
            
        case 2 % Single IGBT Short (Phase A lower)
            for i = fault_start:fault_end
                if Vref_a(i) < -triang(i)
                    Vout_a_faults(fault,i) = -Vdc/2 + 0.3*Vdc*rand(); % Partial short
                end
            end
            
        case 3 % Phase Leg Open (Phase A)
            Vout_a_faults(fault,fault_start:fault_end) = 0;
            
        case 4 % Phase Leg Short (Phase A)
            Vout_a_faults(fault,fault_start:fault_end) = Vout_a_faults(fault,fault_start:fault_end) - 0.6*Vdc;
            
        case 5 % Cross Phase Fault (Phase A-B)
            Vout_a_faults(fault,fault_start:fault_end) = Vout_b(fault_start:fault_end) + 0.2*Vdc*randn(1,fault_end-fault_start+1);
            Vout_b_faults(fault,fault_start:fault_end) = Vout_a(fault_start:fault_end) + 0.2*Vdc*randn(1,fault_end-fault_start+1);
            
        case 6 % DC Link Imbalance
            imbalance_factor = 0.7 + 0.3*rand(); % Random imbalance between 0.7-1.0
            for i = fault_start:fault_end
                if Vout_a(i) > 0
                    Vout_a_faults(fault,i) = Vout_a(i) * imbalance_factor;
                end
                if Vout_b(i) > 0
                    Vout_b_faults(fault,i) = Vout_b(i) * imbalance_factor;
                end
                if Vout_c(i) > 0
                    Vout_c_faults(fault,i) = Vout_c(i) * imbalance_factor;
                end
            end
            
        case 7 % Multiple IGBT Fault
            for i = fault_start:fault_end
                % Phase A upper IGBT fault
                if Vref_a(i) > triang(i)
                    Vout_a_faults(fault,i) = Vout_a(i) * (0.5 + 0.3*rand());
                end
                % Phase B lower IGBT fault
                if Vref_b(i) < -triang(i)
                    Vout_b_faults(fault,i) = Vout_b(i) * (1.5 + 0.5*rand());
                end
            end
    end
    
    % Add fault-specific noise
    Vout_a_faults(fault,:) = Vout_a_faults(fault,:) + noise_level*Vdc*randn(size(Vout_a));
    Vout_b_faults(fault,:) = Vout_b_faults(fault,:) + noise_level*Vdc*randn(size(Vout_b));
    Vout_c_faults(fault,:) = Vout_c_faults(fault,:) + noise_level*Vdc*randn(size(Vout_c));
end

%% Enhanced Feature Extraction
% Now includes time-domain, frequency-domain, and time-frequency features

function [features] = enhancedFeatureExtraction(signal, fs, f)
    N = length(signal);
    
    % 1. Time-domain features
    t_mean = mean(signal);
    t_std = std(signal);
    t_skew = skewness(signal);
    t_kurt = kurtosis(signal);
    t_rms = rms(signal);
    t_crest = max(abs(signal))/t_rms;
    t_clearance = max(abs(signal))/(mean(sqrt(abs(signal))))^2;
    t_shape = t_rms/(mean(abs(signal)));
    t_impulse = max(abs(signal))/(mean(abs(signal)));
    
    % 2. Frequency-domain features (FFT)
    fft_signal = abs(fft(signal)/N);
    fft_signal = fft_signal(1:N/2+1);
    fft_signal(2:end-1) = 2*fft_signal(2:end-1);
    f_axis = fs*(0:(N/2))/N;
    
    % Find fundamental frequency component
    [~, fund_idx] = min(abs(f_axis - f));
    fund_amp = fft_signal(fund_idx);
    
    % Harmonic features (1st to 10th harmonics)
    harmonic_features = zeros(1,10);
    for h = 1:10
        [~, h_idx] = min(abs(f_axis - h*f));
        if h_idx <= length(fft_signal)
            harmonic_features(h) = fft_signal(h_idx)/fund_amp;
        end
    end
    
    % THD calculation
    harmonic_bands = (2:40)*f; % Up to 40th harmonic
    harmonic_power = 0;
    for h = harmonic_bands
        [~, h_idx] = min(abs(f_axis - h));
        if h_idx <= length(fft_signal)
            harmonic_power = harmonic_power + fft_signal(h_idx)^2;
        end
    end
    thd = sqrt(harmonic_power)/fund_amp * 100;
    
    % 3. Time-frequency features (STFT)
    window = hamming(512);
    noverlap = 256;
    nfft = 1024;
    [S,~,~] = spectrogram(signal, window, noverlap, nfft, fs, 'yaxis');
    S = abs(S);
    
    % STFT statistical features
    stft_mean = mean(S(:));
    stft_std = std(S(:));
    stft_max = max(S(:));
    stft_min = min(S(:));
    stft_entropy = -sum(S(:).*log(S(:)+eps))/length(S(:));
    
    % 4. Wavelet features
    [cA, cD] = dwt(signal, 'db4');
    wavelet_energy = sum(abs(cD).^2);
    wavelet_entropy = -sum((abs(cD).^2).*log(abs(cD).^2+eps));
    
    % Combine all features
    features = [t_mean, t_std, t_skew, t_kurt, t_rms, t_crest, t_clearance, ...
                t_shape, t_impulse, harmonic_features, thd, stft_mean, ...
                stft_std, stft_max, stft_min, stft_entropy, wavelet_energy, wavelet_entropy];
end

% Extract features for all three phases and all conditions
num_features = length(enhancedFeatureExtraction(Vout_a, fs, f));
features = zeros(num_fault_types + 1, num_features * 3); % +1 for normal
labels = zeros(num_fault_types + 1, 1);

% Normal operation features (all three phases)
normal_features_a = enhancedFeatureExtraction(Vout_a, fs, f);
normal_features_b = enhancedFeatureExtraction(Vout_b, fs, f);
normal_features_c = enhancedFeatureExtraction(Vout_c, fs, f);
features(1,:) = [normal_features_a, normal_features_b, normal_features_c];
labels(1) = 0;

% Fault conditions features
for fault = 1:num_fault_types
    fault_features_a = enhancedFeatureExtraction(Vout_a_faults(fault,:), fs, f);
    fault_features_b = enhancedFeatureExtraction(Vout_b_faults(fault,:), fs, f);
    fault_features_c = enhancedFeatureExtraction(Vout_c_faults(fault,:), fs, f);
    features(fault+1,:) = [fault_features_a, fault_features_b, fault_features_c];
    labels(fault+1) = fault;
end

%% 2. Create Comprehensive Dataset with Multiple Operating Conditions
% Generate multiple samples with varying operating conditions

num_samples_per_condition = 500;
num_conditions = num_fault_types + 1; % Including normal
total_samples = num_samples_per_condition * num_conditions;

% Initialize arrays for dataset
X = zeros(total_samples, num_features * 3);
y = zeros(total_samples, 1);

% Parameters for variation
m_range = [0.7, 0.9]; % Modulation index range
vdc_range = [380, 420]; % DC link voltage range
load_range = [0.5, 1.5]; % Load variation (affects noise)

rng(42); % For reproducibility

for sample = 1:num_samples_per_condition
    % Vary operating parameters
    m_current = m_range(1) + diff(m_range)*rand();
    vdc_current = vdc_range(1) + diff(vdc_range)*rand();
    load_current = load_range(1) + diff(load_range)*rand();
    
    % Generate normal operation with current parameters
    [Vout_a_var, Vout_b_var, Vout_c_var] = generateThreePhaseOutput(...
        fs, t, f, vdc_current, m_current, load_current);
    
    idx = (sample-1)*num_conditions + 1;
    X(idx,:) = [enhancedFeatureExtraction(Vout_a_var, fs, f), ...
                enhancedFeatureExtraction(Vout_b_var, fs, f), ...
                enhancedFeatureExtraction(Vout_c_var, fs, f)];
    y(idx) = 0;
    
    % Generate each fault condition with current parameters
    for fault = 1:num_fault_types
        [Vout_a_fault, Vout_b_fault, Vout_c_fault] = generateFaultCondition(...
            fault, Vout_a_var, Vout_b_var, Vout_c_var, fs, t, f, ...
            vdc_current, m_current, load_current);
        
        idx = (sample-1)*num_conditions + 1 + fault;
        X(idx,:) = [enhancedFeatureExtraction(Vout_a_fault, fs, f), ...
                    enhancedFeatureExtraction(Vout_b_fault, fs, f), ...
                    enhancedFeatureExtraction(Vout_c_fault, fs, f)];
        y(idx) = fault;
    end
end

% Helper function to generate three-phase output with variations
function [Vout_a, Vout_b, Vout_c] = generateThreePhaseOutput(fs, t, f, Vdc, m, load_factor)
    phase_shift = 2*pi/3;
    Vref_a = m * sin(2*pi*f*t);
    Vref_b = m * sin(2*pi*f*t - phase_shift);
    Vref_c = m * sin(2*pi*f*t + phase_shift);
    triang = sawtooth(2*pi*20*t, 0.5);
    
    Vout_a = zeros(size(t));
    Vout_b = zeros(size(t));
    Vout_c = zeros(size(t));
    
    for i = 1:length(t)
        % Phase A
        if Vref_a(i) > triang(i)
            Vout_a(i) = Vdc/2;
        elseif Vref_a(i) < -triang(i)
            Vout_a(i) = -Vdc/2;
        end
        
        % Phase B
        if Vref_b(i) > triang(i)
            Vout_b(i) = Vdc/2;
        elseif Vref_b(i) < -triang(i)
            Vout_b(i) = -Vdc/2;
        end
        
        % Phase C
        if Vref_c(i) > triang(i)
            Vout_c(i) = Vdc/2;
        elseif Vref_c(i) < -triang(i)
            Vout_c(i) = -Vdc/2;
        end
    end
    
    % Add load-dependent noise
    noise_level = 0.02 * load_factor;
    Vout_a = Vout_a + noise_level*Vdc*randn(size(Vout_a));
    Vout_b = Vout_b + noise_level*Vdc*randn(size(Vout_b));
    Vout_c = Vout_c + noise_level*Vdc*randn(size(Vout_c));
end

% Helper function to generate fault conditions with variations
function [Vout_a_fault, Vout_b_fault, Vout_c_fault] = generateFaultCondition(...
    fault_type, Vout_a, Vout_b, Vout_c, fs, t, f, Vdc, m, load_factor)
    
    fault_start = floor(length(t)/3);
    fault_end = floor(2*length(t)/3);
    
    Vout_a_fault = Vout_a;
    Vout_b_fault = Vout_b;
    Vout_c_fault = Vout_c;
    
    % Add random fault duration variation
    duration_variation = 0.1 + 0.2*rand(); % 10-30% variation
    fault_duration = floor((fault_end - fault_start) * duration_variation);
    fault_start = fault_start + randi(floor(length(t)/6));
    fault_end = min(fault_start + fault_duration, length(t));
    
    % Add random fault severity
    severity = 0.5 + 0.5*rand(); % 50-100% severity
    
    switch fault_type
        case 1 % Single IGBT Open (Phase A upper)
            for i = fault_start:fault_end
                if Vout_a(i) > 0
                    Vout_a_fault(i) = Vout_a(i) * (1 - severity);
                end
            end
            
        case 2 % Single IGBT Short (Phase A lower)
            for i = fault_start:fault_end
                if Vout_a(i) < 0
                    Vout_a_fault(i) = Vout_a(i) * (1 + severity*rand());
                end
            end
            
        case 3 % Phase Leg Open (Phase A)
            Vout_a_fault(fault_start:fault_end) = 0;
            
        case 4 % Phase Leg Short (Phase A)
            Vout_a_fault(fault_start:fault_end) = Vout_a_fault(fault_start:fault_end) - 0.6*Vdc*severity;
            
        case 5 % Cross Phase Fault (Phase A-B)
            cross_coupling = 0.3 * severity;
            Vout_a_fault(fault_start:fault_end) = Vout_a_fault(fault_start:fault_end) + ...
                cross_coupling*Vout_b(fault_start:fault_end);
            Vout_b_fault(fault_start:fault_end) = Vout_b_fault(fault_start:fault_end) + ...
                cross_coupling*Vout_a(fault_start:fault_end);
            
        case 6 % DC Link Imbalance
            imbalance_factor = 1 - (0.3 * severity * rand());
            for i = fault_start:fault_end
                if Vout_a_fault(i) > 0
                    Vout_a_fault(i) = Vout_a_fault(i) * imbalance_factor;
                end
                if Vout_b_fault(i) > 0
                    Vout_b_fault(i) = Vout_b_fault(i) * imbalance_factor;
                end
                if Vout_c_fault(i) > 0
                    Vout_c_fault(i) = Vout_c_fault(i) * imbalance_factor;
                end
            end
            
        case 7 % Multiple IGBT Fault
            for i = fault_start:fault_end
                % Phase A upper IGBT fault
                if Vout_a_fault(i) > 0
                    Vout_a_fault(i) = Vout_a_fault(i) * (0.5 + 0.5*severity*rand());
                end
                % Phase B lower IGBT fault
                if Vout_b_fault(i) < 0
                    Vout_b_fault(i) = Vout_b_fault(i) * (1.5 + 0.5*severity*rand());
                end
            end
    end
    
    % Add load-dependent noise
    noise_level = 0.02 * load_factor;
    Vout_a_fault = Vout_a_fault + noise_level*Vdc*randn(size(Vout_a));
    Vout_b_fault = Vout_b_fault + noise_level*Vdc*randn(size(Vout_b));
    Vout_c_fault = Vout_c_fault + noise_level*Vdc*randn(size(Vout_c));
end

%% 3. Data Preprocessing and Feature Selection
% Normalize features
X_normalized = normalize(X);

% Perform feature selection using MRMR (Minimum Redundancy Maximum Relevance)
[idx, scores] = fscmrmr(X_normalized, y);

% Select top 30 most important features
num_selected_features = 30;
selected_features = idx(1:num_selected_features);
X_selected = X_normalized(:, selected_features);

% Split into training (70%), validation (15%), and testing (15%) sets
rng(42); % For reproducibility
cv = cvpartition(y, 'HoldOut', 0.3);
idx_train = cv.training;
idx_temp = cv.test;

X_train = X_selected(idx_train,:);
y_train = y(idx_train);

X_temp = X_selected(idx_temp,:);
y_temp = y(idx_temp);

cv_val = cvpartition(y_temp, 'HoldOut', 0.5);
X_val = X_temp(cv_val.training,:);
y_val = y_temp(cv_val.training);
X_test = X_temp(cv_val.test,:);
y_test = y_temp(cv_val.test);

% Visualize feature importance
figure;
barh(scores(selected_features));
set(gca, 'YTick', 1:num_selected_features, 'YTickLabel', selected_features);
title('Feature Importance Scores (MRMR)');
xlabel('Importance Score');
ylabel('Feature Index');
grid on;

%% 4. Train and Optimize Multiple Machine Learning Models
% Set up hyperparameter optimization options

 % Start parallel pool if none exists
% Set optimization options (serial mode)
optimize_options = struct(...
    'Optimizer', 'bayesopt', ...
    'MaxObjectiveEvaluations', 30, ...
    'UseParallel', false, ... % Force serial execution
    'ShowPlots', false);
% Model 1: Optimized Random Forest
disp('Training and optimizing Random Forest...');
rf_template = templateTree('Reproducible', true);
rf = fitcensemble(X_train, y_train, ...
    'Method', 'Bag', ...
    'Learners', rf_template, ...
    'OptimizeHyperparameters', {'NumLearningCycles', 'MinLeafSize'}, ...
    'HyperparameterOptimizationOptions', optimize_options);
% Predict on validation set
y_pred_rf = predict(rf, X_val);
acc_rf = sum(y_pred_rf == y_val)/length(y_val) * 100;
fprintf('Random Forest Validation Accuracy: %.2f%%\n', acc_rf);

% Model 2: Optimized SVM with Gaussian Kernel
disp('Training and optimizing SVM...');
svm = fitcecoc(X_train, y_train, ...
    'OptimizeHyperparameters', {'KernelScale', 'BoxConstraint'}, ...
    'HyperparameterOptimizationOptions', optimize_options);

y_pred_svm = predict(svm, X_val);
acc_svm = sum(y_pred_svm == y_val)/length(y_val) * 100;
fprintf('SVM Validation Accuracy: %.2f%%\n', acc_svm);

% Model 3: Optimized Neural Network
disp('Training and optimizing Neural Network...');
nn = fitcnet(X_train, y_train, ...
    'OptimizeHyperparameters', {'LayerSizes', 'Lambda', 'Activations'}, ...
    'HyperparameterOptimizationOptions', optimize_options);

y_pred_nn = predict(nn, X_val);
acc_nn = sum(y_pred_nn == y_val)/length(y_val) * 100;
fprintf('Neural Network Validation Accuracy: %.2f%%\n', acc_nn);

% Model 4: Optimized Gradient Boosting (multi-class version)
disp('Training and optimizing Gradient Boosting...');
gb = fitcensemble(X_train, y_train, ...
    'Method', 'AdaBoostM2', ...  % Multi-class boosting method
    'Learners', templateTree('Reproducible', true), ...
    'OptimizeHyperparameters', {'NumLearningCycles', 'LearnRate', 'MinLeafSize'}, ...
    'HyperparameterOptimizationOptions', optimize_options);

y_pred_gb = predict(gb, X_val);
acc_gb = sum(y_pred_gb == y_val)/length(y_val) * 100;
fprintf('Gradient Boosting Validation Accuracy: %.2f%%\n', acc_gb);

% Model 5: Optimized k-NN
disp('Training and optimizing k-NN...');
knn = fitcknn(X_train, y_train, ...
    'OptimizeHyperparameters', {'NumNeighbors', 'Distance'}, ...
    'HyperparameterOptimizationOptions', optimize_options);

y_pred_knn = predict(knn, X_val);
acc_knn = sum(y_pred_knn == y_val)/length(y_val) * 100;
fprintf('k-NN Validation Accuracy: %.2f%%\n', acc_knn);

%% 5. Model Evaluation and Selection
% Evaluate all models on test set
model_names = {'Random Forest', 'SVM', 'Neural Network', 'Gradient Boosting', 'k-NN'};
models = {rf, svm, nn, gb, knn};

% Initialize results
test_accuracies = zeros(1, length(models));
precision = zeros(length(fault_types), length(models));
recall = zeros(length(fault_types), length(models));
f1_scores = zeros(length(fault_types), length(models));

for i = 1:length(models)
    % Predict on test set
    y_pred = predict(models{i}, X_test);
    
    % Calculate accuracy
    test_accuracies(i) = sum(y_pred == y_test)/length(y_test) * 100;
    
    % Calculate precision, recall, and F1 for each class
    for class = 0:num_fault_types
        tp = sum((y_pred == class) & (y_test == class));
        fp = sum((y_pred == class) & (y_test ~= class));
        fn = sum((y_pred ~= class) & (y_test == class));
        
        precision(class+1,i) = tp / (tp + fp + eps);
        recall(class+1,i) = tp / (tp + fn + eps);
        f1_scores(class+1,i) = 2 * (precision(class+1,i) * recall(class+1,i)) / ...
                              (precision(class+1,i) + recall(class+1,i) + eps);
    end
end

% Display test accuracies
fprintf('\nTest Set Accuracies:\n');
for i = 1:length(models)
    fprintf('%s: %.2f%%\n', model_names{i}, test_accuracies(i));
end

% Find the best model
[best_acc, best_idx] = max(test_accuracies);
best_model = models{best_idx};
fprintf('\nBest model: %s with %.2f%% accuracy\n', model_names{best_idx}, best_acc);

% Plot model comparison
figure;
bar(test_accuracies);
title('Model Comparison - Test Set Accuracy');
set(gca, 'XTickLabel', model_names, 'XTickLabelRotation', 45);
ylabel('Accuracy (%)');
grid on;

% Plot precision-recall for best model
figure;
subplot(2,1,1);
bar(precision(:,best_idx));
title(['Precision per Class - ', model_names{best_idx}]);
set(gca, 'XTickLabel', fault_types, 'XTickLabelRotation', 45);
ylabel('Precision');
grid on;

subplot(2,1,2);
bar(recall(:,best_idx));
title(['Recall per Class - ', model_names{best_idx}]);
set(gca, 'XTickLabel', fault_types, 'XTickLabelRotation', 45);
ylabel('Recall');
grid on;

%% 6. Detailed Analysis of Best Model
% Confusion matrix
figure;
cm = confusionchart(y_test, predict(best_model, X_test));
title(['Confusion Matrix for ', model_names{best_idx}]);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% Feature importance for tree-based models
if ismember(best_idx, [1,4]) % Random Forest or Gradient Boosting
    figure;
    
    if best_idx == 1 % Random Forest
        imp = predictorImportance(best_model);
    else % Gradient Boosting
        imp = predictorImportance(best_model);
    end
    
    [~, sort_idx] = sort(imp, 'descend');
    barh(imp(sort_idx));
    set(gca, 'YTick', 1:num_selected_features, 'YTickLabel', selected_features(sort_idx));
    title(['Feature Importance - ', model_names{best_idx}]);
    xlabel('Importance Score');
    ylabel('Feature Index');
    grid on;
end
% Replace your ROC curve section with this corrected version:

if ismember(best_idx, [1,3,4]) % Models that support predict with scores
    figure;
    
    % Get predicted scores (ensure proper format)
    [~, scores] = predict(best_model, X_test);
    
    % Verify scores and labels dimensions match
    if size(scores,1) ~= length(y_test)
        error('Scores and labels dimensions mismatch. Scores has %d rows, labels has %d.', ...
              size(scores,1), length(y_test));
    end
    
    % Calculate and plot ROC for each class
    colors = lines(num_fault_types + 1); % Distinct colors
    hold on;
    
    auc = zeros(1, num_fault_types + 1);
    legend_entries = cell(1, num_fault_types + 1);
    
    for class = 0:num_fault_types
        try
            % Convert to binary classification problem for each class
            binary_labels = (y_test == class);
            
            % Calculate ROC curve
            [fpr, tpr, ~, auc_val] = perfcurve(binary_labels, scores(:,class+1), 1);
            
            % Plot curve
            plot(fpr, tpr, 'LineWidth', 2, 'Color', colors(class+1,:));
            
            % Store AUC and legend entry
            auc(class+1) = auc_val;
            legend_entries{class+1} = sprintf('%s (AUC=%.2f)', fault_types{class+1}, auc_val);
        catch ME
            warning('Error calculating ROC for class %d: %s', class, ME.message);
            auc(class+1) = NaN;
            legend_entries{class+1} = fault_types{class+1};
        end
    end
    
    % Add diagonal reference line
    plot([0 1], [0 1], 'k--');
    hold off;
    
    % Add legend and labels
    legend(legend_entries, 'Location', 'southeast');
    title(['ROC Curves - ', model_names{best_idx}]);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    grid on;
end
% Check class distribution
disp('Test set class distribution:');
tabulate(y_test)

% Check scores dimensions
disp('Score matrix dimensions:');
disp(size(scores))

% Check for NaN/inf in scores
if any(isnan(scores(:))) || any(isinf(scores(:)))
    warning('Scores contain NaN or Inf values - replacing with 0');
    scores(isnan(scores)) = 0;
    scores(isinf(scores)) = 0;
end

%% 7. Visualize Fault Signatures with Current Parameters
% Plot normal and fault waveforms for visualization
figure;
set(gcf, 'Position', [100, 100, 1200, 800]);

% Normal operation
subplot(3,3,1);
plot(t, Vout_a);
title('Normal Operation - Phase A');
xlabel('Time (s)');
ylabel('Voltage (V)');
grid on;

% Replace your visualization section (around line 716) with this:

% Visualize fault signatures with current parameters
figure;
set(gcf, 'Position', [100, 100, 1200, 800]);

% Define which faults to display (maximum 3 for 3x3 grid)
fault_examples = [1, 3, 5]; % Show three different fault types
num_examples = min(3, length(fault_examples)); % Ensure we don't exceed display capacity

% Normal operation
subplot(3, num_examples+1, 1); % +1 for normal operation
plot(t, Vout_a);
title('Normal Operation - Phase A');
xlabel('Time (s)');
ylabel('Voltage (V)');
grid on;

% Plot fault conditions
for i = 1:num_examples
    fault = fault_examples(i);
    
    % Phase A faults
    subplot(3, num_examples+1, i+1);
    plot(t, Vout_a_faults(fault,:));
    title(sprintf('%s - Phase A', fault_types{fault+1}));
    xlabel('Time (s)');
    ylabel('Voltage (V)');
    grid on;
    
    % Phase B faults
    subplot(3, num_examples+1, i+1 + (num_examples+1));
    plot(t, Vout_b_faults(fault,:));
    title(sprintf('%s - Phase B', fault_types{fault+1}));
    xlabel('Time (s)');
    ylabel('Voltage (V)');
    grid on;
    
    % Phase C faults
    subplot(3, num_examples+1, i+1 + 2*(num_examples+1));
    plot(t, Vout_c_faults(fault,:));
    title(sprintf('%s - Phase C', fault_types{fault+1}));
    xlabel('Time (s)');
    ylabel('Voltage (V)');
    grid on;
end  
%% 8. Advanced Feature Visualization
% Plot t-SNE visualization of the selected features
rng(42); % For reproducibility
Y = tsne(X_selected, 'NumDimensions', 3, 'Perplexity', 30);

figure;
scatter3(Y(:,1), Y(:,2), Y(:,3), 15, y, 'filled');
title('t-SNE Visualization of Fault Classes');
xlabel('Dimension 1');
ylabel('Dimension 2');
zlabel('Dimension 3');
colormap(jet(num_fault_types+1));
colorbar('Ticks', 0:num_fault_types, 'TickLabels', fault_types);
grid on;