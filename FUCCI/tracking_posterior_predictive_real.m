clc;clear;
load('CellTracking_synthetic_dataset.mat')
%theta_true = theta;
y_true = sy;

load('res/results_SMCABC_real_track20.mat')
%load('res/tracking_syn2_progress26.mat','theta_trans')
theta_smc = theta;

load('res/snpe_c_nsf fucci tracking, round 9.mat')
theta_snpe = double(posterior);

load('res/snle_c_nsf fucci tracking, round 9.mat')
theta_snle = double(posterior);

InitPosData = readmatrix("Data/DataProcessing/FUCCI_processed.xlsx", "sheet","InitPos");
CellTrackingData = readmatrix("Data/DataProcessing/FUCCI_processed.xlsx", "sheet","CellTracking");

N = 1000; nvars = 6;

sx_smc = zeros(N, nvars);
sx_snpe = zeros(10000, nvars);
sx_snle = zeros(10000, nvars);

parfor i = 1:N
    i
    sx_smc(i,:) = simulator_tracking_nonparall(theta_smc(i,:), nvars, InitPosData, CellTrackingData);
end

parfor i = 1:10000
    i
    sx_snpe(i,:) = simulator_tracking_nonparall(theta_snpe(i,:), nvars, InitPosData, CellTrackingData);
    sx_snle(i,:) = simulator_tracking_nonparall(theta_snle(i,:), nvars, InitPosData, CellTrackingData);
end

save('res/tracking_real_posterior_predictive_results.mat');

