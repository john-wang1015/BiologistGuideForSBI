clc;clear;
load('CellTracking_synthetic_dataset.mat');
load('rbsl_cov_tracking_real.mat','cov_rm')

cov_rw = cov_rm;

ssy = sy1; theta_true = theta1;

n = 150;
M = 10000;

start = [0.02, 0.19, 0.19, 0.2, 0.5, 1.2];
reg_mean = 0.5;

tic;
[theta,loglike,gamma] = rbsl_cell_tracking(ssy,n,M,cov_rw,start,reg_mean,'res/bsl/rbsl_tracking/rbsl_tracking_');
time = toc;

save('res/bsl/rbsl_tracking/rbsl_tracking_result.mat')