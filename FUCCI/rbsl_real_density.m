clc;clear;
load('CellDensity_synthetic_dataset.mat');
load('rbsl_cov_density_real.mat','cov_rm')

cov_rw = cov_rm;

ssy = sy1; theta_true = theta1;

n = 300;
M = 10000;

start = [0.01, 0.2, 0.15, 1, 5, 9.5];
reg_mean = 0.5;

tic;
[theta,loglike,gamma] = rbsl_cell_density(ssy,n,M,cov_rw,start,reg_mean,'res/bsl/rbsl_density/rbsl_density_');
time = toc;

%%
save('res/bsl/rbsl_density/rbsl_density_result.mat')