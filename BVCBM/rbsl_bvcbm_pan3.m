clc;clear;
addpath("Model/")

load('CancerDatasets.mat');
load('cov_matrix_pan3.mat','cov_matrix')

cov_rw = cov_matrix;

ssy = Pancreatic_data(1:32,3); 

n = 350;
M = 10000;

start = [350, 16, 150];
reg_mean = 0.5;

tic;
[theta,loglike,gamma] = rbsl_bvcbm(ssy',n,M,cov_rw,start,reg_mean,'res/rbsl_pan3_');
time = toc;

%%
save('res/rbsl_pan3_result.mat')