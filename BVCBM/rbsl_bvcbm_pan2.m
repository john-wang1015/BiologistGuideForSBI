clc;clear;
addpath("Model/")

load('CancerDatasets.mat');
load('cov_matrix_pan2.mat','cov_matrix')

cov_rw = cov_matrix;

ssy = Pancreatic_data(1:26,2); 

n = 350;
M = 10000;

start = [500, 9, 170];
reg_mean = 0.5;


tic;
[theta,loglike,gamma] = rbsl_bvcbm(ssy',n,M,cov_rw,start,reg_mean,'res/rbsl_pan2_');
time = toc;

%%
save('res/rbsl_pan2_result.mat')