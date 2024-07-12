clc;clear;
addpath("Model/")

load('CancerDatasets.mat');
load('cov_matrix_pan1.mat','cov_matrix')

cov_rw = cov_matrix;

ssy = Pancreatic_data(1:19,1); 

n = 350;
M = 10000;

start = [200, 12, 50];
reg_mean = 0.5;


tic;
[theta,loglike,gamma] = rbsl_bvcbm(ssy',n,M,cov_rw,start,reg_mean,'res/rbsl_pan1_');
time = toc;

%%
save('res/rbsl_pan1_result.mat')