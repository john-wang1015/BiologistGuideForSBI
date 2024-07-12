clc;clear;
addpath("Model/")
load('paper2_BVCBM_synthetic_datasets.mat');
load('cov_matrix_syn1.mat');
cov_rw = cov_rm;

Y = sim1; theta_true = param1;

m = 350;
M = 10000;

% define prior
prior.num_params = 3;
prior.p1 = [2, 2, 2];
prior.p2 = [32*24-1, 31, 32*24-1];
prior.sampler = @() [unifrnd(prior.p1,prior.p2)]; 
prior.pdf = @(theta) prod(exp(theta)./(1 + exp(theta)).^2);
prior.trans_f = @(theta) [log((theta - prior.p1)./(prior.p2 - theta))];
prior.trans_finv = @(theta) [(prior.p2.*exp(theta) + prior.p1)./(1 + exp(theta))];

init = [theta_true(4), theta_true(5), theta_true(9)];   % initial value of chain NOT on transformed scale

%%
tic;
[theta,dist] = bsl_bvcbm(Y,init,m,M,cov_rw,prior,'res/bsl_syn1_');
finaltime=toc;


save('res/results_bsl_syn1.mat','theta','dist','finaltime');