clc;clear;
addpath("Model\")
load("BVCBM_syn_dataset.mat");
y1 = syn1;
y2 = syn2;
y3 = syn3;

%%
% std_rw = mean(part_vals);
% corr_rw = corr(part_vals);
% cov_rw = cov(part_vals);
n_vars = 9;
cov_rw = eye(n_vars);
N = 1;
n = 100;
M = 50000;

sim_params1.page = 2;
sim_params1.max_time = length(y2);
sim_params1.startingvol = y2(1);

theta_BSL = bayes_sl(y2, N, M, n, cov_rw, sim_params1);

%%
% load('BVCBM_syn_dataset.mat')
% 
% figure
% for i = 1:9
%     subplot(3,3,i)
%     [f1,xi1] = ksdensity(theta_BSL(:,i));
%     plot(xi1,f1,'color',[8,48,107]/255,'LineWidth',3);
%     hold on 
%     xline(theta1(i),'LineWidth',3)
%     set(gca,'FontSize',24)
% end

%%
save('bsl_synthetic2.mat')