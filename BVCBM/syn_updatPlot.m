clc;clear;
addpath("Model\");
load('BVCBM_SNPE_pan1_theta20k.mat')

%%
observation = double(observation);
sims_old = double(sims_old);
theta_old = double(theta_old);

parfor i = 1:10000
    if theta_new(i,5) > 26
        theta_new(i,5) = 25;
    end
end
theta_new = abs(double(theta_new));

%%
N = 10000;

sim_params.page = 2;
sim_params.max_time = 26;
sim_params.startingvol = observation(1);

post_pred_sim = zeros(N,sim_params.max_time);

parfor i = 1:N
    i
    post_pred_sim(i,:) = simulator(theta_new(i,:), sim_params, sim_params.max_time);
end

%%
old_len = length(theta_old);
len = old_len + N;

sims = zeros(len,26);
theta = zeros(len,9);

sims(1:old_len,:) = sims_old;
sims((old_len+1):len,:) = post_pred_sim;

theta(1:old_len,:) = theta_old;
theta((old_len+1):len,:) = theta_new;

%%
save("Calibration_SNPE_pan2_sims30k.mat")

%%
%load("synthetic_dataset.mat");

figure
for i = 1:9
    subplot(3,3,i)
    [f,xi] = ksdensity(theta_new(:,i));
    plot(xi,f,'color',[173,221,142]/255,'LineWidth',3);
    %hold on
    %plot(theta4(i),0,'rx','LineWidth',3)
end

%%
figure
quant1_95CI = quantile(post_pred_sim,[0.025 0.975]);
quant1_75CI = quantile(post_pred_sim,[0.25 0.75]);
quant1_90CI = quantile(post_pred_sim,[0.1 0.9]);
len1 = 1:length(observation);
cmap1 = [150,150,150;150,150,150;150,150,150]/255;
cmap2 = [107,174,214;107,174,214;107,174,214]/255;
cmap3 = [8,48,107;8,48,107;8,48,107]/255;
h175 = fill([len1, fliplr(len1)], [quant1_75CI(1,:), fliplr(quant1_75CI(2,:))],cmap3(1,:),'LineStyle','none');
hold on
h190 = fill([len1, fliplr(len1)], [quant1_90CI(1,:), fliplr(quant1_90CI(2,:))],cmap2(1,:),'LineStyle','none');
h195 = fill([len1, fliplr(len1)], [quant1_95CI(1,:), fliplr(quant1_95CI(2,:))],cmap1(1,:),'LineStyle','none');
set(h195,'facealpha',.4)
set(h190,'facealpha',.5)
set(h175,'facealpha',.8)
plot(1:length(observation),observation,'k','LineWidth',4);
legend('25% - 75%','10% - 90%','2.5% - 97.5%')
xlabel('time (days)','fontsize',24)
ylabel('Tumour size','fontsize',24)
xlim([1 32])
set(gca,'FontSize',24)