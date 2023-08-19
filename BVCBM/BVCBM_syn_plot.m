clc;clear;
load('BVCBM_syn_dataset.mat')

load('Synthetic1 , simultation 963500.mat')
smc_part_sims = part_sim;
smc_part_vals = part_vals;

load('BVCBM_SNPE_syn1_sims20k.mat')
npe_part_sims = post_pred_sims1;
npe_part_vals = part_vals1;

load('BVCBM_SNLE_syn1_sims20k.mat')
nle_part_sims = post_pred_sims1;
nle_part_vals = part_vals1;

%%
figure
subplot(1,3,1)
[f1,xi1] = ksdensity(smc_part_vals(:,4));
plot(xi1,f1,'color',[8,48,107]/255,'LineWidth',3);
hold on 
[f2,xi2] = ksdensity(npe_part_vals(:,4),'support',[0,1000]);
plot(xi2,f2,'--','color',[0,109,44]/255,'LineWidth',3);
[f3,xi3] = ksdensity(nle_part_vals(:,4),'support',[0,1000]);
plot(xi3,f3,'color',[84,39,143]/255,'LineWidth',3);
xline(theta1(4),'LineWidth',3)
legend('smc abc','npe','nle','true')
title('g_{age}^1')
set(gca,'FontSize',24)

subplot(1,3,2)
[f1,xi1] = ksdensity(smc_part_vals(:,9));
plot(xi1,f1,'color',[8,48,107]/255,'LineWidth',3);
hold on 
[f2,xi2] = ksdensity(npe_part_vals(:,9),'support',[0,1000]);
plot(xi2,f2,'--','color',[0,109,44]/255,'LineWidth',3);
[f3,xi3] = ksdensity(nle_part_vals(:,9),'Bandwidth',1,'support',[0,1000]);
plot(xi3,f3,'color',[84,39,143]/255,'LineWidth',3);
xline(theta1(9),'LineWidth',3)
title('g_{age}^2')
set(gca,'FontSize',24)
xlim([0,150])

subplot(1,3,3)
[f1,xi1] = ksdensity(smc_part_vals(:,5));
plot(xi1,f1,'color',[8,48,107]/255,'LineWidth',3);
hold on 
[f2,xi2] = ksdensity(npe_part_vals(:,5),'Bandwidth',1,'support',[0,32]);
plot(xi2,f2,'--','color',[0,109,44]/255,'LineWidth',3);
[f3,xi3] = ksdensity(nle_part_vals(:,5),'Bandwidth',1,'support',[0,32]);
plot(xi3,f3,'color',[84,39,143]/255,'LineWidth',3);
xline(theta1(5),'LineWidth',3)
title('\tau')
set(gca,'FontSize',24)

%%
figure
subplot(1,3,1)
quant1_95CI = quantile(smc_part_sims,[0.025 0.975]);
quant1_75CI = quantile(smc_part_sims,[0.25 0.75]);
quant1_90CI = quantile(smc_part_sims,[0.1 0.9]);
len1 = 1:length(y);
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
plot(1:length(y),y,'k','LineWidth',4);
legend('25% - 75%','10% - 90%','2.5% - 97.5%')
xlabel('time (days)','fontsize',24)
ylabel('Tumour size','fontsize',24)
xlim([1 32])
title('SMC ABC')
set(gca,'FontSize',24)

subplot(1,3,2)
quant1_95CI = quantile(npe_part_sims,[0.025 0.975]);
quant1_75CI = quantile(npe_part_sims,[0.25 0.75]);
quant1_90CI = quantile(npe_part_sims,[0.1 0.9]);
len1 = 1:length(y);
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
plot(1:length(y),y,'k','LineWidth',4);
xlabel('time (days)','fontsize',24)
ylabel('Tumour size','fontsize',24)
xlim([1 32])
title('SNPE')
set(gca,'FontSize',24)

subplot(1,3,3)
quant1_95CI = quantile(nle_part_sims,[0.025 0.975]);
quant1_75CI = quantile(nle_part_sims,[0.25 0.75]);
quant1_90CI = quantile(nle_part_sims,[0.1 0.9]);
len1 = 1:length(y);
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
plot(1:length(y),y,'k','LineWidth',4);
xlabel('time (days)','fontsize',24)
ylabel('Tumour size','fontsize',24)
xlim([1 32])
title('SNLE')
set(gca,'FontSize',24)