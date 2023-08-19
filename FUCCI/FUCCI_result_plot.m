clc;clear;
%load('Real Data Calibration\SNLE Result\SNLE_CellDensity_40k_logexp.mat')
%theta_SNLE = theta_new;
%sims_SNLE = summaries;

load('SNPE_syn1_CellDensity_50k.mat')
theta_SNPE = theta_new;
sims_SNPE = summaries;

% load('CDandCT_smcabc.mat')

%%
parameter = ['Rr','Ry','Rg','Mr','My','Mg'];
j = 1;

figure
for i = 1:6
    subplot(2,3,i)
    [f1,xi1] = ksdensity(theta_SNPE(:,i));
    plot(xi1,f1, '-.','color',[35,139,69]/255,'LineWidth',3);
%     hold on
%     [f2,xi2] = ksdensity(CD_para(:,i));
%     plot(xi2,f2,'color',[54,144,192]/255,'LineWidth',3);
    if i == 1
        legend('SNPE','SMC ABC')
    end
    title(parameter(j:j+1))
    j = j + 2;
    set(gca,'FontSize',24)
end

%%
figure
for i = 1:15
    subplot(4,4,i)
    [f,xi] = ksdensity(sims_SNLE(:,i));
    plot(xi,f, '--','color',[203,24,29]/255,'LineWidth',3);
    hold on
    [f1,xi1] = ksdensity(sims_SNPE(:,i));
    plot(xi1,f1, '-.','color',[35,139,69]/255,'LineWidth',3);
    [f2,xi2] = ksdensity(CD_smcabc(:,i));
    plot(xi2,f2,'color',[54,144,192]/255,'LineWidth',3);
    xline(observation(:,i),'color',[0,0,0]/255,'LineWidth',3);
    if i == 1
        legend('SNLE','SNPE','SMC ABC')
    end
    string = "s" + num2str(i);
    title(string)
    set(gca,'FontSize',24)
end

%%
clc;clear;
load('Real Data Calibration\SNLE Result\SNLE_CellTrack_50k_logexp.mat')
theta_SNLE = theta_new;
sims_SNLE = summaries;

load('Real Data Calibration\SNPE Result\SNPE_CellTrack_50k.mat')
theta_SNPE = theta_new;
sims_SNPE = summaries;

load('CDandCT_smcabc.mat')

%%
parameter = ['Rr','Ry','Rg','Mr','My','Mg'];
j = 1;

figure
for i = 1:6
    subplot(2,3,i)
    [f,xi] = ksdensity(theta_SNLE(:,i));
    plot(xi,f, '--','color',[203,24,29]/255,'LineWidth',3);
    hold on
    [f1,xi1] = ksdensity(theta_SNPE(:,i));
    plot(xi1,f1, '-.','color',[35,139,69]/255,'LineWidth',3);
    [f2,xi2] = ksdensity(CD_para(:,i));
    plot(xi2,f2,'color',[54,144,192]/255,'LineWidth',3);
    if i == 1
        legend('SNLE','SNPE','SMC ABC')
    end
    title(parameter(j:j+1))
    j = j + 2;
    set(gca,'FontSize',24)
end

%%
figure
for i = 1:6
    subplot(2,3,i)
    [f,xi] = ksdensity(sims_SNLE(:,i));
    plot(xi,f, '--','color',[203,24,29]/255,'LineWidth',3);
    hold on
    [f1,xi1] = ksdensity(sims_SNPE(:,i));
    plot(xi1,f1, '-.','color',[35,139,69]/255,'LineWidth',3);
    [f2,xi2] = ksdensity(CD_smcabc(:,i));
    plot(xi2,f2,'color',[54,144,192]/255,'LineWidth',3);
    xline(observation(:,i),'color',[0,0,0]/255,'LineWidth',3);
    if i == 1
        legend('SNLE','SNPE','SMC ABC')
    end
    string = "s" + num2str(i);
    title(string)
    set(gca,'FontSize',24)
end