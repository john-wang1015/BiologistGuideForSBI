clc;clear;
load('SNPE_nsf_syn2_celltrack_20k.mat')
rng(1234);

N = 10000;
T_record = 48;

t_lb = 0; t_ub = 1;
m_lb = 0; m_ub = 1;
num_params = 6;

InitPosData = readmatrix("C:/Users/aufb/Dropbox/biologists' guide for sbi/FUCCI-main/Data/DataProcessing/FUCCI_processed.xlsx", "sheet","InitPos");
FinalPosData = readmatrix("C:/Users/aufb/Dropbox/biologists' guide for sbi/FUCCI-main/Data/DataProcessing/FUCCI_processed.xlsx", "sheet","FinalPos");
CellTrackingData = readmatrix("C:/Users/aufb\Dropbox/biologists' guide for sbi/FUCCI-main/Data/DataProcessing/FUCCI_processed.xlsx", "sheet","CellTracking");
ntrack = max(CellTrackingData(:,4));
    
Xmax = 1309.09; %Length of the domain
Ymax = 1745.35; %Width of the domain
    
%summary statistics: 
Nred = sum(FinalPosData(:,3) == 1);
Nyellow = sum(FinalPosData(:,3) == 2);
Ngreen = sum(FinalPosData(:,3) == 3);
CellTracking = true;

if CellTracking == true%if using cell tracking summary statistics
    %compute distance travelled through each cell phase by all tracked cells
    RedDistance = 0;
    YellowDistance = 0;
    GreenDistance = 0;
    j = 2;
    for i = 1:ntrack
       while CellTrackingData(j,4) == i
           if CellTrackingData(j,3) == 1 && CellTrackingData(j-1,3) == 1
               RedDistance = RedDistance + sqrt((CellTrackingData(j,1)-CellTrackingData(j-1,1))^2+(CellTrackingData(j,2)-CellTrackingData(j-1,2))^2);
           elseif CellTrackingData(j,3) == 2 
               YellowDistance = YellowDistance + sqrt((CellTrackingData(j,1)-CellTrackingData(j-1,1))^2+(CellTrackingData(j,2)-CellTrackingData(j-1,2))^2);
           elseif CellTrackingData(j,3) == 3 
               GreenDistance = GreenDistance + sqrt((CellTrackingData(j,1)-CellTrackingData(j-1,1))^2+(CellTrackingData(j,2)-CellTrackingData(j-1,2))^2);
           end
           j = j + 1;
           if (j > size(CellTrackingData,1))
               break
           end
       end
    end

   TotalDistance = [RedDistance/ntrack, YellowDistance/ntrack, GreenDistance/ntrack];
else %if using cell density summary statistics

    %compute median positon of cells on LHS and RHS of scratch
    RedDistanceMed = [median(FinalPosData(FinalPosData(:,3) == 1 & FinalPosData(:,1) <= Xmax/2, 1)), median(FinalPosData(FinalPosData(:,3) == 1 & FinalPosData(:,1) > Xmax/2, 1))];
    YellowDistanceMed = [median(FinalPosData(FinalPosData(:,3) == 2 & FinalPosData(:,1) <= Xmax/2, 1)), median(FinalPosData(FinalPosData(:,3) == 2 & FinalPosData(:,1) > Xmax/2, 1))];
    GreenDistanceMed = [median(FinalPosData(FinalPosData(:,3) == 3 & FinalPosData(:,1) <= Xmax/2, 1)), median(FinalPosData(FinalPosData(:,3) == 3 & FinalPosData(:,1) > Xmax/2, 1))];

    %compute IQR of cells on the LHS and RHS of scratch
    RedDistanceIQR = [iqr(FinalPosData(FinalPosData(:,3) == 1 & FinalPosData(:,1) <= Xmax/2, 1)), iqr(FinalPosData(FinalPosData(:,3) == 1 & FinalPosData(:,1) > Xmax/2, 1))];
    YellowDistanceIQR = [iqr(FinalPosData(FinalPosData(:,3) == 2 & FinalPosData(:,1) <= Xmax/2, 1)), iqr(FinalPosData(FinalPosData(:,3) == 2 & FinalPosData(:,1) > Xmax/2, 1))];
    GreenDistanceIQR = [iqr(FinalPosData(FinalPosData(:,3) == 3 & FinalPosData(:,1) <= Xmax/2, 1)), iqr(FinalPosData(FinalPosData(:,3) == 3 & FinalPosData(:,1) > Xmax/2, 1))];

    TotalDistance = [RedDistanceMed,YellowDistanceMed,GreenDistanceMed,RedDistanceIQR,YellowDistanceIQR,GreenDistanceIQR]; 

end

%%
load("syntheticData2_cellTracking.mat");
%sy = [Nred, Nyellow, Ngreen, TotalDistance]; 
s = SetupStruct(ntrack, Xmax, Ymax, InitPosData, CellTrackingData);

theta = zeros(N,num_params);
summaries = zeros(N,length(sy));
theta_temp = double(theta_new);

% creating initial sample
parfor i = 1:N
    i
    %while ~all(theta(i,:)) %repeat untill 1 proposed value is accepted
            % Draw theta ~ pi(theta)
            theta_prop = theta_temp(i,:);

            %simulate x ~ f(x|theta)
            [SummaryStatData, ExitSimStatus] = Main_simulate(theta_prop, s, T_record, CellTracking);

            if ExitSimStatus %check if simulation finished correctly
                continue;
            end 

            sx = GenerateSummaryStatistics(SummaryStatData, CellTracking, Xmax); %compute simulated summary stats
            if any(isnan(sx)) %check integrity of summary statistics
                continue  
            end 

            d = norm(sx - sy, 2); %discrepency function
            %store results
            theta(i,:) = theta_prop;
            summaries(i,:) = sx; %store simulated summary statistics
    %end
end

%%
dim_old = length(theta_old);
dim_new = N + dim_old;

theta_D = zeros(dim_new,num_params);
summaries_D = zeros(dim_new,length(sy));

theta_D(1:dim_old,:) = theta_old;
theta_D(dim_old+1:dim_new,:) = theta;

summaries_D(1:dim_old,:) = sims_old;
summaries_D(dim_old+1:dim_new,:) = summaries;

%%
save("SNLE_syn2_CellTrack_30k.mat");

%%
num_plots = length(sy);

figure
subplot(4,3,1)
[f,xi] = ksdensity(sims_old(:,1)); 
plot(xi,f,'LineWidth',3);
hold on 
xline(sy(1),'LineWidth',3)

subplot(4,3,2)
[f,xi] = ksdensity(sims_old(:,2)); 
plot(xi,f,'LineWidth',3);
hold on 
xline(sy(2),'LineWidth',3)

subplot(4,3,3)
[f,xi] = ksdensity(sims_old(:,3)); 
plot(xi,f,'LineWidth',3);
hold on 
xline(sy(3),'LineWidth',3)

subplot(4,3,4)
[f,xi] = ksdensity(summaries(:,1)); 
plot(xi,f,'LineWidth',3);
hold on 
xline(sy(1),'LineWidth',3)

subplot(4,3,5)
[f,xi] = ksdensity(summaries(:,2)); 
plot(xi,f,'LineWidth',3);
hold on 
xline(sy(2),'LineWidth',3)

subplot(4,3,6)
[f,xi] = ksdensity(summaries(:,3)); 
plot(xi,f,'LineWidth',3);
hold on 
xline(sy(3),'LineWidth',3)

% First 6
subplot(4,3,7)
[f,xi] = ksdensity(sims_old(:,4)); 
plot(xi,f,'LineWidth',3);
hold on 
xline(sy(1),'LineWidth',3)

subplot(4,3,8)
[f,xi] = ksdensity(sims_old(:,5)); 
plot(xi,f,'LineWidth',3);
hold on 
xline(sy(2),'LineWidth',3)

subplot(4,3,9)
[f,xi] = ksdensity(sims_old(:,6)); 
plot(xi,f,'LineWidth',3);
hold on 
xline(sy(3),'LineWidth',3)

subplot(4,3,10)
[f,xi] = ksdensity(summaries(:,4)); 
plot(xi,f,'LineWidth',3);
hold on 
xline(sy(1),'LineWidth',3)

subplot(4,3,11)
[f,xi] = ksdensity(summaries(:,5)); 
plot(xi,f,'LineWidth',3);
hold on 
xline(sy(2),'LineWidth',3)

subplot(4,3,12)
[f,xi] = ksdensity(summaries(:,6)); 
plot(xi,f,'LineWidth',3);
hold on 
xline(sy(3),'LineWidth',3)
