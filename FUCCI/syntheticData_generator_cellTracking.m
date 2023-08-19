clc;clear;

rng(1234);

T_record = 48;

t_lb = 0; t_ub = 1;
m_lb = 0; m_ub = 1;
num_params = 6;

InitPosData = readmatrix("C:\Users\aufb\Dropbox\biologists' guide for sbi\FUCCI-main/Data/DataProcessing/FUCCI_processed.xlsx", "sheet","InitPos");
FinalPosData = readmatrix("C:\Users\aufb\Dropbox\biologists' guide for sbi\FUCCI-main/Data/DataProcessing/FUCCI_processed.xlsx", "sheet","FinalPos");
CellTrackingData = readmatrix("C:\Users\aufb\Dropbox\biologists' guide for sbi\FUCCI-main/Data/DataProcessing/FUCCI_processed.xlsx", "sheet","CellTracking");
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


%sy = [Nred, Nyellow, Ngreen, TotalDistance]; 
s = SetupStruct(ntrack, Xmax, Ymax, InitPosData, CellTrackingData);

%summaries = zeros(N,length(sy));
theta_true = [0.3, 0.36, 0.28, 4, 4, 4];

[SummaryStatData, ExitSimStatus] = Main_simulate(theta_true, s, T_record, CellTracking);

synthetic_data = GenerateSummaryStatistics(SummaryStatData, CellTracking, Xmax);

%%
sy = synthetic_data;
save("syntheticData4_cellTracking.mat")

