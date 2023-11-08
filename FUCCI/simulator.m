function sx = simulator(CellTracking, theta)
    %load data
    InitPosData = readmatrix("Data/DataProcessing/FUCCI_processed.xlsx", "sheet","InitPos");
    CellTrackingData = readmatrix("Data/DataProcessing/FUCCI_processed.xlsx", "sheet","CellTracking");
    ntrack = max(CellTrackingData(:,4));
    
    Xmax = 1309.09; %Length of the domain
    Ymax = 1745.35; %Width of the domain
    T_record = 48;

    s = SetupStruct(ntrack, Xmax, Ymax, InitPosData, CellTrackingData);

    [SummaryStatData, ~] = Main_simulate(theta, s, T_record, CellTracking);

    sx = GenerateSummaryStatistics(SummaryStatData, CellTracking, Xmax); 