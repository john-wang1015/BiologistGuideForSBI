function sx = simulator_FUCCI(theta, T_record, CellTracking, Xmax, Ymax)





s = SetupStruct(ntrack, Xmax, Ymax, InitPosData, CellTrackingData);

[SummaryStatData, ExitSimStatus] = Main_simulate(theta, s, T_record, CellTracking);
                
sx = GenerateSummaryStatistics(SummaryStatData, CellTracking, Xmax);





