 load('APD_data_4.mat');
 figure
 imagesc(apdMap)
 colorbar
 avg = nanmean(nanmean(apdMap));
 apdMap_new = apdMap - avg;
 figure
 imagesc(apdMap_new)
 colorbar