%% CaT Run File

% Function - run this file to compute calcium transient duration for 200 ms
% for all the studies

% [CaT_mat] = CaT_prep(study, file_number, pixel_x,pixel_y, firstAP, lastAP, prom,start_off,end_off,minapd,maxapd, percent)

% First study - R1
[R1_CaT80_mat] = CaT_prep('R1', 30, 150, 150, 1, [], 0.3, 0, 0, 50, 170,80);