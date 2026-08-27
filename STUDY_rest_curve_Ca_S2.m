%% CaT Calculation - RUN FILE

%% 1. create the pdf where we will publish our figures.
% Change directory to your rhythm folder. For the file name, change "STUDY"
% to the study number and the final number the file number you will use (leave blank
% if you don't know the file yet).
filename_pdf =fullfile('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2','STUDY_CaT_rest_curve_S2_#.ps');
fs = figure('visible','on');
set(gca,'visible','off')
% write in the study number
text(0,0.9,'STUDY','FontWeight','bold')
print(fs,filename_pdf,'-dpsc','-fillpage')
close(gcf)

%% 2. set parameters
% write in the study number
study = 'STUDY';
% change your time range if needed (usually 0.001 - 4.999)
start = 0.001;
endp = 4.999;
% no need to change AP_num
AP_num = 1;
% define your first and last action potential. If you don't know you can
% write firstAP = 1 and lastAP = [] to evaluate all the action potentials. 
firstAP = 1;
lastAP = 8; %
% write in the file number you want to analyze and its cycle length. For
% S2, we usually leave cycle_length = 180 for all the files.
file_number = #;
cycle_length = 180;
% Create vectors where we will store our results.
APD50_vec_mat = zeros(15,1);
APD80_vec_mat = zeros(15,1);

%% Run code
% The next line calls the function "auto_CaT_STUDY". This means it runs
% that file with the input parameters that we have previously specified.
% Make sure to change STUDY to the study number.
[APD50_vec, APD80_vec] = auto_CaT_STUDY(start, endp, study, file_number,filename_pdf, AP_num,firstAP,lastAP, cycle_length);
APD50_vec_mat(1:length(APD50_vec)) = APD50_vec';
APD80_vec_mat(1:length(APD50_vec)) = APD80_vec';
close all

%% Store our results in an excel sheet
% Change the study number (STUDY) and the file number (#) used
T = array2table(APD50_vec_mat, 'VariableNames',{'CaT50'});
filename = 'STUDY_CaT50_rest_curve_S2_#.xlsx';
writetable(T, filename)

T = array2table(APD80_vec_mat, 'VariableNames',{'CaT80'});
filename = 'STUDY_CaT80_rest_curve_S2_#.xlsx';
writetable(T, filename)




close all