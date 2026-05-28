%% APD calculation for all CLs

filename_pdf =fullfile('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2','STUDY_APD_rest_curve_S1.ps');
fs = figure('visible','on');
set(gca,'visible','off')
text(0,0.9,'STUDY','FontWeight','bold')
print(fs,filename_pdf,'-dpsc','-fillpage')
close(gcf)
study = 'STUDY';
start = 0.001;
endp = 4.999;
AP_num = 1;
firstAP = 1;
lastAP = []; %
% write in the file numbers you will use (eg. file_number = [2, 3, 4, 5];)
file_number = [];
% Change the 5 to however many files you are analyzing
APD50_vec_mat = zeros(100,5);
APD80_vec_mat = zeros(100,5);
for i=1:length(file_number)
    file_number(i)
    [APD50_vec, APD80_vec] = auto_APD_STUDY_rest_curve_S1(start, endp, study, file_number(i),filename_pdf, AP_num,firstAP,lastAP);
    APD50_vec_mat(1:length(APD50_vec),i) = APD50_vec';
    APD80_vec_mat(1:length(APD50_vec),i) = APD80_vec';
    close all
end

T = array2table(APD50_vec_mat, 'VariableNames',{'APD5024', '25', '26', '27','28'});
filename = 'STUDY_APD50_rest_curve_S1.xlsx';
writetable(T, filename)

T = array2table(APD80_vec_mat, 'VariableNames',{'APD8024', '25', '26', '27','28'});
filename = 'STUDY_APD80_rest_curve_S1.xlsx';
writetable(T, filename)



close all