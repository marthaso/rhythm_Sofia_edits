%% APD calculation. PCL = 200 ms

filename_pdf =fullfile('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2','STUDY_APD_rest_curve_S2.ps');
fs = figure('visible','on');
set(gca,'visible','off')
text(0,0.9,'STUDY','FontWeight','bold')
print(fs,filename_pdf,'-dpsc','-fillpage')
close(gcf)
study = 'STUDY';
start = 0.001;
endp = 4.999;
% AP_num = 1;
firstAP = 1;
lastAP = []; %

file_number = [1,2,3,4];

AP_num =  [1,2,3,4];
 
CL = [140, 150, 160, 170];

APD50_vec_mat = [];
APD80_vec_mat = [];
for i=1:length(file_number)
    [APD50_vec, APD80_vec] = auto_APD_STUDY_rest_curve_S2(start, endp, study, file_number(i),filename_pdf, AP_num(i),firstAP,lastAP);
    APD50_vec_mat = [APD50_vec_mat, APD50_vec'];
    APD80_vec_mat = [APD80_vec_mat, APD80_vec']; 
    close all
end

T = table(CL', APD50_vec_mat', APD80_vec_mat', 'VariableNames',{'CL','APD50', 'APD80'});
filename = 'STUDY_APD_rest_curve_S2.xlsx';
writetable(T, filename)



close all