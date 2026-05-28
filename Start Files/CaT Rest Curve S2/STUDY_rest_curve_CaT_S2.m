%% APD calculation. PCL = 200 ms

filename_pdf =fullfile('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2','STUDY_CaT_rest_curve_S2.ps');
fs = figure('visible','on');
set(gca,'visible','off')
text(0,0.9,'STUDY','FontWeight','bold')
print(fs,filename_pdf,'-dpsc','-fillpage')
close(gcf)
study = 'STUDY';
start = 0.001;
endp = 4.999;
AP_num = [1,1,1,1,1];
firstAP = 1;
lastAP = []; %
file_number = [4,5,6,7];
AP_num = [10, 16, 17, 18];
%AP_num =      [10, 17, 21, 11, 16, 7 , 9,  12, 12];
CL =          [180,160,140,120];
APD50_vec_mat = [];
APD80_vec_mat = [];
for i=1:length(file_number)
    disp(file_number(i))
    [APD50_vec, APD80_vec] = auto_CaT_STUDY_rest_curve_S2(start, endp, study, file_number(i),filename_pdf, AP_num(i),firstAP,lastAP);
    APD50_vec_mat = [APD50_vec_mat, APD50_vec'];
    APD80_vec_mat = [APD80_vec_mat, APD80_vec'];
     close all
end

T = table(CL', APD50_vec_mat', APD80_vec_mat', 'VariableNames',{'CL','CaT50', 'CaT80'});
filename = 'STUDY_CaT_rest_curve_S2.xlsx';
writetable(T, filename)


close all