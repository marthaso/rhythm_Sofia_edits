%% APD calculation. PCL = 200 ms

filename_pdf =fullfile('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2','STUDY_CaT_all.ps');
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
lastAP = []; 
file_number = [FILENUMBER];
APD50_vec_mat = [];
APD80_vec_mat = [];
for i=1:length(file_number)
    disp('new apd')
    [APD50_vec, APD80_vec] = auto_CaT_STUDY(start, endp, study, file_number(i),filename_pdf, AP_num,firstAP,lastAP);
    APD50_vec_mat = [APD50_vec_mat, APD50_vec'];
    APD80_vec_mat = [APD80_vec_mat, APD80_vec'];
    close all
end

T = table(APD50_vec_mat, APD80_vec_mat, 'VariableNames',{'CaT50', 'CaT80'});
filename = 'STUDY_file_FILENUMBER_CaT_all.xlsx';
writetable(T, filename)


close all