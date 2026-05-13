%% Run this file

% Before you run make sure you have:
% Raw file you are trying to analyze
% Mask in .txt format

filename_pdf =fullfile('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2','STUDY_APD_all.ps');
fs = figure('visible','on');
set(gca,'visible','off')
text(0,0.9,'STUDY','FontWeight','bold')
print(fs,filename_pdf,'-dpsc','-fillpage')
close(gcf)
study = 'STUDY';
% Normally we want the whole recording but can change accordingly.
start = 0.001;
endp = 4.999; 
% For APD analysis AP_num does not matter, no need to change it.
AP_num = 1;
% If you want to analyze all the APs you can set firstAP = 1 and lastAP =
% []. You can also choose specific APs to analyze.
firstAP = 1; %
lastAP = []; %
% Write in the number of the file you need to open (eg. 22)
file_number = [FILENUMBER];
APD50_vec_mat = [];
APD80_vec_mat = [];
for i=1:length(file_number)
    [APD50_vec, APD80_vec] = auto_APD_STUDY(start, endp, study, file_number(i),filename_pdf, AP_num,firstAP,lastAP);
    APD50_vec_mat = [APD50_vec_mat, APD50_vec'];
    APD80_vec_mat = [APD80_vec_mat, APD80_vec'];
    close all
end

T = table(APD50_vec_mat, APD80_vec_mat, 'VariableNames',{'APD50', 'APD80'});
% Change name as needed
filename = 'STUDY_file_FILENUMBER_APD_all.xlsx';
writetable(T, filename)


close all