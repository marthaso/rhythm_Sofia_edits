%% Prep the files for calcium transient duration map calculation

% Inputs:
% study - string containing the study's number
% file_number - the file you want to evaluate
% pixel_x - one number for the x coordinate of a good pixel
% pixel_y - one number of the y coordinate of a good pixel
% firstAP - the first action potential (beat) you want to evaluate
% lastAP - the last action potential (beat) you want to evaluate (can be a
%          blank vector if you want to evaluate all the beats)
% prom - prominence of peaks to identify beats
% start_off - if needed, the start of each beat can have an offset instead
%             of starting at the previous beat. useful if the code is identifying two
%             beats or more in many pixels.
% end_off - same as start_off but truncating the end of the beat
% minapd - minimum acceptable value for the calcium transient duration
% maxapd - maximum acceptable value for the calcium transient duration
% percent - percent of repolarization you want to compute (usually 50 or
% 80)

% Output:
% CaT_mat = calcium transient duration map (256 x 256 matlab matrix)
% This code will also write an excel sheet storing the average calcium
% transient duration for each beat.

function [CaT_mat] = CaT_prep(study, file_number, pixel_x,pixel_y, firstAP, lastAP, prom,start_off,end_off,minapd,maxapd, percent)
%% Create a file where you will store your results
filename_pdf =fullfile(pwd,sprintf('%s_CaT%g.ps',study,percent));
fs = figure('visible','on');
set(gca,'visible','off')
text(0,0.9,study,'FontWeight','bold')
print(fs,filename_pdf,'-dpsc','-fillpage')
close(gcf)

%% Load the data
% First check if the file already exists, otherwise, load the raw file and
% convert it to .mat
if ~isfile(sprintf('%s_%g_Ca_bin3_100Hz.mat',study,file_number))
    % Find raw file in current directory
    file_name = sprintf('%g_CAM2.raw',file_number);
    data = CMOSconverter(fullfile(pwd, file_name));
    % get mask
    mask = load(sprintf('%s_Ca.txt',study));
    % bin data
    kernel_name = 'gaussian';
    kernel_size = 3;
    [data] = binning(data, mask, kernel_size, kernel_name);
    % filter data
    data = filter_data(data, 1000, 100, 0.5, 100);
    data = remove_60hz(data, 1000);
    % % Invert data for Ca
    data=-data+max(data(:))+min(data(:));
    data = normalize_data(data);
    title = sprintf('%s_%g_Ca_bin3_100Hz.mat',study,file_number,b,f);
    save(title,'data','-v7.3');

end
data = load(sprintf('%s_%g_Ca_bin3_100Hz.mat',study,file_number));
data = struct2cell(data);
data{1,1} = data{1,1}(:,:,1:4999);
data= cell2mat(data);
data = normalize_data(data);

%% Plot one good pixel and find peaks
pixel_calc = data(pixel_x,pixel_y,:);
pixel_calc = normalize_data(pixel_calc);
pixel_calc = squeeze(pixel_calc);
figure
findpeaks(pixel_calc,'MinPeakDistance',50,'MinPeakProminence',prom,'MaxPeakWidth',120)
[pks, locs] =findpeaks(pixel_calc,'MinPeakDistance',50,'MinPeakProminence',prom,'MaxPeakWidth',120);

if isempty(lastAP)
    lastAP = length(locs);
end

%% Find start/end points
figure
plot(1:4999,pixel_calc,locs(firstAP:lastAP),pks(firstAP:lastAP),'o')
print(gcf,filename_pdf,'-dpsc','-bestfit','-append')

if firstAP == 1
    start = [1, (locs(1:lastAP-1)+start_off)'];
else
    start = (locs(firstAP-1:lastAP-1)+start_off)';
end
if lastAP == length(locs)
    endp = [(locs(firstAP+1:end)-end_off)', 4999];
else
    endp = [(locs(firstAP+1:lastAP+1)-end_off)'];
end

%% Calculate CaT duration for the APs specified
[CaT_avgs] = [];
for i = 1: length(start)
    [CaT_mat] = CaTMap_calculation(data, start(i), endp(i), minapd, maxapd, percent, filename_pdf,study,prom);
    CaT_avg = nanmean(CaT_mat(:));
    CaT_avgs = [CaT_avgs; CaT_avg];
end

%% Create a table with the CaT durations
T = table(CaT_avgs, 'VariableNames',{sprintf('CaT%g',percent)});
filename = sprintf('%s_CaT%g_file_%g.xlsx',study,percent,file_number);
writetable(T, filename)
end