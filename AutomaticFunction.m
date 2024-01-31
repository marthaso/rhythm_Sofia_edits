function [handles,saveDataLat,saveDataCV,saveDataCVVec, ...
          meanresults, medianresults,SDresults,...
          angleresults,num_membersresults] = AutomaticFunction(filename,filemask,...
        c_start,c_end,varargin)
%AUTOMATICFUNCTION Automtic background removal and CV map
%   Uses the rhythm function to remove the background and create CV map
%   This function is for batch processing.




%filename ='/data/data/Project-Fibrosis/Optical/Control/G19009/right/right2019-10-18-160750_Ultima (IF1-CAM1).mat';
%filemask = '/data/data/Project-Fibrosis/Optical/Control/G19009/mask6.txt';
%c_start = 2.8;
%c_end = 3.1; %[s]
% Create a standard configuration
handles = rhythmHandles;
handles.activeCamData.xres = 0.17 ;%[mm]
handles.activeCamData.yres = 0.17 ;%[mm]
removeBG_state          = 1;
bin_state               = 1;
kernel_size= 3;
kernel_name='gaussian';
filt_state              = 1;
filt_pop_state          = 3;
filt_60hz_state         = 0;
drift_state             = 1;
method_name ='polynomial';
first_drift_param =1;
norm_state              = 1;
inverse_state           = 0;
ensemble_state          = 0;
smoothness_param_power  = 9;
asym_param              = 0.05;
rect =[1 1 99 99];

%look for config values
for i=1:(nargin-4)/2
    switch( varargin{i*2-1})
        case 'smoothness_param_power'
            smoothness_param_power=varargin{i*2};
        case 'asym_param'
            asym_param=varargin{i*2};
        case 'xres'
            handles.activeCamData.xres=varargin{i*2};
        case 'yres'
            handles.activeCamData.yres=varargin{i*2};
        case 'bin_state'
            bin_state =varargin{i*2};
        case 'kernel_size'
            kernel_size =varargin{i*2};
        case 'kernel_name'
            kernel_name =varargin{i*2};
        case 'filt_state'
            filt_state =varargin{i*2};
        case 'removeBG_state'
            removeBG_state =varargin{i*2};
        case 'filt_60hz_state'
            filt_60hz_state =varargin{i*2};
        case 'drift_state'
            drift_state =varargin{i*2};
        case 'method_name'
            method_name =varargin{i*2};
        case 'first_drift_param'
            first_drift_param =varargin{i*2};
        case 'norm_state'
            norm_state =varargin{i*2};
        case 'rect'
            rect =varargin{i*2};
        case 'filt_pop_state'
            filt_pop_state =varargin{i*2};
        otherwise
            warning(['SockMeshPlot does not know option ',varargin{i*2-1} ])
    end
end

%openFile
rawload=load(filename);
handles.activeCamData.cmosRawData=rawload.cmosData;
%handles.activeCamData.bg = double(rawload.bgimage);


handles.activeCamData.Fs=rawload.frequency;
handles.Fs=rawload.frequency;
mask = load(filemask);
mask = logical(mask);
handles.activeCamData.finalSegmentation = mask;

if length(size(rawload.bgimage)) == 3
    handles.activeCamData.bg = double(rgb2gray(rawload.bgimage));
else
    handles.activeCamData.bg = double(rawload.bgimage);
end

handles.activeCamData.bgRGB = real2rgb(handles.activeCamData.bg,'gray');

% Create variable for tracking conditioning progress
trackProg = [removeBG_state,...
    filt_state,...
    filt_60hz_state,...
    bin_state,...
    drift_state,...
    norm_state,...
    inverse_state,...
    ensemble_state];
trackProg = sum(trackProg);

counter = 0;
g1 = waitbar(counter,'Conditioning Signal');

% Return to raw unfiltered cmos data
handles.normflag = 0; % Initialize normflag
handles.activeCamData.cmosData = handles.activeCamData.cmosRawData;
handles.activeCamData.drawMap = 0;
handles.activeCamData.drawPhase = 0;
handles.matrixMax = 1;
handles.normalizeMinVisible = 0.3;

maxFrame = size(handles.activeCamData.cmosRawData, 3);
handles.activeCamData.maxFrame = maxFrame;
handles.maxFrame = maxFrame;

mask = handles.activeCamData.finalSegmentation;
%if handles.drawSegmentation == 0
%    mask = ones(size(mask));
%end

%% Remove Background
if removeBG_state == 1
    % Update counter % progress bar
    counter = counter + 1;
    waitbar(counter/trackProg,g1,'Removing Background');
    handles.activeCamData.cmosData =...
        handles.activeCamData.cmosData.* repmat(mask,...
        [1 1 size(handles.activeCamData.cmosData, 3)]);
    
    %set(removeBG_button,'Value',0);
    %removeBGcheckbox_callback(hObject);
    %handles.drawSegmentation = 0;
    %handles.drawBrush = 0;
end

%% Bin Data
if bin_state == 1
    % Update counter % progress bar
    counter = counter + 1;
    waitbar(counter/trackProg, g1, 'Binning Data');
    
    %bin_pop_state = kernel_size;%get(kernel_popup,'Value');
    %kernel_name_list = get(kernel_popup, 'String');
    %kernel_name = kernel_name_list{bin_pop_state};
    
    %kernel_size = str2double(get(kernel_size_edit, 'String'));
    
    handles.activeCamData.cmosData = binning(handles.activeCamData.cmosData, mask, kernel_size, kernel_name);
end

%% Filter Data
if filt_state == 1
    % Update counter % progress bar
    counter = counter + 1;
    waitbar(counter/trackProg,g1,'Filtering Data');
    %filt_pop_state = get(filt_popup,'Value');
    if filt_pop_state == 4
        or = 100;
        lb = 0.5;
        hb = 150;
    elseif filt_pop_state == 3
        or = 100;
        lb = 0.5;
        hb = 100;
    elseif filt_pop_state == 2
        or = 100;
        lb = 0.5;
        hb = 75;
    else
        or = 100;
        lb = 0.5;
        hb = 50;
    end
    handles.activeCamData.cmosData = filter_data(handles.activeCamData.cmosData,...
        handles.activeCamData.Fs,...
        or, lb, hb);
end
% %
% % %% Remove 60 Hz hum
% % if filt_60hz_state == 1
% %     % Update counter % progress bar
% %     counter = counter + 1;
% %     waitbar(counter/trackProg, g1, 'Removing 60 Hz hum');
% %     handles.activeCamData.cmosData = remove_60hz(handles.activeCamData.cmosData,...
% %         handles.activeCamData.Fs);
% % end

%% Remove Drift
if drift_state == 1
    % Update counter % progress bar
    counter = counter + 1;
    waitbar(counter/trackProg,g1,'Removing Drift');
    
    %drift_popup_state = get(drift_popup,'Value');
    %method_name_list = get(drift_popup, 'String');
    
    %method_name = method_name_list{drift_popup_state};
    
    %first_drift_param = str2double(get(first_drift_param_edit, 'String'));
    %first_drift_param = round(first_drift_param);
    %set(first_drift_param_edit,'String',num2str(first_drift_param));
    
    if strcmp(method_name, 'polynomial')
        order = first_drift_param;
        method_params = [order];
    elseif strcmp(method_name, 'asLS')
        n_iter = first_drift_param;
        %smoothness_param_power = get(smooth_param_slider, 'Value');
        smoothness_param = 10^smoothness_param_power;
        %asym_param = str2double(get(asym_param_edit, 'String'));
        method_params = [smoothness_param, asym_param, n_iter];
    end
    
    handles.activeCamData.cmosData = remove_Drift(handles.activeCamData.cmosData, mask,...
        method_name, method_params);
end

% % %% Full Ensemble Average
% % if ensemble_state == 1
% %     counter = counter + 1;
% %     waitbar(counter/trackProg,g1,'Full Ensemble Average');
% %     CL = str2double(get(ensembleAverageFull_edit, 'String'));
% %     handles.activeCamData.cmosData = ensembleAverageFull(handles.activeCamData.cmosData,...
% %         CL, handles.activeCamData.Fs);
% %
% %     maxFrame = size(handles.activeCamData.cmosData, 3);
% %     handles.maxFrame = maxFrame;
% %     handles.activeCamData.maxFrame = maxFrame;
% % end


%% Inverse Data
if inverse_state == 1
    counter = counter + 1;
    waitbar(counter/trackProg,g1,'Inversing Data');
    handles.activeCamData.cmosData=-handles.activeCamData.cmosData+max(handles.activeCamData.cmosData(:))+min(handles.activeCamData.cmosData(:));
end

%% Normalize Data
if norm_state == 1
    % Update counter % progress bar
    counter = counter + 1;
    waitbar(counter/trackProg,g1,'Normalizing Data');
    handles.activeCamData.cmosData = normalize_data(handles.activeCamData.cmosData);
    handles.normflag = 1;
end

%% Delete the progress bar
delete(g1)

%% Save conditioned signal
hObject.UserData = handles.activeCamData.cmosData;
data = handles.activeCamData.cmosData;

fNull = figure('visible','off');
f1 = figure('visible','off');
f2 = figure('visible','off');
handles.meanresults = uicontrol('Parent',fNull,'Style','text','Visible','off');
handles.medianresults = uicontrol('Parent',fNull,'Style','text','Visible','off');
handles.SDresults = uicontrol('Parent',fNull,'Style','text','Visible','off');
handles.num_members_results = uicontrol('Parent',fNull,'Style','text','Visible','off');
handles.angleresults = uicontrol('Parent',fNull,'Style','text');
movieScreen1 = axes('Parent',fNull,'Units','normalized','YTick',[],'XTick',[],...
    'Units','normalized','Position',[0.05, 0.05, 0.9, 0.9],...
    'color', 'black','box','on', 'linewidth',2, ...
    'CameraUpVector',[0,1,1], 'YDir','reverse');

movieScreen2 = axes('Parent',fNull,'Units','normalized','YTick',[],'XTick',[],...
    'Units','normalized','Position',[0.05, 0.05, 0.9, 0.9],...
    'color', 'black','box','on', 'linewidth',2, ...
    'CameraUpVector',[0,1,1], 'YDir','reverse');

handles.activeCamData.cmap = colormap('Jet');
numOfBeats=length(c_start);
saveDataLat= cell(numOfBeats,1);
saveDataCV= cell(numOfBeats,1);
saveDataCVVec= cell(numOfBeats,1);
meanresults=zeros(numOfBeats,1);
medianresults=zeros(numOfBeats,1);
SDresults=zeros(numOfBeats,1);
angleresults=zeros(numOfBeats,1);
num_membersresults=zeros(numOfBeats,1);
for i= 1:numOfBeats
    aMap(handles.activeCamData.cmosData,c_start(i),c_end(i),...
        rect,... % rectangle of ROI coords
        handles.activeCamData.Fs,handles.activeCamData.cmap,...
        movieScreen1, handles);
    saveDataLat{i}=handles.activeCamData.saveData;
    
    cMap(handles.activeCamData.cmosData,c_start(i),c_end(i),...
        handles.activeCamData.Fs,handles.activeCamData.bg,rect, f2,...
        movieScreen2, handles);
    saveDataCV{i}=handles.activeCamData.saveData;
    saveDataCVVec{i}= handles.activeCamData.VecArray;
    %extract the values
    temp =strsplit(handles.activeCamData.meanresults);
    meanresults(i)= str2double(temp{end});
    temp =strsplit(handles.activeCamData.medianresults);
    medianresults(i)= str2double(temp{end});
    temp =strsplit(handles.activeCamData.SDresults);
    SDresults(i)= str2double(temp{end});
    temp =strsplit(handles.activeCamData.angleresults);
    angleresults(i)= str2double(temp{end});
    temp =strsplit(handles.activeCamData.num_membersresults);
    num_membersresults(i)= str2double(temp{end});
end
close(fNull)
close(f1)
close(f2)
end

