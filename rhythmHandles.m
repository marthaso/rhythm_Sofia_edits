% rhythm handle class
% by Roman Syunyaev

classdef rhythmHandles < handle
    properties
        filename = [];
        cmosData = [];
        rawData = [];
        time = [];
        wave_window = 1; % this handle indicate the window number of the next wave displayed
        normflag = 0; % this handle indicate if normalize is clicked
        Fs = 1000; % this is the default value. it will be overwritten
        starttime = 0;
        fileLength = 1;
        endtime = 1;
        timeScale = 1;
        grabbed = -1;
        markers = []; % this handle stores the locations of the markers
        slide=-1; % parameter for recognize clicking location % this handle indicate if the movie slider is clicked
        %%minimum values pixels require to be drawn
        
        matrixMax=1;
        normalizeMinVisible = .3;
        
        cmap = []; %colormap('Jet'); %saves the default colormap values
        projectDir = '';
        dir='';
        dir_output='';
        file_list=[];
        filenames_loaded = ["", "", "", ""]; % filenames loaded for each window (4 of them) 
        frame=1;% this handles indicate the current frame being displayed by the movie screen
        bg=[];
        ecg=[];
        cmosRawData=[];
        bgRGB=[];
        
        signalScreens=[];
        
        movie_img=[];
        objectToDrawOn=[];
        playback=0;
        
        % Conduction Velocity
        c_start = 0;
        c_end = 0;
        contour_state = 0;
        a_start = 0;
        a_end = 0;
        Line = [];
        
        % APD and Alternance mapping
        apd_alternance_start_time = 0;
        apd_alternance_end_time = 1;
        minapd = 10;
        maxapd = 1000;
        percentAPD = 80;
        
        % FUNCTIONAL FOR 4 WINDOW VIEW
        activeScreen=[];
        activeScreenNo=0;
        expandedScreen=0;
        expandedScreenPos=[0.1400 0.1800 0.5100 0.8200];
        
        activeCamData=[];
        allCamData=[];
        
        maxFrame=0;
        dispWaveClicked=0;
        
        markerColors='bgrkcm'
        linked = 0;
        
        % bounds shows number of screen group: 0 for not linked, 1 or 2 for
        % linked. Maximum number of groups for 4 screens is 2
        bounds = [0,0,0,0];
        wave_window1 = 1;
        markers1 = [];
        wave_window2 = 1;
        markers2 = [];
        
        sweepBar = [];
        signalGroup = [] % handles of signalPannelHandles.m
        
        %handles for statistics
        meanresults=[];
        medianresults=[];
        SDresults=[];
        num_members_results=[];
        angleresults=[];
        
        %Limits for colorbar in RiseTime and CalciumDecay function
        RT_max=0;
        RT_min=0;
        T_min=0;
        T_max=0;
        APstart=0;
        APend=0;
        
        numOfContourLevels = 2;
        drawBrush = 0;
        drawSegmentation=0;
        brushSize=1;
        brushMaskIndices=[];
        isFillHoles = 0;
        isRemoveIslands = 0;
        removeIslandsPercent = 0.01;
        
        matlabVersion = 2018;
        
        signalPanel, signalSlider
    end
end
