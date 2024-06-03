function [cMap] = cMap(data,stat,endp,Fs,bg,rect, f, movie_scrn, handles)


%% my try

%% If you want to load previously filtered data, use the next lines.
data = uigetfile('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2\data.mat');
data = load(data);
data = struct2cell(data);
data= cell2mat(data);
disp('Loaded data')
toc
tic

% Get mask to get rid of background and activation time map
[data_avg, mask3] = avg_mask_cleaning_Sofia(Fs, data);

% Conduction Velocity Calculation
rect = round(rect);
rect = abs(rect);
[xx, yy] = meshgrid(rect(1):rect(1)+rect(3), rect(2):rect(2)+rect(4));
xx = reshape(xx,[],1);
yy = reshape(yy,[],1);
croppedAmap = data_avg(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3));
t = reshape(croppedAmap,[],1);

xyt = [xx yy t];
M = size(xyt,1);
prompt = "Resolution? ";
xres = input(prompt);
yres = xres;

for space = 30 % number of spatial neighbors
    for time_wind = 10 % number of temporal neighbors
        % Get all coefficients
        [XYT] = xyt_matrix(M,xyt,space,time_wind,xres,yres);
        disp('Got coefficients')
        toc
        tic

        for max_Vx = 200 % Define max velocity allowed
            % Calculate velocities
            [Vx, Vy, V] = vel_calc(XYT,max_Vx,max_Vx);
            disp('Got velocities')
            toc
            tic

            for ds_fac = 6 % Define downsampling factor
                % Downsample velocities for quiver plot
                [X, Y, U, V] = downsample(croppedAmap,xx,yy,Vx,Vy,ds_fac);
                disp('Downsampled data')
                toc
                tic

                for auto = 2 % Autoscale factor
                    % Create activation map
                    mask3(isnan(data_avg)) = 0;
                    figure
                    G = real2rgb(bg, 'gray');
                    imagesc(G)
                    hold on
                    imagesc(data_avg, 'AlphaData', mask3)
                    colormap(flipud(jet));
                    c = colorbar;

                    % Overlay quiver plot
                    hold on
                    q = quiver(X,Y,U,V, auto , 'k');
                    title('Activation Map')
                    c.Label.String = 'Activation Time (ms)';
                    axis off

                    disp(['Average velocity: ', num2str(nanmean(V)), ' cm/s'])

                end
            end
        end
    end
end




% now you have the polynomial for your surface. I noticed that most
% coefficients are zero or close to zero except for those multiplying x
% and y. Due to this, the derivative of the function wrt x is simply
% the coefficient, same with y. This feels like an oversimplification
% but rhythm people also do it so I am going to go with it.
% to find Vx, we need to do Tx / (Tx2 + Ty2) where Tx is the derivative
% wrt x, in this case the 7th coefficient and Ty is the derivative wrt
% y, in this case the 8th coefficient
% length_rect = size(croppedAmap,1);
% width_rect = size(croppedAmap,2);
% X_plot_rect = reshape(xx,[length_rect,width_rect]);
% Y_plot_rect = reshape(yy,[length_rect,width_rect]);
% Vx_plot_rect = reshape(Vx,[length_rect,width_rect]);
% Vy_plot_rect = reshape(Vy,[length_rect,width_rect]);
% downsample_factor = 8;
% %Create index vectors for rows and columns
% row_indices1 = 1:downsample_factor:length_rect;
% col_indices1 = 1:downsample_factor:width_rect;
% % get the downsampled matrices and convert them to vectors
% downsampled_Vx1 = Vx_plot_rect(row_indices1, col_indices1);
% Vx_downsampled_vector1 = reshape(downsampled_Vx1,[],1);
% downsampled_Vy1 = Vy_plot_rect(row_indices1, col_indices1);
% Vy_downsampled_vector1 = reshape(downsampled_Vy1,[],1);
% downsampled_X_plot = X_plot_rect(row_indices1, col_indices1);
% X_plot_downsampled_vector1 = reshape(downsampled_X_plot,[],1);
% downsampled_Y_plot = Y_plot_rect(row_indices1, col_indices1);
% Y_plot_downsampled_vector1 = reshape(downsampled_Y_plot,[],1);
% hold on
% %q = quiver(xx(1:3:end),yy(1:3:end),Vx(1:3:end),Vy(1:3:end),'k')
% q = quiver(X_plot_downsampled_vector1, Y_plot_downsampled_vector1, Vx_downsampled_vector1, Vy_downsampled_vector1, 3, 'k')
%

% make many figures at once


    function [X, Y, U, V] = downsample(croppedAmap,xx,yy,Vx,Vy,downsample_factor)

        length_rect = size(croppedAmap,1);
        width_rect = size(croppedAmap,2);
        X_plot_rect = reshape(xx,[length_rect,width_rect]);
        Y_plot_rect = reshape(yy,[length_rect,width_rect]);
        Vx_plot_rect = reshape(Vx,[length_rect,width_rect]);
        Vy_plot_rect = reshape(Vy,[length_rect,width_rect]);
        % downsample_factor = 8;
        %Create index vectors for rows and columns
        row_indices1 = 1:downsample_factor:length_rect;
        col_indices1 = 1:downsample_factor:width_rect;
        % get the downsampled matrices and convert them to vectors
        downsampled_Vx1 = Vx_plot_rect(row_indices1, col_indices1);
        U = reshape(downsampled_Vx1,[],1);
        downsampled_Vy1 = Vy_plot_rect(row_indices1, col_indices1);
        V = reshape(downsampled_Vy1,[],1);
        downsampled_X_plot = X_plot_rect(row_indices1, col_indices1);
        X = reshape(downsampled_X_plot,[],1);
        downsampled_Y_plot = Y_plot_rect(row_indices1, col_indices1);
        Y = reshape(downsampled_Y_plot,[],1);

    end

    function [XYT] = xyt_matrix(M,xyt,space_window_width,time_window_width,xres,yres)

        XYT = zeros(M,10);

        for num_pix = 1:M %go through each pixel

            % Find how far spatially and temporally each pixel is from the one you
            % are evaluating

            dx = abs(xyt(:,1)-xyt(num_pix,1));
            dy=abs(xyt(:,2)-xyt(num_pix,2));
            dt=abs(xyt(:,3)-xyt(num_pix,3));

            % Find the pixels near your pixel

            near = find ((sqrt(dx.^2 + dy.^2) <= space_window_width) & (isfinite(xyt(:,3))) & (dt <= time_window_width));

            %

            % xyt(near, 1:2) - get the x and y coord of each near pixel
            % ones(length(near),1) - make a vector of 1s the size of the number of
            % near pixels
            % xyt(i, 1:2) - get the x and y coords of your pixel
            % substract your pixel's location from all the near neighbors. make a
            % vector which says for each near pixel, how far (x and y) it is from
            % your pixel
            xyn = xyt(near, 1:2) - ones(length(near),1) * xyt(num_pix,1:2);

            % convert pixel distance to physical distance. xres and yres are user
            % inputs

            x = xyn(:,1) * xres;
            y = xyn(:,2) * yres;
            time = xyt(near,3);

            % create your A matrix. (look at powerpoint for more info)

            fit = [x.^2  y.^2  x.*y  x  y  ones(length(near),1)];

            % find coefficienta (a-f) such that Aa=t. these are your a-f

            coefs = fit\time;

            % find error. not sure why they use this specific error and what a
            % meaningful number would be. we want it to be small.
          
            resi    = sqrt(sum((time-fit*coefs).^2)/sum(time.^2));

            % store coords of pixel, time, coefs, and error

            XYT(num_pix,:) = [xyt(num_pix,:), coefs',resi];
        end
    end

    function [Vx, Vy, V] = vel_calc(XYT,max_Vx,max_Vy)

        Vx = (XYT(:,7)./(XYT(:,7).^2 + XYT(:,8).^2))*100; % times 100 to go from mm/msec to cm/sec
        Vy = (XYT(:,8)./(XYT(:,7).^2 + XYT(:,8).^2))*100;
        Vx(abs(Vx) > max_Vx) = NaN;
        Vy(abs(Vy) > max_Vy) = NaN;
        % standard_dev = std(Vx);
        % standard_dev_y = std(Vy);
        % Vx(abs(Vx) > standard_dev*2) = NaN;
        % Vy(abs(Vy) > standard_dev_y*2) = NaN;
        V = sqrt(Vx.^2+Vy.^2);
    end

% Can use this function to create the activation map for just one AP.
    function [actMap1, mask] = activationmap(stat, Fs, endp, data)

        stat=round(stat*Fs)+1;
        endp=round(endp*Fs)+1;
        %actMap = zeros(size(data,1),size(data,2));
        dataCroppedInTime = data(:,:,stat:endp); % truncate data

        % Re-normalize data in case of drift
        dataCroppedInTime = normalize_data(dataCroppedInTime);

        % identify channels that have been zero-ed out due to noise
        mask = max(dataCroppedInTime,[],3) > 0;

        % Find First Derivative and time of maxium
        derivatives = diff(dataCroppedInTime,1,3); % first derivative
        [~,max_i] = max(derivatives,[],3); % find location of max derivative

        % Create Activation Map
        actMap1 = max_i.*mask;
        actMap1(actMap1 == 0) = nan;
        offset1 = min(min(actMap1));
        actMap1 = actMap1 - offset1*ones(size(data,1),size(data,2));
        actMap1 = actMap1/Fs*1000; %% time in ms

    end



% %% Code
%
% %% Find Activation Times for Polynomial Surface
% stat=round(stat*Fs)+1;
% endp=round(endp*Fs)+1;
% actMap = zeros(size(data,1),size(data,2));
% dataCroppedInTime = data(:,:,stat:endp); % truncate data
%
% % Re-normalize data in case of drift
% dataCroppedInTime = normalize_data(dataCroppedInTime);
%
% % identify channels that have been zero-ed out due to noise
% mask = max(dataCroppedInTime,[],3) > 0;
%
% % Find First Derivative and time of maxium
% derivatives = diff(dataCroppedInTime,1,3); % first derivative
% [~,max_i] = max(derivatives,[],3); % find location of max derivative
%
% % Create Activation Map
% actMap1 = max_i.*mask;
% actMap1(actMap1 == 0) = nan;
% offset1 = min(min(actMap1));
% actMap1 = actMap1 - offset1*ones(size(data,1),size(data,2));
% actMap1 = actMap1/Fs*1000; %% time in ms
%
% %% Find Conduction Velocity Map - Bayly Method
% % Isolate ROI Specified by RECT
% rect = round(rect);
% rect = abs(rect)
% croppedAmap = actMap1(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)); % get the activation times for the pixels in your rectangle
% %exclude everything, but activation front
%
% use_window=1; % use windowed least-squares fitting
%
% if use_window
%     [xx yy]= meshgrid(rect(1):rect(1)+rect(3),rect(2):rect(2)+rect(4)); % make two matrices. both are the size of your rectangle (ROI)
%     % in xx each COLUMN is an x coord
%     % in yy each ROW is a y coord
%     xx = reshape(xx,[],1); % put it all in a vector. so you have x1 repeated however many y pixels you have,
%     % x2 repeated however many y pixels you have, etc.
%     yy = reshape(yy,[],1); % put it all in a vector. so you have y1, y2, y3...yn, repeated however many x pixels you have
%     % at this point you have two 1 D vectors. if you take the first two
%     % points, that is your first pixel. the second two points are your
%     % second pixel etc for all of you ROI
%     t = reshape(croppedAmap,[],1); % now we take the activation times and put them in a vector too.
%     %so now if you take the first point in all three, you have all the info
%     %for one pixel. etc.
%
%     xyt = [xx yy t]; % put them all together. each row is the info for one pixel in your rectangle
%
%     M=size(xyt,1); % this is the number of pixels in your ROI
%     XYT=zeros(M,17); % you create a new matrix with the number of rows equal to the pixels you have.
%     % there are 17 columns because you will find 17 coefficients
%
%     space_window_width =  5; %2./ handles.activeCamData.xres; % mm space frame.define neighbors in the spatial space. I am adding just a 5 pixel
%     time_window_width = 5 * handles.Fs / 1000; % time frame
%     how_many = 10; % why???
%     for i=1:M
%        % ������� ������� ���������� i-��� ����� �� ���� ���������
%        dx=abs(xyt(:,1)-xyt(i,1));
%        dy=abs(xyt(:,2)-xyt(i,2));
%        dt=abs(xyt(:,3)-xyt(i,3));
%        %find dx dy dt
%        %------------------------------------------------------------------------
% % find the location of the pixels that are near to yours. the location is
% % the row number
%        near=find((sqrt(dx.^2+dy.^2)<=space_window_width)&(isfinite(xyt(:,3)))&(dt<=time_window_width)); %I added time window which they were ignoring?
%        % make sure you have enough near points to continue with analysis
%        len=length(near);
%        %specify the points by using the points that are close enough
%        %------------------------------------------------------------------------
%
%        if len>how_many
%             xyn=xyt(near,1:2)-ones(len,1)*xyt(i,1:2); % centered around the i-th point. how far each near pixel is from your pixel
%             % here they are finding the location based on mm. this does not
%             % make sense to me, i will implement it in a pixel base
%             %x=xyn(:,1)*handles.activeCamData.xres/1000;%unit of X,Y are mM
%             %y=xyn(:,2)*handles.activeCamData.yres/1000;
%             x = xyn(:,1); %distance in pixels in the x dir
%             y = xyn(:,2); %distance in pixels in the y dir
%             t=xyt(near,3); % time of activation
%             %find dx dy dt of the specific points that are acceptable
%             %------------------------------------------------------------------------
%             fit     = [ones(len,1) x y x.^2 y.^2 x.*y x.^3 y.^3 x.*y.^2 y.*x.^2]; % on the windowed area
%             % we are using the equation T(x,y) = a + bx + cy + dx2 + ex2 +
%             % fxy + gx3 + hy3 + ixy2 + jyx2
%             coefs   = fit\t; % find a-j
%             resi    = sqrt(sum((t-fit*coefs).^2)/sum(t.^2)); % look at residuals, this will help evaluate fit
%             resilin = sqrt(sum((t-fit(:,1:3)*coefs(1:3)).^2)/sum(t.^2));
%             XYT(i,:)= [xyt(i,:),coefs',resi,len,cond(fit),resilin]; % per pixel you will have
%             %[x location, y location, time of act, a, b, c, d, e, f, g, h,
%             %i, j, residual, how many neighbors, cond(fit), resiling]
%        end
%     end
%
%     cropped = max(dataCroppedInTime(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),:), [], 3);
%     %cropped = dataCroppedInTime(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),1);
%     croppedResized = reshape(cropped, [], 1);
%
%     was_fitted=find( (XYT(:,1)~=0) & (XYT(:,2)~=0) & (croppedResized(:) > 0)); % if XYT was not filled or point not inside of segmented region
%     XYT=XYT(was_fitted,:);
%
%     % coef_x / (coef_x^2 + coef_y^2) * TODO multiplier????
%     Vx=(XYT(:,5)./(XYT(:,5).^2 + XYT(:,6).^2))*1000;%handles.activeCamData.xres;
%     % coef_y / (coef_y^2 + coef_y^2) * TODO multiplier????
%     Vy=-(XYT(:,6)./(XYT(:,5).^2 + XYT(:,6).^2))*1000;%handles.activeCamData.yres;
%
%     V=sqrt(Vx.^2+Vy.^2);
% end
%
% %Calculate Conduction Velocity
% if ~use_window
%     Vx = Tx./(Tx.^2+Ty.^2);
%     Vy = -Ty./(Tx.^2+Ty.^2);
%     V = sqrt(Vx.^2 + Vy.^2);
% else
%     %bad=(V>5); %includeMask CV above 2 m/s
%     %Vx(bad)=NaN;
%     %Vy(bad)=NaN;
%     %V(bad)=NaN;
% end
% V = normalize(V,'range');
% % Display the regional statistics
% disp('Regional conduction velocity statistics:')
% meanV=nanmean(V(isfinite(V)));
% disp(['The mean value is ' num2str(meanV) ' m/s.'])
% medV = median(V(isfinite(V)));
% disp(['The median value is ' num2str(medV) ' m/s.'])
% stdV = std2(V(isfinite(V)));
% disp(['The standard deviation is ' num2str(stdV) '.'])
% meanAng = mean(atan2(Vy(isfinite(Vy)),Vx(isfinite(Vy))).*180/pi);
% disp(['The mean angle is ' num2str(meanAng) ' degrees.'])
% medAng = median(atan2(Vy(isfinite(Vy)),Vx(isfinite(Vy))).*180/pi);
% disp(['The median angle is ' num2str(medAng) ' degrees.'])
% stdAng = std2(atan2(Vy(isfinite(Vy)),Vx(isfinite(Vy))).*180/pi);
% disp(['The standard deviation of the angle is ' num2str(stdAng) '.'])
% num_vectors = numel(V(isfinite(V)));
% disp(['The number of vectors is ' num2str(num_vectors) '.'])
%
%         % statistics window
%        handles.activeCamData.meanresults = sprintf('Mean: %0.3f',meanV);
%        handles.activeCamData.medianresults  = sprintf('Median: %0.3f',medV);
%        handles.activeCamData.SDresults = sprintf('S.D.: %0.3f',stdV);
%        handles.activeCamData.num_membersresults = sprintf('#Members: %d',num_vectors);
%        handles.activeCamData.angleresults = sprintf('Angle: %d',meanAng);
%
%        set(handles.meanresults,'String',handles.activeCamData.meanresults);
%        set(handles.medianresults,'String',handles.activeCamData.medianresults);
%        set(handles.SDresults,'String',handles.activeCamData.SDresults);
%        set(handles.num_members_results,'String',handles.activeCamData.num_membersresults);
%        set(handles.angleresults,'String',handles.activeCamData.angleresults);
%
% handles.activeCamData.saveData = actMap1;
% % Plot Results
% cla(movie_scrn);
% %compute isolines
%
% [C,h] = contourf(movie_scrn, actMap1,(endp-stat)/2, 'LineColor','k');
%
%
% %caxis(movie_scrn,[stat endp]);
% contourcmap('copper','SourceObject', movie_scrn, 'ColorAlignment', 'center');
% %colorbar(movie_scrn);
% set(movie_scrn,'YTick',[],'XTick',[]);
%
%
% hold (movie_scrn,'on')
%
% %Y_plot = size(data,1)+1 - y(isfinite(Z_fit));
% if ~use_window
%     Y_plot = y(isfinite(Z_fit));
%     X_plot = x(isfinite(Z_fit));
%     Vx_plot = Vx(isfinite(Z_fit));
% else
%     Y_plot = yy(was_fitted);
%     X_plot = xx(was_fitted);
%     Vx_plot = Vx;
% end
% Vx_plot(abs(Vx_plot) > 5) = 5.*sign(Vx_plot(abs(Vx_plot) > 5));
% if ~use_window
%     Vy_plot = Vy(isfinite(Z_fit));
% else
%     Vy_plot = Vy;%(was_fitted,1);
% end
% Vy_plot(abs(Vy_plot) > 5) = 5.*sign(Vy_plot(abs(Vy_plot) > 5));
% V = sqrt(Vx_plot.^2 + Vy_plot.^2);
% 
% %Create Vector Array to pass to following functions
% VecArray = [X_plot Y_plot Vx_plot Vy_plot V];
% handles.activeCamData.VecArray = VecArray;
% length_rect = size(croppedAmap,1)
% width_rect = size(croppedAmap,2)
% X_plot_rect = reshape(X_plot,[length_rect,width_rect]);
% Y_plot_rect = reshape(Y_plot,[length_rect,width_rect]);
% Vx_plot_rect = reshape(Vx_plot,[length_rect,width_rect]);
% Vy_plot_rect = reshape(Vy_plot,[length_rect,width_rect]);
% downsample_factor = 6;
% %Create index vectors for rows and columns
% row_indices1 = 1:downsample_factor:length_rect;
% col_indices1 = 1:downsample_factor:width_rect;
% % get the downsampled matrices and convert them to vectors
% downsampled_Vx1 = Vx_plot_rect(row_indices1, col_indices1);
% Vx_downsampled_vector1 = reshape(downsampled_Vx1,[],1);
% downsampled_Vy1 = Vy_plot_rect(row_indices1, col_indices1);
% Vy_downsampled_vector1 = reshape(downsampled_Vy1,[],1);
% downsampled_X_plot = X_plot_rect(row_indices1, col_indices1);
% X_plot_downsampled_vector1 = reshape(downsampled_X_plot,[],1);
% downsampled_Y_plot = Y_plot_rect(row_indices1, col_indices1);
% Y_plot_downsampled_vector1 = reshape(downsampled_Y_plot,[],1);
% Vx_downsampled_vector1(abs(Vx_downsampled_vector1)>2) = NaN;
% Vy_downsampled_vector1(abs(Vy_downsampled_vector1)>2) = NaN;
% 
% % X_plot1 = X_plot(1:1:length(X_plot));
% % Y_plot1 = Y_plot(1:1:length(Y_plot));
% % Vx_plot1 = Vx_plot(1:1:length(Vx_plot));
% % Vy_plot1 = Vy_plot(1:1:length(Vy_plot));
% % Vx_plot1(abs(Vx_plot1)>2) = NaN;
% % Vy_plot1(abs(Vy_plot1)>2) = NaN;
% 
% 
% %data for saving
% handles.activeCamData.saveX_plot = X_plot_downsampled_vector1;
% handles.activeCamData.saveY_plot = Y_plot_downsampled_vector1;
% handles.activeCamData.saveVx_plot = Vx_downsampled_vector1;
% handles.activeCamData.saveVy_plot = Vy_downsampled_vector1;
% 
% % plot vector field
% quiver_step = 1;
% q = quiver(movie_scrn, X_plot_downsampled_vector1(1:quiver_step:end),...
%            Y_plot_downsampled_vector1(1:quiver_step:end),Vx_downsampled_vector1(1:quiver_step:end),...
%            -1.0 * Vy_downsampled_vector1(1:quiver_step:end),'k');
% q.LineWidth = 1;
% q.AutoScaleFactor = 2;
% set(movie_scrn,'YDir','reverse');
% 
% hold (movie_scrn,'off');
% 
% %first figure
% %Plot Map
% cc = figure('Name','Activation Map with Velocity Vectors');
% %Create Mask
% actMap_Mask = zeros(size(bg));
% % Your mask needs to be called mask3 (can probably change this). Make sure
% % you are loading the right heart!!
% %load('C:\Users\Sofia\Desktop\Optical Data - Can studies\OM_MATLAB_C23-001\mask3.txt')
% %load('C:\Users\Sofia\Desktop\Optical Data - Can studies\OM_MATLAB_C23-002\mask3.txt')
% %load('C:\Users\Sofia\Desktop\Optical Data - Can studies\OM_MATLAB_C23-003\mask3.txt')
% load('C:\Users\Sofia\Desktop\Optical Data - Can studies\OM_MATLAB_C23-004\mask3.txt')
% actMap_Mask(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)) = 1;
% %Build the Image
% G = real2rgb(bg, 'gray');
% 
% %% activation map just for rectangle
% croppedAmap(croppedAmap == 0) = nan;
% offset1 = min(min(croppedAmap));
% croppedAmap = croppedAmap - offset1*ones(size(croppedAmap,1),size(croppedAmap,2));
% actMap1 = croppedAmap/Fs*1000; %% time in ms
% new_actMap1 = zeros(256,256);
% new_actMap1(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)) = actMap1;
% 
% 
% J = real2rgb(-new_actMap1, 'jet');%,[min(min(temp)) max(max(temp))]);
% final_mask = actMap_Mask .* mask3;
% A = real2rgb(final_mask, 'gray');
% %A = real2rgb(mask3, 'gray');
% I = J .* A + G .* (1-A);
% image(I)
% hold on
% 
% c = colorbar;
% clim([min(actMap1(:)), max(actMap1(:))]);
% colormap(jet);
% c.Label.String = 'Activation Time (ms)';
% 
% q = quiver(X_plot_downsampled_vector1(1:quiver_step:end),...
%            Y_plot_downsampled_vector1(1:quiver_step:end),Vx_downsampled_vector1(1:quiver_step:end),...
%            -1.0 * Vy_downsampled_vector1(1:quiver_step:end),'k');
% q.LineWidth = 1;
% q.LineWidth = 1;
% q.AutoScaleFactor = 2;
% 
% % second figure 
% cv = figure('Name','Conduction Velocity Map');
% %Create Mask
% actMap_Mask = zeros(size(bg)); %zeros matrix the size of your og image
% actMap_Mask(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)) = 1; % make it 1 in your selected region
% cvMap_Mask = zeros(size(bg)); % zeros matrix the size of your og image
% %V(V > 2.0) = NaN;
% V=reshape(V,size(cropped,1),size(cropped,2));
% cvMap_Mask(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)) = V; % plug in the velocity matrix in the selected coordinates
% %cvMap_Mask(mask3) = V;
% %Build the Image
% %cvMap_Mask(cvMap_Mask > 0.7) = NaN;
% G = real2rgb(bg, 'gray'); % background image
% J = real2rgb(cvMap_Mask, flipud(jet),[min(min(V)) max(max(V))]);
% final_mask = actMap_Mask .* mask3;
% A = real2rgb(final_mask, 'gray');
% %A = real2rgb(actMap_Mask, 'gray');
% I = J .* A + G .* (1-A);
% %subplot(121)
% 
% 
% 
% 
% image(I)
% c = colorbar;
% colormap(flipud(jet));
% new_mask = mask3(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3));
% new_vel = new_mask .* V;
% new_vel = new_mask .* V;
% new_vel(new_vel == 0) = NaN;
% new_vel(new_vel > 1.0) = NaN;
% clim([min(min(new_vel)), max(max(new_vel))]);
% c.Label.String = 'Conduction Velocity Magnitude (m/s)';
% axis off
% axis image
% 
% figure %last figure
% %subplot(122)
% % outside the mask, it should not have a value
% new_mask = mask3(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3));
% new_vel = new_mask .* V;
% new_vel(new_vel < 0.05) = NaN;
% new_vel(new_vel > 3.0) = NaN;
% 
% 
% % Set NaN color to white
% nan_color = [1, 1, 1]; % RGB values for white
% 
% % Create a colormap with NaN color
% custom_colormap = colormap(flipud(jet)); % or any other colormap you prefer
% custom_colormap(1, :) = nan_color; % Set the first row of the colormap to NaN color
% colormap(custom_colormap);
% 
% % Plot the image with NaN values
% imagesc(new_vel);
% 
% c = colorbar;
% c.Label.String = 'Conduction Velocity Magnitude (m/s)';
% 
% axis image
% axis off
% title('Conduction Velocity Magnitude')
% 
% b=6;
% % 
% % title('Activation Map with Velocity Vectors')
% axis image
% axis off

% rect_plot = [rect(1) (size(data,1) + 1 - rect(2)-rect(4)) rect(3) rect(4)];
% rectangle(movie_scrn, 'Position',rect_plot,'EdgeColor','c')
%axis (movie_scrn,'off')
end


% function [cMap] = cMap(data,stat,endp,Fs,bg,rect, f, movie_scrn, handles)
% %% cMap is the central function for creating conduction velocity maps
% % [cMap] = cMap(data,stat,endp,Fs,bg,rect) calculates the conduction
% % velocity map for a single action potential upstroke by fitting a
% % polynomial and calculating the surface derivative to a pre-defined area
% % of cmosData.  This area is specified by the vector rect.
% 
% % INPUTS
% % data = cmos data (voltage, calcium, etc.) from the micam ultima system.
% % 
% % stat = start of analysis (in msec)  %in ms????
% %
% % endp = end of analysis (in msec)
% %
% % Fs = sampling frequency
% %
% % bg = black and white background image from the CMOS camera.  This is a
% % 100X100 pixel image from the micam ultima system. bg is stored in the
% % handles structure handles.bg.
% %
% % rect = area of interest specified by getrect in Rhythm.m GUI
% 
% % OUTPUT
% % cMap = conduction velocity map
% 
% % METHOD
% % The method used for calculating conduction velocity is fully described by
% % Bayly et al in "Estimation of Conduction Velocity Vecotr Fields from
% % Epicardial Mapping Data".  Briefly, this function calculates the
% % conduction velocity for a region of interest (ROI) for a single optical
% % action potential.  First, an activation map is calculated for the ROI
% % by identifying the time of maximum derivative of each ROI pixel.  Next, a
% % third-order polynomial surface is fit to the activation map and the
% % surface derivative of the fitted surface is calculated.  Finally, the x
% % and y components of conduction velocity are calculated per pixel
% % (pixel/msec).
% 
% 
% % REFERENCES
% % Bayly PV, KenKnight BH, Rogers JM, Hillsley RE, Ideker RE, Smith WM.
% % "Estimation of COnduction Velocity Vecotr Fields from Epicardial Mapping
% % Data". IEEE Trans. Bio. Eng. Vol 45. No 5. 1998.
% 
% % ADDITIONAL NOTES
% % The conduction velocity vectors are highly dependent on the goodness of
% % fit of the polynomial surface.  In the Balyly paper, a 2nd order polynomial 
% % surface is used.  We found this polynomial to be insufficient and thus increased
% % the order to 3.  MATLAB's intrinsic fitting functions might do a better
% % job fitting the data and should be more closely examined if velocity
% % vectors look incorrect.
% 
% % RELEASE VERSION 1.0.1
% 
% % AUTHOR: Jacob Laughner (jacoblaughner@gmail.com)
% 
% % Email optocardiography@gmail.com for any questions or concerns.
% % Refer to efimovlab.org for more information.
% % 
% %% Code
% %% Find Activation Times for Polynomial Surface
% %data = imresize(data,0.5);
% stat=round(stat*Fs)+1; %stat is the time in seconds. this line multiplies it by the frequency and adds 1. Gets the frame where you start
% endp=round(endp*Fs)+1; %find time point where you end
% actMap = zeros(size(data,1),size(data,2)); % make a zero map the size of your image
% dataCroppedInTime = data(:,:,stat:endp); % truncate data get only the data for the time you want
% 
% % Re-normalize data in case of drift
% dataCroppedInTime = normalize_data(dataCroppedInTime); %renormalize now that you took other data away
% 
% % identify channels that have been zero-ed out due to noise
% mask = max(dataCroppedInTime,[],3) > 0; %create a mask with pixels that have signal
% 
% 
% % Find First Derivative and time of maxium
% derivatives = diff(dataCroppedInTime,1,3); % first derivative between each time point in each pixel
% [~,max_i] = max(derivatives,[],3); % find location of max derivative. which time point
% 
% % Create Activation Map
% actMap1 = max_i.*mask; %only consider the pixels that aren't zero
% actMap1(actMap1 == 0) = nan; %if it is zero just put nan
% offset1 = min(min(actMap1)); %find the time where the first pixel activates (place of max der)
% actMap1 = actMap1 - offset1*ones(size(data,1),size(data,2)); % change so the first point activates at 0
% actMap1 = actMap1/Fs*1000; %% time in ms. doesn't change anything if Fs is 1000
% 
% %% Find Conduction Velocity - Following Bayly Paper
% 
% % choose the active neighbors for a pixel
% 
% % chose delta x - variance due to propagation should be much larger than
% % that due to noise. 4-5 times the sampling interval. sampling interval is
% % 0,1749. times 5 = 0.8745 mm. for now - delta x = 1 mm = 6 samples
% 
% %choose delta y - variance due to propagation should be much larger than
% %that due to noise. 4-5 times the sampling interval. delta y = 1
% 
% % choose delta t - variance due to propagation should be much larger than
% % that due to noise. 4-5 times the sampling interval in each dimension.
% % sampling rate is 1000 hz (samples per second). delta t = 5 samples (0.005
% % sec = 5 msec)
% 
% % First, we isolate ROI specified by rect
% rect = round(rect); % round to the nearest pixel
% croppedAmap = actMap1(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)); % get the activation map of your selection
% 
% % Make a matrix where each row is a pixel
% [xx yy]= meshgrid(rect(1):rect(1)+rect(3),rect(2):rect(2)+rect(4));% make two matrices the size of your rect. xx takes the first x-cordinate and fills the first column. Then the second x-coordinate fills the second column, etc. yy takes the first y-cord and fills the first row. the second row is the second y-cord,etc.
% xx = reshape(xx,[],1); % make it one vector where you have the first x cord however many y pixels you chose. then the second x-cord the same number of times. etc etc
% yy = reshape(yy,[],1); % same but with y cords. but this time you get the first y cord, then the second then the third, etc repeated however many x pixels you chose.
% t = reshape(croppedAmap,[],1); % put all your activation times into a vector
% 
% xyt = [xx yy t]; % now you have [x cord, y cord, act time] for each pixel
% 
% M=size(xyt,1); % find the size of xyt, aka how many pixels you have
% 
% % Now go through each pixel and find their nearest neighbors
% space_window_width = 6;
% time_window_width = 5;
% for i=1:M % go pixel by pixel
% 
%     % Find how far away your current pixel is from all others in x, y
%     % and t
%     dx=abs(xyt(:,1)-xyt(i,1)); %all the x variables - the pixel you are in.tells you how far your pixel is in the x cord from all the others
%     dy=abs(xyt(:,2)-xyt(i,2));% same but with y cords
%     dt=abs(xyt(:,3)-xyt(i,3)); % same but in time
%     near=find((sqrt(dx.^2+dy.^2)<=space_window_width)&(dt<=time_window_width)); %pixels that are within the window are "near"
% 
%     xyn=xyt(near,1:2)-xyt(i,1:2); % centered around the i-th point, how far away each pixel is from your pixel
%     x=xyn(:,1); % x coords of near neighbors
%     y=xyn(:,2); % y coords of near neighbors
%     t=xyt(near,3);% find the times of activation of all the near pixels
%     fit = [ones(length(xyn),1) y x x.*y y.^2 x.^2]; 
%     coefs   = fit\t;
% 
% end
% 
% 
% 
% % fit them using a least-squares algorithm to a smooth polynomial surface
% 
% %% Find Conduction Velocity Map - Bayly Method
% % Isolate ROI Specified by RECT
% rect = round(rect);
% croppedAmap = actMap1(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)); % get the map of only your selection
% %exclude everything, but activation front
% 
% use_window=0; % use windowed least-squares fitting
% 
% %%
% % if ~use_window
%     % includeMask=zeros(rect(4)+1,rect(3)+1); %make a mask of zeroes?
%     % includeMask(2:end-1,2:end-1)=abs(croppedAmap(1:end-2,2:end-1)-croppedAmap(3:end,2:end-1))+abs(croppedAmap(2:end-1,1:end-2)-croppedAmap(2:end-1,3:end));
%     % includeMask(2:end-1,2:end-1)=((croppedAmap(2:end-1,2:end-1)<croppedAmap(3:end,2:end-1))|(croppedAmap(2:end-1,2:end-1))<croppedAmap(1:end-2,2:end-1)|(croppedAmap(2:end-1,2:end-1)<croppedAmap(2:end-1,3:end))|(croppedAmap(2:end-1,2:end-1))<croppedAmap(2:end-1,1:end-2));
%     % croppedAmap(includeMask==0)=NaN;
% % end
% if use_window
%     [xx yy]= meshgrid(rect(1):rect(1)+rect(3),rect(2):rect(2)+rect(4));% make two matrices the size of your rect. xx takes the first x-cordinate and fills the first column. Then the second x-coordinate fills the second column, etc. yy takes the first y-cord and fills the first row. the second row is the second y-cord,etc.
%     xx = reshape(xx,[],1); % make it one vector where you have the first x cord however many y pixels you chose. then the second x-cord the same number of times. etc etc
%     yy = reshape(yy,[],1); % same but with y cords. but this time you get the first y cord, then the second then the third, etc repeated however many x pixels you chose.
%     t = reshape(croppedAmap,[],1); % put all your activation times into a vector
% 
%     xyt = [xx yy t]; % now you have [x cord, y cord, act time] for each pixel
% 
%     M=size(xyt,1); % find the size of xyt, aka how many pixels you have
%     % matrix with as many rows as pixels. it has 17 columns because it will
%     % store 17 parameters per pixel: x cord, y cord, t cord, 10
%     % coefficients(?),  the root mean square error, number of points
%     % included in the fit, the condition number of the fit matrix (?), and
%     % the linear part root mean square error.
%     XYT=zeros(M,17); %as many rows you have pixels and 17 columns???
% 
%     space_window_width =  3; %2./ handles.activeCamData.xres; % mm space frame. might need to change, why??? how far in space will you look for neighbors
%     time_window_width = 5 * handles.Fs / 1000; % time frame % it's just 5 but again why. how far in time will you look for neighbors
%     how_many = 10; % why??? they had 12, i am changing it to 6 just for analysis
%     for i=1:M % go pixel by pixel
% 
%        % Find how far away your current pixel is from all others in x, y
%        % and t
%        dx=abs(xyt(:,1)-xyt(i,1)); %all the x variables - the pixel you are in.tells you how far your pixel is in the x cord from all the others
%        dy=abs(xyt(:,2)-xyt(i,2));% same but with y cords
%        dt=abs(xyt(:,3)-xyt(i,3)); % same but in time
%        %find dx dy dt 
%        %------------------------------------------------------------------------
%         % Define a pixel as being "close enough" if:
%         % first calculate the spatial distance from your pixel to every
%         % other pixel. if this distance is less than your specified
%         % distance AND the time for that pixel is finite (?) not NAN, then
%         % that pixel is considered near your pixel. you get a vector with
%         % locations of those pixels.
%        near=find((sqrt(dx.^2+dy.^2)<=space_window_width)&(isfinite(xyt(:,3))));
%        % find how many pixels are near your pixel.
%        len=length(near);
%        %specify the points by using the points that are close enough
%        %------------------------------------------------------------------------
% 
%        if len>how_many % this seems really arbitrary, look into the paper what is the significance of this parameter
%            % get the x and y cord of each "near" pixel. subtract that from your current pixel xy cords.  
%             xyn=xyt(near,1:2)-ones(len,1)*xyt(i,1:2); % centered around the i-th point, how far away each pixel is from your pixel
% 
%             %see how far physically each pixel is from your pixel. check
%             %these next lines, don't make sense
%             x=xyn(:,1)*handles.activeCamData.xres/1000;%unit of X,Y are mM. why divide by 1000? resolution is already in mm
%             y=xyn(:,2)*handles.activeCamData.yres/1000;
%             % find the times of activation of all the near pixels
%             t=xyt(near,3);
%             %find dx dy dt of the specific points that are acceptable
%             %------------------------------------------------------------------------
%             % creates a matrix where the first column is 1, then x, then y,
%             % etc.
%             fit     = [ones(len,1) x y x.^2 y.^2 x.*y x.^3 y.^3 x.*y.^2 y.*x.^2]; % on the windowed area
%             % performs polynomial fit using the least squares method to
%             % find the coefficients that minize the squared differences
%             % between the fitted curve and the observed time values
%             coefs   = fit\t;
%             % calculates root mean square error between the observed time
%             % values and the values predicted by the fitted polynomial
%             resi    = sqrt(sum((t-fit*coefs).^2)/sum(t.^2));
%             % calculates root mean square error between the observed time
%             % values and the values predicted by the linear part of the
%             % fitted polynomial.
%             resilin = sqrt(sum((t-fit(:,1:3)*coefs(1:3)).^2)/sum(t.^2));
%             % store these values in XYT per pixel: x cord, y cord, t cord, 10
%             % coefficients(?),  the root mean square error, number of points
%             % included in the fit, the condition number of the fit matrix (?), and
%             % the linear part root mean square error.
%             XYT(i,:)= [xyt(i,:),coefs',resi,len,cond(fit),resilin];
%         end
%     end
%     % find the maximum value in the time plane for your selected data.
%     % creates a matrix with however many times you hit the max
%     cropped = max(dataCroppedInTime(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),:), [], 3);
%     %cropped = dataCroppedInTime(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),1);
%     % make it a column of all the times you got the max time
%     croppedResized = reshape(cropped,[],1);
% 
%     % double check what they mean with this next line
%     was_fitted=find( (XYT(:,1)~=0) & (XYT(:,2)~=0) & ( croppedResized(:) > 0) ); % if XYT was not filled or point not inside of segmented region
%     XYT=XYT(was_fitted,:);
% 
%     % coef_x / (coef_x^2 + coef_y^2) * TODO multiplier????
%     Vx=(XYT(:,5)./(XYT(:,5).^2 + XYT(:,6).^2))*1000;%handles.activeCamData.xres; 
%     % coef_y / (coef_y^2 + coef_y^2) * TODO multiplier????
%     Vy=-(XYT(:,6)./(XYT(:,5).^2 + XYT(:,6).^2))*1000;%handles.activeCamData.yres; 
% 
%     V=sqrt(Vx.^2+Vy.^2);
% end         
% 
% 
% %%
% % Fit Activation Map with 3rd-order Polynomial
% if ~use_window %
%     cind = isfinite(croppedAmap); % find the pixels that have data in the selected rectangle
%     %cind = ones(size(croppedAmap,1), size(croppedAmap,2));
%     [x_full, y_full]= meshgrid(rect(1):rect(1)+rect(3),rect(2):rect(2)+rect(4)); % make two matrices the size of
%     % the selected rectangle. each column of x is an x coord. each row of y
%     % is a y coord.
%     x = reshape(x_full,[],1); % put everything in a vector. you have the first x coordinate repeated, then the second, etc
%     y = reshape(y_full,[],1); % put everyting in a vector. you have the first y coord, the second, the third, etc, and then repeat that
%     z = reshape(croppedAmap,[],1); % activation times (msecs) in a vector.
%     a = [x.^3 y.^3 x.*y.^2 y.*x.^2 x.^2 y.^2 x.*y x y ones(size(x,1),1)]; % for the whole rectangle. First column is all the
%     % x coords cubed, the second row the y coords cubed, etc. gives you a
%     % rectangle with the number of pixels x 10 (you are calculating 10
%     % values)
%     X = x(cind); % find which x coords have data. make a vector going pixel by pixel
%     Y = y(cind); % same with y coords
%     Z = z(cind); % same with time (msecs)
%     A = [X.^3 Y.^3 X.*Y.^2 Y.*X.^2 X.^2 Y.^2 X.*Y X Y ones(size(X,1),1)]; % for the active front. get same values as before but just for
%     % pixels with data
%     solution = A\Z; % solution is the set of coefficients
%     Z_fit = a*solution; % Z_fit is a polynome surface on the whole rectangle. compute surface for that point.
%     Z_fit = reshape(Z_fit,size(cind)); % reshape Z_fit to be rectangle shaped
% end
% %Z_fit=nan(size(cind));
%  %Z_fit(cind)=A*a;
% % zres=reshape(Z_fit,[],1)-Z;
% % SSres=sum(zres.^2);
% % SStot=(length(Z)-1)*var(Z);
% % rsq=1-SSres/SStot;
% % disp(['rsq of fit is ' num2str(rsq)]);
% % Find Gradient of Polynomial Surface
% if ~use_window %
%     [Tx, Ty] = gradient(Z_fit);
%     Tx=Tx/handles.activeCamData.xres;
%     Ty=Ty/handles.activeCamData.yres;
% 
%     % Calculate Conduction Velocity
%     Vx = -Tx./(Tx.^2+Ty.^2);
%     Vy = -Ty./(Tx.^2+Ty.^2);
%     % Vx(Vx > 0.9) = NaN;
%     % Vy(abs(Vy) > 0.7) = NaN;
%     V = sqrt(Vx.^2 + Vy.^2);
%     %V(V > 50.0) = NaN;
% end
% meanV = nanmean(V,"all")
% stdV = std(V(:),"omitmissing")
% meanAng = mean2(atand(Vy./Vx))
% stdAng = std2(atand(Vy./Vx))
% %first figure
% %Plot Map
% cc = figure('Name','Activation Map with Velocity Vectors');
% %Create Mask
% actMap_Mask = zeros(size(bg));
% % Your mask needs to be called mask3 (can probably change this). Make sure
% % you are loading the right heart
% load('C:\Users\Sofia\Desktop\Optical Data - Can studies\OM_MATLAB_C23-004\mask3.txt')
% actMap_Mask(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)) = 1;
% %Build the Image
% G = real2rgb(bg, 'gray');
% J = real2rgb(actMap1, 'jet');%,[min(min(temp)) max(max(temp))]);
% A = real2rgb(mask3, 'gray');
% I = J .* A + G .* (1-A);
% image(I)
% hold on
% 
% c = colorbar;
% %clim([min(actMap1(:)), max(actMap1(:))]);
% %colormap(flipud(jet));
% c.Label.String = 'Activation Time (ms)';
% 
% %Overlay Conduction Velocity Vectors
% 
% % I need to downsample the velocity MATRIX. also downsample in the SAME WAY
% % the cind matrix. reshape both as a vector and multiply them.
% 
% %the velocity at this point is already only on the selected rectangle. need
% %to go higher
% 
% 
% % I don't want to change the other code so I will re-calculate conduction
% % velocity using the whole mask.
% 
% %% conduction velocity for the quiver plot %%
% figure
% open('C:\Users\Sofia\Desktop\Optical Data - Can studies\OM_MATLAB_C23-004\Figures-4-11\AM_S1_File#5_bin9.fig')
% hold on
% cind1 = isfinite(actMap1); % find the pixels that have data in the whole masked image
% [x_full1, y_full1]= meshgrid(1:length(cind1), 1:width(cind1)); % make two matrices the size of
% % the image. each column of x is an x coord. each row of y
% % is a y coord.
% x1 = reshape(x_full1,[],1); % put everything in a vector. you have the first x coordinate repeated, then the second, etc
% y1 = reshape(y_full1,[],1); % put everyting in a vector. you have the first y coord, the second, the third, etc, and then repeat that
% z1 = reshape(actMap1,[],1); % activation times in a vector.
% a1 = [x1.^3 y1.^3 x1.*y1.^2 y1.*x1.^2 x1.^2 y1.^2 x1.*y1 x1 y1 ones(size(x1,1),1)]; % for the whole rectangle. First column is all the
% % x coords cubed, the second row the y coords cubed, etc. gives you a
% % rectangle with the number of pixels x 10 (you are calculating 10
% % values)
% X1 = x1(cind1); % find which x coords have data. make a vector going pixel by pixel
% Y1 = y1(cind1); % same with y coords
% Z1 = z1(cind1); % same with time
% A1 = [X1.^3 Y1.^3 X1.*Y1.^2 Y1.*X1.^2 X1.^2 Y1.^2 X1.*Y1 X1 Y1 ones(size(X1,1),1)]; % for the active front. get same values as before but just for
% % pixels with data
% solution1 = A1\Z1; % solution is the set of coefficients
% Z_fit1 = a1*solution1; % Z_fit is a polynome surface on the whole rectangle
% Z_fit1 = reshape(Z_fit1,size(cind1)); % reshape Z_fit to be rectangle shaped
% 
% [Tx1, Ty1] = gradient(Z_fit1);
% Tx1=Tx1/handles.activeCamData.xres;
% Ty1=Ty1/handles.activeCamData.yres;
% 
% Vx1 = -Tx1./(Tx1.^2+Ty1.^2);
% Vy1 = -Ty1./(Tx1.^2+Ty1.^2);
% Vx1(abs(Vx1) > 4) = NaN;
% Vy1(abs(Vy1) > 4) = NaN;
% V1 = sqrt(Vx1.^2 + Vy1.^2);
% V1(V1 > 2.0) = NaN;
% 
% % downsample velocity matrices
% downsample_factor = 8; % Adjust this factor as needed
% [rows1, cols1] = size(Vx1);
% % Create index vectors for rows and columns
% row_indices1 = 1:downsample_factor:rows1;
% col_indices1 = 1:downsample_factor:cols1;
% % get the downsampled matrices and convert them to vectors
% downsampled_Vx1 = Vx1(row_indices1, col_indices1);
% Vx_downsampled_vector1 = reshape(downsampled_Vx1,[],1);
% downsampled_Vy1 = Vy1(row_indices1, col_indices1);
% Vy_downsampled_vector1 = reshape(downsampled_Vy1,[],1);
% downsampled_cind1 = cind1(row_indices1, col_indices1);
% cind_downsampled_vector1 = reshape(downsampled_cind1,[],1);
% %multiply to get velocity on mask only
% new_Vx1 = cind_downsampled_vector1 .* Vx_downsampled_vector1;
% new_Vy1 = cind_downsampled_vector1 .* Vy_downsampled_vector1;
% 
% % downsample x coords
% downsampled_xmatrix1 = x_full1(row_indices1, col_indices1);
% x_downsampled1 = reshape(downsampled_xmatrix1,[],1);
% 
% % downsample y coords
% downsampled_ymatrix1 = y_full1(row_indices1, col_indices1);
% y_downsampled1 = reshape(downsampled_ymatrix1,[],1);
% 
% quiver_plot_full = quiver(x_downsampled1, y_downsampled1, new_Vx1, new_Vy1, 2, 'k');
% 
% title('Activation Map with Velocity Vectors')
% axis image
% axis off
% 
% figure
% image(I)
% hold on

% 
% c = colorbar;
% clim([min(actMap1(:)), max(actMap1(:))]);
% %colormap(flipud(jet));
% c.Label.String = 'Activation Time (ms)';
% % Plot quiver plot with adjusted size
% x_downsampled1(x_downsampled1 < rect(1) | x_downsampled1 > rect(1) + rect(3)) = NaN;
% y_downsampled1(y_downsampled1 < rect(2) | y_downsampled1 > rect(2) + rect(4)) = NaN;
% quiver_plot = quiver(x_downsampled1, y_downsampled1, new_Vx1, new_Vy1, 2, 'k');
% 
% title('Activation Map with Velocity Vectors')
% axis image
% axis off
% 
% % second figure 
% cv = figure('Name','Conduction Velocity Map');
% %Create Mask
% actMap_Mask = zeros(size(bg)); %zeros matrix the size of your og image
% actMap_Mask(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)) = 1; % make it 1 in your selected region
% cvMap_Mask = zeros(size(bg)); % zeros matrix the size of your og image
% %V(V > 2.0) = NaN;
% cvMap_Mask(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)) = V; % plug in the velocity matrix in the selected coordinates
% %cvMap_Mask(mask3) = V;
% %Build the Image
% %cvMap_Mask(cvMap_Mask > 0.7) = NaN;
% G = real2rgb(bg, 'gray'); % background image
% J = real2rgb(cvMap_Mask, flipud(jet),[min(min(V)) max(max(V))]);
% final_mask = actMap_Mask .* mask3;
% A = real2rgb(final_mask, 'gray');
% %A = real2rgb(actMap_Mask, 'gray');
% I = J .* A + G .* (1-A);
% %subplot(121)
% 
% 
% 
% 
% image(I)
% c = colorbar;
% colormap(flipud(jet));
% new_mask = mask3(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3));
% new_vel = new_mask .* V;
% new_vel = new_mask .* V;
% new_vel(new_vel == 0) = NaN;
% new_vel(new_vel > 1.0) = NaN;
% clim([min(min(new_vel)), max(max(new_vel))]);
% c.Label.String = 'Conduction Velocity Magnitude (m/s)';
% axis off
% axis image
% 
% figure %last figure
% %subplot(122)
% % outside the mask, it should not have a value
% new_mask = mask3(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3));
% new_vel = new_mask .* V;
% new_vel(new_vel == 0) = NaN;
% new_vel(new_vel > 5.0) = NaN;
% 
% 
% % Set NaN color to white
% nan_color = [1, 1, 1]; % RGB values for white
% 
% % Create a colormap with NaN color
% custom_colormap = colormap(flipud(jet)); % or any other colormap you prefer
% custom_colormap(1, :) = nan_color; % Set the first row of the colormap to NaN color
% colormap(custom_colormap);
% 
% % Plot the image with NaN values
% imagesc(new_vel);
% 
% c = colorbar;
% c.Label.String = 'Conduction Velocity Magnitude (m/s)';
% 
% axis image
% axis off
% title('Conduction Velocity Magnitude')
% 
% %% Find Conduction Velocity Map - Efimov Method
% 
% % Calculate Conduction Velocity
% if ~use_window
%     Vx = Tx./(Tx.^2+Ty.^2); %not being used
%     Vy = -Ty./(Tx.^2+Ty.^2); %not being used
%     V = sqrt(Vx.^2 + Vy.^2); %not being used
% else
%     bad=(V>100); 
%     Vx(bad)=NaN;
%     Vy(bad)=NaN;
%     V(bad)=NaN;
% end
% 
% disp('Regional conduction velocity statistics:')
% meanV=nanmean(V(isfinite(V)));
% disp(['The mean value is ' num2str(meanV) ' m/s.'])
% medV = median(V(isfinite(V)));
% disp(['The median value is ' num2str(medV) ' m/s.'])
% stdV = std2(V(isfinite(V)));
% disp(['The standard deviation is ' num2str(stdV) '.'])
% meanAng = mean(atan2(Vy(isfinite(Vy)),Vx(isfinite(Vy))).*180/pi);
% disp(['The mean angle is ' num2str(meanAng) ' degrees.'])
% medAng = median(atan2(Vy(isfinite(Vy)),Vx(isfinite(Vy))).*180/pi);
% disp(['The median angle is ' num2str(medAng) ' degrees.'])
% stdAng = std2(atan2(Vy(isfinite(Vy)),Vx(isfinite(Vy))).*180/pi);
% disp(['The standard deviation of the angle is ' num2str(stdAng) '.'])
% num_vectors = numel(V(isfinite(V)));
% disp(['The number of vectors is ' num2str(num_vectors) '.'])
% 
%         % statistics window
%        handles.activeCamData.meanresults = sprintf('Mean: %0.3f',meanV);
%        handles.activeCamData.medianresults  = sprintf('Median: %0.3f',medV);
%        handles.activeCamData.SDresults = sprintf('S.D.: %0.3f',stdV);
%        handles.activeCamData.num_membersresults = sprintf('#Members: %d',num_vectors);
%        handles.activeCamData.angleresults = sprintf('Angle: %d',meanAng);
% 
%        set(handles.meanresults,'String',handles.activeCamData.meanresults);
%        set(handles.medianresults,'String',handles.activeCamData.medianresults);
%        set(handles.SDresults,'String',handles.activeCamData.SDresults);
%        set(handles.num_members_results,'String',handles.activeCamData.num_membersresults);
%        set(handles.angleresults,'String',handles.activeCamData.angleresults);
% 
% handles.activeCamData.saveData = actMap1;
% cla(movie_scrn); 
% 
% [C,h] = contourf(movie_scrn, actMap1,(endp-stat)/2, 'LineColor','k');
% 
% contourcmap('copper','SourceObject', movie_scrn, 'ColorAlignment', 'center');
% 
% set(movie_scrn,'YTick',[],'XTick',[]);
% 
% hold (movie_scrn,'on')
% if ~use_window
%     Y_plot = y(isfinite(Z_fit));
%     X_plot = x(isfinite(Z_fit));
%     Vx_plot = Vx(isfinite(Z_fit));
% else
%     Y_plot = yy(was_fitted);
%     X_plot = xx(was_fitted);
%     Vx_plot = Vx;
% end
% Vx_plot(abs(Vx_plot) > 5) = 5.*sign(Vx_plot(abs(Vx_plot) > 5));
% if ~use_window
%     Vy_plot = Vy(isfinite(Z_fit));
% else
%     Vy_plot = Vy;
% end
% Vy_plot(abs(Vy_plot) > 5) = 5.*sign(Vy_plot(abs(Vy_plot) > 5));
% V = sqrt(Vx_plot.^2 + Vy_plot.^2);
% 
% VecArray = [X_plot Y_plot Vx_plot Vy_plot V];
% handles.activeCamData.VecArray = VecArray;
% 
% handles.activeCamData.saveX_plot = X_plot;
% handles.activeCamData.saveY_plot = Y_plot;
% handles.activeCamData.saveVx_plot = Vx_plot;
% handles.activeCamData.saveVy_plot = Vy_plot;
% 
% quiver_step = 2;
% q = quiver(movie_scrn, X_plot(1:quiver_step:end),...
%            Y_plot(1:quiver_step:end),Vx_plot(1:quiver_step:end),...
%            -1.0 * Vy_plot(1:quiver_step:end),'k');
% q.LineWidth = 2;
% q.AutoScaleFactor = 2;
% set(movie_scrn,'YDir','reverse');
% hold (movie_scrn,'off');
% 
% end

