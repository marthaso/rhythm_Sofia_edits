% Second Degree Velocity Calculation

function [Vx, Vy, V] = xyt_matrix(M,xyt,space_window_width,time_window_width,xres,yres,max_Vx,max_Vy)

% This is your final output matrix that you will build up.
XYT = zeros(M,14);

for num_pix = 1:M %go through each pixel change back to 1

    % Find how far spatially and temporally each pixel is from the one you
    % are evaluating

    dx = abs(xyt(:,1)-xyt(num_pix,1));
    dy=abs(xyt(:,2)-xyt(num_pix,2));
    dt=abs(xyt(:,3)-xyt(num_pix,3));

    % Find the pixels near your pixel

    near = find ((sqrt(dx.^2 + dy.^2) <= space_window_width) & (isfinite(xyt(:,3))) & (dt <= time_window_width));
    len = length(near);

    how_many = 8;

    if len>how_many

        % xyt(near, 1:2) - get the x and y coord of each near pixel
        % ones(length(near),1) - make a vector of 1s the size of the number of
        % near pixels
        % xyt(i, 1:2) - get the x and y coords of your pixel
        % substract your pixel's location from all the near neighbors. make a
        % vector which says for each near pixel, how far (x and y) it is from
        % your pixel
        % xyn = xyt(near, 1:2) - ones(length(near),1) * xyt(num_pix,1:2);
        %Electro Map
        xyn = xyt(near, :) - ones(length(near),1) * xyt(num_pix,:);

        % convert pixel distance to physical distance. xres and yres are user
        % inputs
        %
        x = xyn(:,1) * xres;
        y = xyn(:,2) * yres;
        time = xyt(near,3);

        % time=xyn(:,3);
        % x=xyn(:,1);
        % y=xyn(:,2);

        % create your A matrix. (look at powerpoint for more info)

        % fit = [x.^2  y.^2  x.*y  x  y  ones(length(near),1)];

        %ElectroMap

        fit = [ones(len,1) x y x.^2 y.^2 x.*y];

        % find coefficienta (a-f) such that Aa=t. these are your a-f

        coefs = fit\time;

        % From ElectroMap
        var_t = sum((time-mean(time)).^2);
        resi = sqrt(sum((time-fit*coefs).^2)/var_t);
        resilin = sqrt(sum((time-fit(:,1:3)*coefs(1:3)).^2)/var_t);

        % find error. not sure why they use this specific error and what a
        % meaningful number would be. we want it to be small.

        % resi    = sqrt(sum((time-fit*coefs).^2));
        % resilin = sqrt(sum((time-fit(:,4:6)*coefs(4:6)).^2)/sum(time.^2));

        % store coords of pixel, time, coefs, and error

        XYT(num_pix,:) = [xyt(num_pix,:), coefs',resi, len, cond(fit), resilin,sqrt(var_t)];
    end

end


% NEED TO LOOK MORE INTO THIS. When you compute coefficients, only
% two of them differ from zero so your derivative wrt x is just the
% 7th coefficient. the derivative wrt y is just the 8th
% coefficient. % to find Vx, we need to do Tx / (Tx2 + Ty2) where
% Tx is the derivative wrt x, Ty is the derivative wrt y.; Same for
% Vy
Vx = (XYT(:,5)./(XYT(:,5).^2 + XYT(:,6).^2))*100; % times 100 to go from mm/msec to cm/sec
Vy = (XYT(:,6)./(XYT(:,5).^2 + XYT(:,6).^2))*100;
Vx(abs(Vx) > max_Vx) = NaN;
Vy(abs(Vy) > max_Vy) = NaN;
% standard_dev = std(Vx);
% standard_dev_y = std(Vy);
% Vx(abs(Vx) > standard_dev*2) = NaN;
% Vy(abs(Vy) > standard_dev_y*2) = NaN;
% Get velocity magnitude
V = sqrt(Vx.^2+Vy.^2);
sdev = nanstd(V);
avg = nanmean(V);
V(V>(avg+sdev+sdev)) = NaN;
V(V<(avg-sdev-sdev)) = NaN;
end
