% Third Degree Velocity Calculation

function [Vx, Vy, V] = xyt_matrix_third(M,xyt,space_window_width,time_window_width,xres,yres,max_Vx,max_Vy)

XYT = zeros(M,15);

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

    fit     = [y.*x.^2 x.*y.^2 y.^3 x.^3 x.*y y.^2 x.^2 y x ones(length(near),1)];

    % find coefficienta (a-f) such that Aa=t. these are your a-f

    coefs = fit\time;

    % find error. not sure why they use this specific error and what a
    % meaningful number would be. we want it to be small.
    var_t = sum((time-mean(time)).^2);
    resi = sqrt(sum((time-fit*coefs).^2)/var_t);
    resilin = sqrt(sum((time-fit(:,8:10)*coefs(8:10)).^2)/var_t);

    % store coords of pixel, time, coefs, and error

    XYT(num_pix,:) = [xyt(num_pix,:), coefs',resi, resilin];


end
end

% Function name: vel_calc_third
% Function Summary: Same as function below just using the third degree
% polynomial (eerything is the same but the coefficient number you use is
% different which is why this is separate function).
% Inputs: XYT - matrix with pixels info including coefficients for the
% surface at each pixel. max values of velocities in our range (user
% defined).
% Outputs: Vx - line vector of x velocities for each pixel in your ROI. Vy
% - line vector of y velocities for each pixel in your ROI. V - line
% vector of total velocity magnitude for each pixel in your ROI.


Vx=(XYT(:,12)./(XYT(:,11).^2 + XYT(:,12).^2))*100; % times 100 to go from mm/msec to cm/sec
Vy=(XYT(:,11)./(XYT(:,11).^2 + XYT(:,12).^2))*100;
Vx(abs(Vx) > max_Vx) = NaN;
Vy(abs(Vy) > max_Vy) = NaN;
% standard_dev = std(Vx);
% standard_dev_y = std(Vy);
% Vx(abs(Vx) > standard_dev*2) = NaN;
% Vy(abs(Vy) > standard_dev_y*2) = NaN;
V = sqrt(Vx.^2+Vy.^2);
sdev = nanstd(V);
avg = nanmean(V);
V(V>(avg+sdev+sdev)) = NaN;
V(V<(avg-sdev-sdev)) = NaN;
end