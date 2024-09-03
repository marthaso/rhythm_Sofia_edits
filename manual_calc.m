%% Manual testing of conduction velocity

function [Vx, Vy, V] = manual_calc(activation_times)


% Size of the grid (assuming it's a square grid)
grid_size = size(activation_times);
nrows = grid_size(1);
ncols = grid_size(2);

% Neighborhood size
neighborhood_size = 3;

% Initialize vectors to store results
vector_field = zeros(nrows, ncols, 2); % 2 for x and y components of vectors
start = neighborhood_size-1;


% Iterate over each point in the grid
for i = 2:nrows-1
    for j = 2:ncols-1
        % Extract local neighborhood
        neighborhood = activation_times(i-1:i+1, j-1:j+1);

        % Create coordinate grid for least squares fitting
        [X, Y] = meshgrid(-1:1, -1:1);

        % Flatten matrices for least squares calculation
        A = [ones(numel(X), 1), X(:), Y(:)];
        b = neighborhood(:);

        % Perform least squares fit
        coeffs = A\b;

        % Extract coefficients for vector (ignore constant term)
        vector_x = coeffs(2);
        vector_y = coeffs(3);

        % Store vector in vector field
        vector_field(i, j, 1) = vector_x;
        vector_field(i, j, 2) = vector_y;
    end
end

Vx = (vector_field(:,:,1)./(vector_field(:,:,1).^2 + vector_field(:,:,2).^2))*100; % times 100 to go from mm/msec to cm/sec
Vy = (vector_field(:,:,2)./(vector_field(:,:,1).^2 + vector_field(:,:,2).^2))*100;
sdev = nanstd(Vx);
avg = nanmean(Vx);
Vx(Vx>(avg+sdev+sdev)) = NaN;
Vx(Vx<(avg-sdev-sdev)) = NaN;
sdev = nanstd(Vy);
avg = nanmean(Vy);
Vy(Vy>(avg+sdev+sdev)) = NaN;
Vy(Vy<(avg-sdev-sdev)) = NaN;


U = reshape(Vx,[],1);
V = reshape(Vy,[],1);

figure
imagesc(activation_times)
colorbar
hold on
croppedAmap = activation_times;
X = zeros(size(croppedAmap,1),size(croppedAmap,2)) ;

for i = 1:size(croppedAmap,1)
    X(i,:) = 1:size(croppedAmap,2);
end
Y = zeros(size(croppedAmap,1),size(croppedAmap,2)) ;
for i = 1:size(croppedAmap,2)
    Y(i,:) = i;
end

X = reshape(X,[],1);
Y = reshape(Y,[],1);

q = quiver(X,Y,U,V,'k');


% We need to change from pixel to distance. current vectors are pixel/msec

% Vx = vector_field(:,:,1);
% Vy = vector_field(:,:,2);

% Now find magnitude
V = sqrt(Vx.^2+Vy.^2);%.*17.49;


end




% % Display or use vector_field as needed
% 
% 
% U = reshape(vector_field(:,:,1),[],1);
% V = reshape(vector_field(:,:,2),[],1);
% 
% figure
% imagesc(activation_times)
% colorbar
% hold on
% X = zeros(50,50);
% for i = 1:50
%     X(i,:) = 1:50;
% end
% Y = zeros(50,50);
% for i = 1:50
%     Y(i,:) = i;
% end
% 
% X = reshape(X,[],1);
% Y = reshape(Y,[],1);
% 
% q = quiver(X,Y,U,V,'k');
% 
% 
% 
% 
% 
% % Make a matrix of zeros the size of your FOV
% maskvel2 = zeros(size(bg));
% % Place your velocity matrix in the right position
% % (ROI)
% maskvel2(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)) = CVmap_2;
% % Turn any nan values into 0
% maskvel2(isnan(maskvel2))=0;
% 
% 
% % Plot velocity values on top of the gray background
% figure
% G = real2rgb(bg, 'gray');
% imagesc(G)
% hold on
% imagesc(maskvel2, 'AlphaData', maskvel2)
% colormap(flipud(jet));
% c = colorbar;
% %title(['Velocity Magnitude ', num2str(space), ' Second Degree'])
% c.Label.String = 'Velocity Magnitude (cm/s)';
% axis off
% 
% figure
% histogram(CVmap_2)