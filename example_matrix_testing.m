%% Example matrix testing

%activation_times = [0,1,2,3,5;1,2,3,4,5;2,3,4,9,4;3,4,5,6,7;4,5,6,7,8];

activation_times = zeros(50,50);
for i = 1:50
    activation_times(i,:) = i:i+49;
end




figure
imagesc(activation_times)



% Size of the grid (assuming it's a square grid)
grid_size = size(activation_times);
nrows = grid_size(1);
ncols = grid_size(2);

% Neighborhood size
neighborhood_size = 3;

% Initialize vectors to store results
vector_field = zeros(nrows, ncols, 2); % 2 for x and y components of vectors

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

% Display or use vector_field as needed


U = reshape(vector_field(:,:,1),[],1);
V = reshape(vector_field(:,:,2),[],1);

figure
imagesc(activation_times)
colorbar
hold on
X = zeros(50,50);
for i = 1:50
    X(i,:) = 1:50;
end
Y = zeros(50,50);
for i = 1:50
    Y(i,:) = i;
end

X = reshape(X,[],1);
Y = reshape(Y,[],1);

q = quiver(X,Y,U,V,'k');
