function [mos_img]=mosaic()
% creates a mosaic image from a movie
% assuming translation only between frames



warning off MATLAB:mir_warning_variable_used_as_function;

% using global names to cache frames in memory
% performance boosting
global Nframes;
global AviName;
filename='vid.avi';
AviName = filename;
xx= VideoReader(AviName);

h=xx.Height;
    w=xx.Width;
    Nframes= xx.NumberOfFrames;

% relative displacements
d = zeros(2,Nframes);

RefFrame = GetImage(1);

% align every two consequent frames
for ii=2:Nframes

    % read a frame
    Frame = GetImage(ii);

    d(:,ii) = align(RefFrame, Frame, d(:,ii-1));

    % current frame is next one's reference
    RefFrame = Frame;
end

% cumulative displacement relative to first frame
D = cumsum(d,2);

% minimal
Minimum = min(D,[],2);
%maximal
Maximum = max(D,[],2);

% minimal displacement is one
D = D - Minimum*ones(1,Nframes)+1;

% create an empty mosaic (compute the bounding box)
mos_img = zeros([w, h]+ceil(Maximum'-Minimum'))';
[bm bn] = size(mos_img);
% creates weights
weights = zeros(bm,bn);

% shfoch everything into the mosaic
for ii=1:Nframes
    % read a frame
    Frame = GetImage(ii);

    tmpF(1:(h+1),1:(w+1)) = 0;
    tmpF(2:h+1,2:w+1) = Frame;
    intD = floor(D(:,ii));
    wF = warp(tmpF, (D(:,ii)-intD));
    wF(find(isnan(wF))) = 0;

    % mask
    m = zeros(size(tmpF));
    m(2:h+1,2:w+1) = 1;
    wm = warp(m, (D(:,ii)-intD));
    wm(find(isnan(wm))) = 0;

    [A B] = GetPosArray(bm, h, bn, w, intD);
    mos_img(A, B) = mos_img(A, B) + wF;
    weights(A, B) = weights(A, B) + wm;

end

weights = weights + (weights == 0); % in order to avoid zero division
mos_img = mos_img./weights;



function [A, B] = GetPosArray(bm, h, bn, w, intD)
% Gets an array, where to put the image.
% bm, bn - the total image's size
% h, w - single frame's size
% intD - the integer displacement.
A = bm - h + 1 - intD(2): bm - intD(2) + 1;
B = bn - w + 1- intD(1): bn - intD(1) + 1;



function image = GetImage(index)
% Gets the image data in the correct form (Double, greyscale).
% using chached frames to boost performance

% caching BUFFER_SIZE images at each access to the avi file
BUFFER_SIZE = 40;
persistent CurrentBlock M;
global AviName Nframes;

if (isempty(CurrentBlock))
    CurrentBlock = -1;
end;

% zindex to start from zero rather than one.
zindex = index - 1;

if (floor(zindex / BUFFER_SIZE) ~= CurrentBlock)
    CurrentBlock = zindex / BUFFER_SIZE;
    ReadTo = min((CurrentBlock+1)*BUFFER_SIZE, Nframes);
    M = VideoReader(AviName);
end

localIndex = mod(zindex, BUFFER_SIZE) + 1;

%returning the right image in the correct format.
image = im2double(rgb2gray(M(localIndex).cdata ));



function [shift] = align(im1, im2, initial_shift)
% This function aligns im2 to im1 assuming an homography consist of shift only
% Input parameters
%   im1, im2 - images in double representation grey level range [0,1]
%
% The output parameters:
%   shift - a vector [dx, dy]' of displacement



% initial guess for displacement
shift = [0, 0]';
if (nargin == 3)
    shift = initial_shift;
end;

% global parameters
ITER_EPS = 0.001;
PYRAMID_DEPTH = 5;
ITERATIONS_PER_LEVEL = 5;
FILTER_PARAM = 0.4;         % parameter for gaussian pyramid

% Creating pyramids for both images.
F = gaussian_pyramid(im1, PYRAMID_DEPTH, FILTER_PARAM);
G = gaussian_pyramid(im2, PYRAMID_DEPTH, FILTER_PARAM);

% Allowing initial shift - scale it to fit pyramid level
shift = shift ./ (2^(PYRAMID_DEPTH+1));

% work coarse to fine in pyramids
for ii = PYRAMID_DEPTH : -1 : 1 %PYRAMID_DEPTH
    clear warped_im2;    % we are about to change its dimensionality

    shift = 2 * shift;   % compensate the scale factor between pyramid levels.

    % compute the gradient once per pyramid level
    [Fx Fy] = gradient(F{ii});

    % although we can re-calculate it according to overlapping zone
    % it is negligible thus we save computation by doing it once per level
    tmp = sum(Fx(:).*Fy(:));
    A = inv( [ sum(Fx(:).^2) tmp ; tmp sum(Fy(:).^2) ] );

    % iterate per pyramid level
    for jj = 1 : ITERATIONS_PER_LEVEL
        % always warp original image by accumulated shift, avoid over
        % blurring
        warped_im2 = warp(G{ii}, shift);

        % mask to rule out boundary (NaN values)
        mask = isnan(warped_im2);
        warped_im2(find(mask)) = 0;
        mask = 1-mask;

        % difference image - ignore parts that are not overlapping
        diff = ( F{ii}(:)-warped_im2(:) ).*mask(:);

        % Solve and update shift;
        tmpshift =  A*[sum(diff.*Fx(:)) ; sum(diff.*Fy(:))] ;
        shift = shift + tmpshift;
        if (abs(tmpshift) < ITER_EPS)
            break;
        end;
    end;
end;

% wrapping an image with shift using bilinear interpolation
function [warped] = warp(image, shift)
[x y] = size(image);
[X Y] = meshgrid(1:y,1:x); % Behold!!!
warped = interp2(image, X+shift(1), Y+shift(2), 'linear');


function [G] = gaussian_pyramid(image, n, a)
% compute n gaussian pyramid of an image
% a - filter parameter

G = cell(n,1);
G{1} = image;
for ii=1:n-1
    % use tmp variable to comply with varying size of image through down sampling
    G{ii+1}=gaussian_pyramid_one_stage(G{ii}, a);
end


function [G]=gaussian_pyramid_one_stage(im, a)
% compute one stage of a gaussian pyramid of an image
% a is the filter parameter

[m n]=size(im);

% construct the filter
filterKernel = [0.25 - (a/2), 0.25, a, 0.25, 0.25-(a/2)];
h = filterKernel'*filterKernel;

% filter the image, i.e. blur it with the gaussian, replicate to overcome
% boundary conditions
filt = imfilter(im, h, 'replicate');

% down sample
G = filt(1:2:m,1:2:n);
