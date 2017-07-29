% videoFReader = vision.VideoFileReader('vid.MP4');
% videoPlayer = vision.VideoPlayer;
% v = VideoWriter('newfile.avi','Motion JPEG AVI');
% 
% i=0;
% v.FrameRate = 12;
% open(v);
% while ~isDone(videoFReader)
%    i=i+1;
%    image_hazy = step(videoFReader);
% %    I = rgb2gray(frame);
% %    display(i);
%    %step(videoPlayer,frame);
%    
%    gamma = 1;
%    A = reshape(estimate_airlight(im2double(image_hazy).^(gamma)),1,1,3);
% 
%    % Dehaze the image	
%    [image_dehazed, transmission_refined] = non_local_dehazing(image_hazy, A, gamma );
% 
%    % Display results
% %    figure('Position',[50,50, size(image_hazy,2)*3 , size(image_hazy,1)]);
% %    subplot(1,3,1); imshow(image_hazy);    title('Hazy input')
% %    subplot(1,3,2); imshow(image_dehazed); title('De-hazed output')
% %    subplot(1,3,3); imshow(transmission_refined); colormap('jet'); title('Transmission')
% 
%    
%    writeVideo(v,image_dehazed);
% end
% % imshow(frame);
% close(v);
% release(videoFReader);
% release(videoPlayer);

image_hazy = imread('img.png');

gamma = 1;
A = reshape(estimate_airlight(im2double(image_hazy).^(gamma)),1,1,3);

% Dehaze the image	
[image_dehazed, transmission_refined] = non_local_dehazing(image_hazy, A, gamma );

imshow(image_dehazed);