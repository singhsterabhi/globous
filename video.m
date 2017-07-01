videoFReader = vision.VideoFileReader('vid.MP4');
videoPlayer = vision.VideoPlayer;
v = VideoWriter('newfile.avi','Motion JPEG AVI');

i=0;
v.FrameRate = 12;
open(v);
while ~isDone(videoFReader)
   i=i+1;
   frame = step(videoFReader);
   I = rgb2gray(frame);
   display(i);
   %step(videoPlayer,frame);
   writeVideo(v,I);
end
imshow(frame);
close(v);
release(videoFReader);
release(videoPlayer);