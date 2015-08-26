times = load('descriptor_times.txt');

detector_labels = {'SIFT', 'SURF', 'ORB', 'BRISK', 'KAZE', 'AKAZE'};
size_labels = {'300x200', '640x480', '800x600'};

figure();
hold on;
grid;
[m, n] = size(times);
ax = gca;
set(gca,'xtick',1:3)
ax.XTickLabel = size_labels;
for i=1:m
   plot(times(i,:), '*-');
   legend(detector_labels);
end

title('Descriptors Times');