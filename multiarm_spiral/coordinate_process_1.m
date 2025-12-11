%% 将.xlsx转换为.csv
data = readmatrix('mic_positions.xlsx');
writematrix(data, 'mic_positions.csv');
data = readmatrix('mic_positions.csv');
X = data(:,1);
Y = data(:,2);
Z = zeros(size(X));
mic_pos = [X Y Z];

figure;
scatter(X, Y, 50, 'filled');
axis equal;
grid on;
xlabel('X (m)');
ylabel('Y (m)');
title('Microphone Array Geometry');

% exportgraphics(gca,'spiral_array.png','Resolution', 600);
