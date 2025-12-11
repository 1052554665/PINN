%% 直接读取.xlsx

data = readmatrix('mic_positions.xlsx') % 默认读取所有数值

% 提取 X、Y 坐标
X = data(:, 1);
Y = data(:, 2);

% 假设阵列在 z=0 平面
Z = zeros(size(X));

% 组合为 128x3 的坐标矩阵 [x y z]
mic_pos = [X Y Z];

% 检查结果
disp(size(mic_pos))

figure;
scatter(X, Y, 50, 'filled');
axis equal;
grid on;
xlabel('X (m)');
ylabel('Y (m)');
title('Microphone Array Geometry');

exportgraphics(gca,'spiral_array.png','Resolution', 600);