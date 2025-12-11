% Arcondoulis 螺旋阵列 —— 阵列几何 + 波束图
% 基于俯仰角绘制的阵列方向图
%% 阵列参数
N = 128;    % 总阵元数
M = 1;      % 螺旋臂数
a = 0.01;   % 起始半径(m)
b = 0.15;   % 螺旋扩展率
nTurns = 3; % 螺旋圈数
fc = 20e3;  % 工作频率(Hz)
c = 340;   % 声速(m/s)(假设水声场景)

lambda = c / fc;    % 波长

%% 生成阵元位置
N_arm = N / M;
pos = zeros(3, N);

idx = 1;
for m = 0:M-1
    for k = 1:N_arm
        theta = (k-1) / (N_arm - 1) * 2 * pi * nTurns;  % 角度分布
        r = a * exp(b * theta);     % 半径（等角螺旋）  对数螺旋

        % 螺旋臂相位偏移
        th = theta + 2*pi*m/M;

        % 笛卡尔坐标
        x = r * cos(th);
        y = r * sin(th);
        z = 0;   % 平面阵列

        pos(:,idx) = [x;y;z];
        idx = idx + 1;
    end
end

% 阵列几何示意图
figure;
scatter(pos(1,:),pos(2,:),50,'filled');
hold on;
text(pos(1,:),pos(2,:),num2str((1:N)'), 'FontSize', 7);   % 标号
axis equal;
grid on;
xlabel('x (m)');
ylabel('y (m)');
title('Arcondoulis 螺旋阵列');


%% 构造阵列对象
array = phased.ConformalArray('ElementPosition',pos);

%% 二维波束图（方位角切片）
figure;
pattern(array,fc, ...
    [-180:180],0, ...       % azimuth扫描, elevation=0
    'PropagationSpeed',c, ...
    'CoordinateSystem','polar', ...
    'Type','directivity');
title('Arcondoulis螺旋阵列——二维波束图');

%% 三维波束图
figure;
pattern(array,fc, ...
    [-180:180],[-90:90], ...    % azimuth & elevation
    'PropagationSpeed',c, ...
    'CoordinateSystem','polar', ...
    'Type','directivity');
title('Arcondoulis螺旋阵列——三维波束图');