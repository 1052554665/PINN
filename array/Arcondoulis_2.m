%% Arcondoulis 阿基米德螺旋 多臂 基于方向余弦坐标绘制的方向图
N = 64;    % 阵元总数
M = 8;      % 螺旋臂数
a = 0.01;   % 起始半径(m)
b = 0.005;  % 螺距（阿基米德螺旋）
nTurns = 3; % 螺旋圈数
fc = 20e3;  % 频率
c = 340;    % 声速
lambda = c / fc     % 波长
k0 = 2*pi/lambda;   % 波数

% 每臂阵元数
N_arm = N / M;

%% 生成螺旋阵列坐标
pos = zeros(3, N);
idx = 1;
for m = 0:M-1
    for n_arm = 1:N_arm
        theta = (n_arm - 1) / (N_arm - 1) * 2 * pi * nTurns;    % 螺旋角度
        r = a + b * theta;  % 阿基米德螺旋
        th = theta + 2 * pi * m / M;    % 多臂相位偏移
        x = r * cos(th);
        y = r * sin(th);
        z = 0;
        pos(:,idx) = [x;y;z];
        idx = idx + 1;
    end
end

% 定义方向余弦网格
ux = linspace(-1,1,201);
uy = linspace(-1,1,201);
[UX,UY] = meshgrid(ux, uy);

% 合法区域
mask = (UX.^2 + UY.^2) <= 1;

%% 计算阵列因子
AF = zeros(size(UX));
for n = 1:N
    xn = pos(1,n);
    yn = pos(2,n);
    zn = pos(3,n);
    uz = sqrt(max(0,1 - UX.^2 - UY.^2));    % 保证在单位球面上
    AF = AF + exp(1j * k0 * (xn * UX + yn * UY + zn * uz));
end

%% 归一化
AF = AF ./ max(abs(AF(:)));
AFdB = 20 * log10(abs(AF));
AFdB(~mask) = -40;  % 合法区域外填充低值

%% 阵列几何结构图
figure;
scatter(pos(1,:), pos(2,:), 40,'ro','filled');
hold on;
for m = 1:M
    idx_arm = (m-1) * N_arm + (1:N_arm);
    plot(pos(1,idx_arm),pos(2,idx_arm), '--k');
end
axis equal;
grid on;
xlabel('x (m)');
ylabel('y (m)');
title(sprintf('Arcondoulis 螺旋阵列几何结构 (M=%d 臂)', M));

%% 绘制二维方向图
figure;
imagesc(ux, uy, AFdB);
axis xy;
axis square;
colorbar;
caxis([-30 0]);
xlabel('u_x');
ylabel('u_y');
title(sprintf('Arcondoulis 螺旋阵列二维方向图 (M=%d 臂)', M));

%% 绘制三维方向图
figure;
surf(UX, UY, AFdB, 'EdgeColor','none');
axis square;
colorbar;
caxis([-30 0]);
view(30,40);
xlabel('u_x');
ylabel('u_y');
zlabel('增益 (dB)');
title(sprintf('Arcondoulis 螺旋阵列三维方向图 (M=%d 臂)', M));