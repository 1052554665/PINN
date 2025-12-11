%% 阵列参数
N = 128;    % 阵元数
r_min = 0.015;  % 最小半径（m）
r_max = 0.1;    % 最大半径（m）
theta_max = 13 * pi / 2;    % 最大旋转幅度

fc = 20e3;  % 频率（Hz）
c = 340;    % 声速（m/s）
lambda = c / fc;    % 波长
k0 = 2 * pi / lambda;   % 波数

%% 生成螺旋坐标
theta = linspace(0, theta_max, N);
r = r_min + (r_max - r_min) / theta_max * theta;

x = r .* cos(theta);
y = r .* sin(theta);
z = zeros(1, N);
pos = [x; y; z];

%% 阵列几何图
figure;
scatter(x, y, 36, 'red', 'filled');
hold on;
plot(x, y, '--k');
text(x, y, arrayfun(@num2str, 1:N, 'UniformOutput',false), 'FontSize',6);
axis equal;
grid on;
xlabel('x (m)');
ylabel('y (m)');
title('128 阵元 Archimedean 螺旋阵列几何结构');

%% 方向余弦坐标
ux = linspace(-1, 1, 201);
uy = linspace(-1, 1, 201);
[UX, UY] = meshgrid(ux, uy);
mask = (UX.^2 + UY.^2) <= 1;

%% 计算阵列因子
AF = zeros(size(UX));
for n = 1:N
    xn = pos(1,n);
    yn = pos(2,n);
    zn = pos(3,n);
    uz = sqrt(max(0, 1 - UX.^2 - UY.^2));
    AF = AF + exp(1j * k0 * (xn * UX + yn * UY + zn * uz));
end

AF = AF ./ max(abs(AF(:)));
AFdB = 20 * log10(abs(AF) + eps);
AFdB(~mask) = -60;  % 单位圆外置为低值

%% 二维方向图
figure;
imagesc(ux, uy, AFdB);
axis xy;
axis square;
colorbar;
caxis([-40 0]);
xlabel('u_x');
ylabel('u_y');
title('二维波束方向图 (方向余弦坐标)');

%% 三维方向图
figure;
surf(UX, UY, AFdB, 'EdgeColor','none');
axis square;
view(30, 40);
colorbar;
caxis([-40 0]);
xlabel('u_x');
ylabel('u_y');
zlabel('增益 (dB)');
title('三维波束方向图 (方向余弦坐标)');
