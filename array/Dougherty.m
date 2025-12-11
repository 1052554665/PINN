N = 128;
r_min = 0.015;
r_max = 0.100;
theta_max = 13 * pi / 2;
% Dougherty 对数螺旋: r(theta) = r_min * exp(b * theta), theta in [0, theta_max]
% 由 r(theta_max)=r_max 解出 b:
b = log(r_max / r_min) / theta_max;
a0 = r_min; % 对数螺旋常数 a

fc = 20e3;                
c = 340;                 
lambda = c / fc;
k0 = 2*pi / lambda;

%% ===== 生成阵元位置 (单臂 Dougherty 对数螺旋) =====
theta = linspace(0, theta_max, N);
r = a0 * exp(b * theta);
x = r .* cos(theta);
y = r .* sin(theta);
z = zeros(1,N);
pos = [x; y; z];  % 3xN

%% ===== 绘制阵列几何结构 =====
figure('Name','阵列几何','NumberTitle','off');
scatter(x, y, 48, 'filled'); hold on;
plot(x, y, '--k', 'LineWidth', 0.8);  % 连成一条螺旋线
text(x, y, arrayfun(@num2str, 1:N, 'UniformOutput', false), 'FontSize',7, ...
    'VerticalAlignment','bottom','HorizontalAlignment','right');
axis equal; grid on;
xlabel('x (m)'); ylabel('y (m)');
title(sprintf('Dougherty 对数螺旋阵列 几何 (N=%d)', N));

%% ===== 方向余弦网格 (ux, uy) =====
ux = linspace(-1, 1, 401);   % 足够密的网格
uy = linspace(-1, 1, 401);
[UX, UY] = meshgrid(ux, uy);
mask = (UX.^2 + UY.^2) <= 1; % 物理可达半球 (单位圆)

%% ===== 计算阵列因子（基于导向矢量） =====
AF = zeros(size(UX));
% 采用上半球 uz = +sqrt(1-ux^2-uy^2)
UZ = sqrt(max(0, 1 - UX.^2 - UY.^2)); 

for nIdx = 1:N
    xn = pos(1, nIdx);
    yn = pos(2, nIdx);
    zn = pos(3, nIdx);
    % 相位项，注意点乘： k0 * (r_n · u) where u = [ux; uy; uz]
    AF = AF + exp(1j * k0 * (xn * UX + yn * UY + zn * UZ));
end

% 归一化 & 转 dB
AF = AF ./ max(abs(AF(:)));
AFdB = 20 * log10(abs(AF) + eps);
AFdB(~mask) = -80;   % 单位圆外设为很小值，便于显示

%% ===== 绘制二维方向图 (ux-uy 平面) =====
figure('Name','二维方向图','NumberTitle','off');
imagesc(ux, uy, AFdB);
set(gca, 'YDir', 'normal');   % 保持 y 轴方向与直觉一致
axis square;
colorbar;
caxis([-60 0]);              % 显示动态范围，可根据需要调整
xlabel('u_x'); ylabel('u_y');
title('Dougherty 对数螺旋阵列 - 二维方向图 (u_x, u_y)');

% 在二维图上绘制单位圆边界以便参考
hold on;
t = linspace(0, 2*pi, 400);
plot(cos(t), sin(t), 'w--', 'LineWidth', 1.0);   % 单位圆

%% ===== 绘制三维方向图 =====
figure('Name','三维方向图','NumberTitle','off');
h = surf(UX, UY, AFdB, 'EdgeColor', 'none');
axis square; view(30,40);
shading interp;
colorbar;
caxis([-60 0]);
xlabel('u_x'); ylabel('u_y'); zlabel('增益 (dB)');
title('Dougherty 对数螺旋阵列 - 三维方向图 (u_x, u_y, dB)');