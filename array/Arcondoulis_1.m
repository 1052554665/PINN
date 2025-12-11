% Arcondoulis 螺旋阵列 —— 阵列几何 + 波束图
% 单臂螺旋、基于方向余弦坐标绘制的阵列方向图
% M控制螺旋臂数
%% 阵列参数
N = 128;    % 阵元数
M = 6;      % 单臂螺旋
a = 0.01;          % 起始半径 (m)
b = 0.15;          % 螺旋扩展率
nTurns = 3;        % 螺旋圈数
fc = 20e3;
c = 340;
lambda = c / fc;    % 波长
k = 2*pi/lambda;    % 波数

% 每臂阵元数
N_arm = N / M;

%% 单臂螺旋阵元位置
pos = zeros(3,N);
idx = 1;

for m = 0:M-1
    for k = 1:N_arm
        theta = (k - 1) / (N_arm-1) * 2 * pi * nTurns; % 螺旋角度
        r = a * exp(b*theta);   % 对数螺旋
        th = theta + 2*pi*m/M; % 单臂时 m=0，即不偏移
        x = r * cos(th);
        y = r * sin(th);
        z = 0; % 平面阵列
        pos(:,idx) = [x;y;z];
        idx = idx + 1;
    end
end



%% 定义方向余弦网格
ux = linspace(-1,1,201);
uy = linspace(-1,1,201);
[UX,UY] = meshgrid(ux,uy);

% 合法区域(ux^2 + uy^2 <= 1)
mask = (UX.^2 + UY.^2) <= 1;

%% 计算阵列因子
AF = zeros(size(UX));
for n = 1:N
    % 阵元位置
    xn = pos(1,n);
    yn = pos(2,n);
    zn = pos(3,n);

    % 相位项 exp(j*k*(ux*x + uy*y + uz*z))
    % uz = sqrt(1 - ux^2 - uy^2)，保证在单位球面上
    uz = sqrt(max(0,1 - UX.^2 - UY.^2));
    AF = AF + exp(1j * k * (xn * UX + yn * UY + zn * uz));
end

%% 归一化
AF = AF ./ max(abs(AF(:)));
AFdB = 20 * log10(abs(AF));
AFdB(~mask) = -40;  % 合法区域外填充低值


%% 阵列几何结构图
figure;
scatter(pos(1,:), pos(2,:), 40, 'ro');
hold on;
plot(pos(1,:), pos(2,:), 'k--');    % 螺旋线连接
text(pos(1,:), pos(2,:), num2str((1:N)'), 'FontSize', 6);
axis equal;
grid on;
xlabel('x (m)');
ylabel('y (m)');
title('Arcondoulis 螺旋阵列几何结构');

%% 绘制二维方向图
figure;
imagesc(ux, uy, AFdB);
axis xy;
axis square;
colorbar;
caxis([-30 0]);     % 截止范围 -30 ~ 0 dB
xlabel('u_x');
ylabel('u_y');
title('Arcondoulis 螺旋阵列二维方向图');

%% 绘制三维方向图
figure;
surf(UX, UY, AFdB,'EdgeColor','none');
axis square;
colorbar;
caxis([-30 0]);
view(30,40);
xlabel('u_x');
ylabel('u_y');
zlabel('增益 (dB)');
title('Arcondoulis 螺旋阵列三维方向图')