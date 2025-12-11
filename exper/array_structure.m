%% 多臂螺旋阵列绘制
clc; clear; close all;

% 参数
nArms = 6;           % 螺旋臂数
nPerArm = 10;        % 每臂阵元数
r_min = 0.02;        % 最小半径 (m)
r_max = 0.10;        % 最大半径 (m)
a = r_min;           
b = (r_max - r_min) / (2*pi);   % 控制螺旋扩展速度

theta = linspace(0, 2*pi, nPerArm);  % 每臂角度分布
figure; hold on; axis equal; box on;

% 绘制螺旋臂
for m = 1:nArms
    offset = (m-1)*2*pi/nArms;        % 臂间相位差
    for n = 1:nPerArm
        r = a + b*(theta(n) + offset);
        x = r * cos(theta(n) + offset);
        y = r * sin(theta(n) + offset);
        plot(x, y, 'ko', 'MarkerFaceColor', 'w', 'MarkerSize', 6);
    end
end

% 绘制螺旋曲线
t = linspace(0, 2*pi, 500);
r = a + b*t;
plot(r.*cos(t), r.*sin(t), 'k');

% 参数标注（示意）
text(0,0,'中心','HorizontalAlignment','center');
text(0.11,0,'r_{max}','FontSize',10);
text(0.05,0.03,'\alpha','FontSize',10);

xlabel('x (m)');
ylabel('y (m)');
title('多臂螺旋麦克风阵列结构示意图');
