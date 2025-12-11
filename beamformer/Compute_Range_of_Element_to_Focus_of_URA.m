%{
计算从聚焦的 3 x 4 URA 的焦点到阵列元素的范围。阵列聚焦在 1000 米外的点，方位角30，仰角45。
阵列工作频率为 300 Mhz，阵列元件间隔 1/2 波长。
%}
fc = 300.0e6;
c = physconst('lightspeed');
az = 45.0;
el = 30.0;
foc = 1000.0;
lambda = c/fc;  % 聚焦点坐标定义在 (45° az, 30° el, 1000 m)
elementspacing = 0.5*lambda;
% 创建一个 3×4 的均匀矩形阵列 (URA)，即 12 个阵元，阵元间距为半波长。
% 排列方式默认是 3 行 4 列。
array = phased.URA([3,4],elementspacing);
% 创建聚焦导向矢量对象，用于计算考虑球面波的近场聚焦
fsteervec = phased.FocusedSteeringVector('SensorArray',array);
[fsvec,elemrng] = fsteervec(fc,[az;el],foc);
disp(elemrng)