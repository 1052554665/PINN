%{
FOC
焦点范围，指定为正值的标量或 M 元素行向量。
如果 ANG 有多个列，则 FOC 必须是标量或具有与 ANG 相同的列数。单位以米为单位。
%}
fc = 300.0e6;
c = physconst('lightspeed');
az = 45.0;
el = 30.0;
foc = 1000.0;   % 焦点距离 (Focus Range)，即波束聚焦在空间中某一点的距离（近场波束形成）
lambda = c/fc;
elementspacing = 0.4*lambda;
nelem = 11;
array = phased.ULA(nelem,elementspacing);
% 创建 聚焦导向矢量对象，它和 phased.SteeringVector 类似，但多了一个焦点距离的参数。
fsteervec = phased.FocusedSteeringVector('SensorArray',array);
% 计算在300 MHz下，阵列对焦点 (45°,30°,1000 m) 的聚焦导向矢量 (focused steering vector)
fsvec = fsteervec(fc,[az;el],foc);  % fsvec 是一个 11×1 的复数权重向量，可以直接用来加权阵列信号，以便聚焦到近场目标
