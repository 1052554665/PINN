fc = 300e6;
c = physconst('LightSpeed');
array = phased.ULA('NumElements',4);
%{
创建一个 SteeringVector 对象，并计算在 fc = 300 MHz 下，目标方向 方位角 30°，仰角 20° 的导向矢量 sv
sv 是一个 4×1 复数向量，其相位差对应于波从 (30°,20°) 方向到达各阵元的路径差
将 sv 作为加权向量应用，就能让阵列波束指向该方向
%}
steervec = phased.SteeringVector('SensorArray',array);
sv = steervec(fc,[30;20]);
% 绘制未应用导向矢量和应用导向矢量时均匀线性阵列的波束方向图。
subplot(211)
pattern(array,fc,-180:180,0,'CoordinateSystem','rectangular',...
    'PropagationSpeed',c,'Type','powerdb')
title('Without steering')
subplot(212)
pattern(array,fc,-180:180,0,'CoordinateSystem','rectangular',...
    'PropagationSpeed',c,'Type','powerdb','Weights',sv)
title('With steering')