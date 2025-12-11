%{
计算并显示方位角 30 度和仰角 20 度方向的 4 单元均匀线性阵列的转向矢量。
假设阵列的工作频率为300 MHz。
%}
array = phased.ULA('NumElements',4);
steervec = phased.SteeringVector('SensorArray',array);
fc = 3e8;
ang = [30;20];
sv = steervec(fc,ang)