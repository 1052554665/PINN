%{
对 7 元 ULA 接收到的正弦波信号应用相移波束成形。
波束成形方向为 45° 方位角和 0° 仰角。假设阵列在 300 MHz 下运行。
%}
% 模拟信号
t = (0:1000)';
fsignal = 0.01; % % 信号频率 (归一化到采样率)
x = sin(2*pi*fsignal*t);
c = physconst('Lightspeed');
fc = 300e6;
incidentAngle = [45;0];
array = phased.ULA('NumElements',7);    
x = collectPlaneWave(array,x,incidentAngle,fc,c);   % x 是 1001×7 矩阵，每列代表一个阵元的接收信号
noise = 0.1*(randn(size(x)) + 1j*randn(size(x)));
rx = x + noise;
% 设置相移波束成形器，然后对输入数据进行波束成形
beamformer = phased.PhaseShiftBeamformer('SensorArray',array,...
    'OperatingFrequency',fc,'PropagationSpeed',c,...
    'Direction',incidentAngle,'WeightsOutputPort',true);
[y,w] = beamformer(rx);
% 在中间单元处绘制原始信号和波束形成的信号。
figure;
plot(t,real(rx(:,4)),'r:',t,real(y))
xlabel('Time (sec)')
ylabel('Amplitude')
legend('Input','Beamformed')
% 应用权重绘制阵列响应图
figure;
pattern(array,fc,[-180:180],0,'PropagationSpeed',c,'Type',...
    'powerdb','CoordinateSystem','polar','Weights',w)   % CoordinateSystem','polar'：极坐标方向图

