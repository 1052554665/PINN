%{
对 5 元件 ULA 接收到的信号应用相移波束成形。波束成形方向为 45° 方位角和 0° 仰角。
假设阵列工作频率为300 MHz。使用输入端口指定波束成形方向。
%}
% 构造一个基带正弦波信号，长度为 1001 点
t = (0:1000)';  % 时间采样点，1001个样本
fsignal = 0.01;     % 信号频率(相对低频，单位为采样点的倒数)
x = sin(2*pi*fsignal*t);
c = physconst('LightSpeed');
fc = 300e6;
incidentAngle = [45;0];
array = phased.ULA('NumElements',5);
% collectPlaneWave 用来模拟 平面波入射到阵列时，每个阵元接收到的信号（包含相位差）
% x 是 1001 × 5 的矩阵，表示 5 个阵元接收到的信号
x = collectPlaneWave(array,x,incidentAngle,fc,c);   
noise = 0.1*(randn(size(x)) + 1j*randn(size(x)));
rx = x + noise;
% 构建相移波束形成器，然后对输入数据进行波束成形
beamformer = phased.PhaseShiftBeamformer('SensorArray',array,...
    'OperatingFrequency',fc,'PropagationSpeed',c,...
    'DirectionSource','Input port', 'WeightsOutputPort',true);
% 获取波束形成信号和波束形成器权重
[y,w] = beamformer(rx,incidentAngle);
% 在中间单元处绘制原始信号和波束形成的信号
figure;
plot(t,real(rx(:,3)),'r:',t,real(y))    % rx(:,3)：阵列第3个单元接收到的原始信号；y：波束形成后的输出信号
xlabel('Time')
ylabel('Amplitude')
legend('Original','Beamformed')
figure;
% 应用权重绘制阵列响应图
pattern(array,fc,[-180:180],0,'PropagationSpeed',c,'CoordinateSystem','rectangular','Weights',w)