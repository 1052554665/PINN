%{
将 Frost 波束成形应用于 11 个元件的声学 ULA 阵列。
输入信号的入射角方位角为 -50 度，仰角为 30 度。
假设空气中的声速为 340 m/sec。该信号添加了高斯白噪声。
%}
% 模拟宽带测试信号(chirp)
array = phased.ULA('NumElements',11,'ElementSpacing',0.04);
array.Element.FrequencyRange = [20 20000];  % 声学常用频率范围（20 Hz ~ 20 kHz）
fs = 8e3;   % 采样率 8 kHz
t = 0:1/fs:0.3; % 0.3 秒信号
x = chirp(t,0,1,500);   % 从 0 Hz 扫到 500 Hz 的线性调频信号 (chirp)
c = 340;
% NumSubbands = 8192 用大量子带来逼近连续宽带信号传播
% ModulatedInput = false 表示输入信号为基带
collector = phased.WidebandCollector('Sensor',array,...
    'PropagationSpeed',c,'SampleRate',fs,...
    'ModulatedInput',false,'NumSubbands',8192); 
incidentAngle = [-50;30];
x = collector(x.',incidentAngle);   % x是一个 N×11 矩阵，包含了阵列接收到的 chirp 信号
noise = 0.2*randn(size(x));
rx = x + noise; % 给阵列接收信号加上高斯白噪声，模拟实际声学环境
% 波束成形信号
% FilterLength = 5：每个阵元对应的自适应 FIR 滤波器长度（这里是 5 taps）。
% Frost 算法会动态调整这些滤波器权重，以抑制干扰、增强目标方向信号
beamformer = phased.FrostBeamformer('SensorArray',array,...
    'PropagationSpeed',c,'SampleRate',fs,...
    'Direction',incidentAngle,'FilterLength',5);
y = beamformer(rx);
plot(t,rx(:,6),'r:',t,y)
xlabel('Time')
ylabel('Amplitude')
legend('Original','Beamformed')