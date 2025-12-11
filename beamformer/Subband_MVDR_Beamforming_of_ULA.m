%{
将子带MVDR波束形成技术应用于水下声学11单元线性阵列。
信号入射角为方位角10∘、仰角0∘。该信号为带宽1 kHz的调频线性调频信号。声速为1500 m/s。
%}
array = phased.ULA('NumElements',11,'ElementSpacing',0.3);
fs = 2e3;
carrierFreq = 2000;
t = (0:1/fs:2)';
sig = chirp(t,0,2,fs/2);    % 信号最大频率是 1 kHz，而载波频率是 2 kHz。因此，实际信号频谱在 2±0.5 kHz
c = 1500;   % 水中声速
collector = phased.WidebandCollector('Sensor',array,'PropagationSpeed',c,...
    'SampleRate',fs,'ModulatedInput',true,...
    'CarrierFrequency',carrierFreq);    % ModulatedInput = true 表示输入的是低频基带信号，会自动调制到载波频率 carrierFreq=2kHz
incidentAngle = [10;0];
sig1 = collector(sig,incidentAngle);
noise = 0.3*(randn(size(sig1)) + 1j*randn(size(sig1)));
rx = sig1 + noise;
% SubbandMVDRBeamformer 将信号分解为子带，然后在每个子带上执行 MVDR
% TrainingInputPort = true → 使用噪声作为训练数据，估计协方差矩阵
beamformer = phased.SubbandMVDRBeamformer('SensorArray',array,...
    'Direction',incidentAngle,'OperatingFrequency',carrierFreq,...
    'PropagationSpeed',c,'SampleRate',fs,'TrainingInputPort',true,...
    'SubbandsOutputPort',true,'WeightsOutputPort',true);
[y,w,subbandfreq] = beamformer(rx,noise);
% 绘制输入到中间传感器（通道 6）的信号与波束形成器输出的信号。
% y → 波束形成后的时域信号；w → 每个子带的波束形成权重；subbandfreq → 子带频率中心点
plot(t(1:300),real(rx(1:300,6)),'r:',t(1:300),real(y(1:300)))
xlabel('Time')
ylabel('Amplitude')
legend('Original','Beamformed');
% 绘制五个波段的方向图
pattern(array,subbandfreq(1:5).',-180:180,0,...     % 绘制前 5 个子带的方向图
    'PropagationSpeed',c,'Weights',w(:,1:5));
