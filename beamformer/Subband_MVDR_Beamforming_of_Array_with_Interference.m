%{
将子带 MVDR 波束成形应用于水声 11 单元 ULA。
对到达的信号进行波束成形，以优化从 0 度方位角和 0 度仰角到达的线性 FM 线性调频信号的增益。
信号的带宽为 2.0 kHz。此外，还有单位振幅 2.250 kHz 的干扰正弦波从 28 度方位角和 0 度仰角到达。
展示MVDR波束形成器如何使干扰信号归零。显示 2.250 kHz 附近多个频率的方向图。声速为1500米/秒。
%}
% 模拟到达信号和噪声
array = phased.ULA('NumElements',11,'ElementSpacing',0.3);
fs = 2000;
carrierFreq = 2000;
t = (0:1/fs:2)';
sig = chirp(t,0,2,fs/2);
c = 1500;
collector = phased.WidebandCollector('Sensor',array,'PropagationSpeed',c,...
    'SampleRate',fs,'ModulatedInput',true,...
    'CarrierFrequency',carrierFreq);
incidentAngle = [0;0];
sig1 = collector(sig,incidentAngle);
noise = 0.3*(randn(size(sig1)) + 1j*randn(size(sig1)));
% 模拟干扰信号
fint = 2250;
sigint = sin(2*pi*fint*t);
interfangle = [28;0];
sigint1 = collector(sigint,interfangle);
rx = sig1 + sigint1 + noise;
% 使用组合噪声和干扰信号作为训练数据。
beamformer = phased.SubbandMVDRBeamformer('SensorArray',array,...
    'Direction',incidentAngle,'OperatingFrequency',carrierFreq,...
    'PropagationSpeed',c,'SampleRate',fs,'TrainingInputPort',true,...
    'NumSubbands',64,...
    'SubbandsOutputPort',true,'WeightsOutputPort',true);
[y,w,subbandfreq] = beamformer(rx, sigint1 + noise);
tidx = [1:300];
figure;
plot(t(tidx),real(rx(tidx,6)),'r:',t(tidx),real(y(tidx)))
xlabel('Time')
ylabel('Amplitude')
legend('Original','Beamformed')

fdx = [5,7,9,11,13];
figure;
pattern(array,subbandfreq(fdx).',-50:50,0,...
    'PropagationSpeed',c,'Weights',w(:,fdx),...
    'CoordinateSystem','rectangular');
