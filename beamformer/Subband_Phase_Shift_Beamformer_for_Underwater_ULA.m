% 将子带相移波束成形应用于 11 元件水下 ULA。宽带信号的入射角方位角为 10°，仰角为 30°。载波频率为 2 kHz。
antenna = phased.ULA('NumElements',11,'ElementSpacing',0.3);
antenna.Element.FrequencyRange = [20 20000];
fs = 1e3;
carrierFreq = 2e3;
t = (0:1/fs:2)';
x = chirp(t,0,2,fs);
c = 1500;
collector = phased.WidebandCollector('Sensor',antenna,...
    'PropagationSpeed',c,'SampleRate',fs,...
    'ModulatedInput',true,'CarrierFrequency',carrierFreq);
incidentAngle = [10;30];
x = collector(x,incidentAngle);
noise = 0.3*(randn(size(x)) + 1j*randn(size(x)));
rx = x + noise;
beamformer = phased.SubbandPhaseShiftBeamformer('SensorArray',antenna,...
    'Direction',incidentAngle,'OperatingFrequency',carrierFreq,...
    'PropagationSpeed',c,'SampleRate',fs,'SubbandsOutputPort',true,...
    'WeightsOutputPort',true);
[y,w,subbandfreq] = beamformer(rx);
% 绘制原始信号和波束成形信号的实部
figure;
plot(t(1:300),real(rx(1:300,6)),'r:',t(1:300),real(y(1:300)))
xlabel('Time')
ylabel('Amplitude')
legend('Original','Beamformed')
% 绘制五个频段的响应模式
figure;
pattern(antenna,subbandfreq(1:5).',[-180:180],0,'PropagationSpeed',c,...
    'CoordinateSystem','rectangular','Weights',w(:,1:5))
legend('Location','SouthEast')