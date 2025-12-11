%{
为空气中的 11 个元件声学阵列创建 GSC 波束形成器。
线性调频信号以 -50 方位角和 0 仰角入射到阵列上。
将 GSC 波束成形信号与 Frost 波束成形信号进行比较。信号传播速度为340 m/s，采样率为8 kHz。
创建麦克风和阵列系统对象。阵列元件间距为二分之一波长。将信号频率设置为奈奎斯特频率的二分之一。
%}
c = 340.0;
fs = 8.0e3;     % 采样率 (Hz)
fc = fs/2;      % 信号频率设置为奈奎斯特频率一半 (4 kHz)
lam = c/fc;
transducer = phased.OmnidirectionalMicrophoneElement('FrequencyRange',[20 20000]);
array = phased.ULA('Element',transducer,'NumElements',11,'ElementSpacing',lam/2);
% 模拟带宽为 500 Hz 的线性调频信号。
t = 0:1/fs:.5;
signal = chirp(t,0,0.5,500);
% 创建到达阵列的入射波。将高斯噪声添加到波中。
collector = phased.WidebandCollector('Sensor',array,'PropagationSpeed',c,...
    'SampleRate',fs,'ModulatedInput',false,'NumSubbands',512);
incidentAngle = [-50;0];
signal = collector(signal.',incidentAngle);
noise = 0.5*randn(size(signal));
recsignal = signal + noise;
% 在实际入射角处执行Frost 波束成形。
frostbeamformer = phased.FrostBeamformer('SensorArray',array,'PropagationSpeed',...
    c,'SampleRate',fs,'Direction',incidentAngle,'FilterLength',15);
yfrost = frostbeamformer(recsignal);
% 执行 GSC 波束成形并将波束形成器输出与 Frost 波束形成器输出进行绘制。
% 还绘制到达阵列中间元件的非波束成形信号。
gscbeamformer = phased.GSCBeamformer('SensorArray',array,...
    'PropagationSpeed',c,'SampleRate',fs,'Direction',incidentAngle,...
    'FilterLength',15);
ygsc = gscbeamformer(recsignal);
plot(t*100,recsignal(:,6),t*100,yfrost,t,ygsc)
xlabel('Time (ms)')
ylabel('Amplitude')
% 放大输出的一小部分。
idx = 1000:1300;
plot(t(idx)*1000,recsignal(idx,6),t(idx)*1000,yfrost(idx),t(idx)*1000,ygsc(idx))
xlabel('Time (ms)')
legend('Received signal', 'Frost beamformed signal', 'GSC beamformed signal')