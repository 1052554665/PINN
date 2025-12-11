% 将延时波束形成器应用于 11 个单元的均匀线性声学阵列。信号的到达角方位角为-50度，仰角为30度。
% 到达的信号是具有 500 Hz 带宽的线性 FM 线性调频的 0.3 秒段。假设空气中的声速为 340.0 m/s。
microphone = phased.CustomMicrophoneElement(FrequencyVector=[20,20000],FrequencyResponse=[1,1]);
array = phased.ULA(Element=microphone,NumElements=11,ElementSpacing=0.04);
fs = 8000;
t = 0:1/fs:0.3;
x = chirp(t,0,1,500);
c = 340;
collector = phased.WidebandCollector(Sensor=array,...
    PropagationSpeed=c,SampleRate=fs,ModulatedInput=false);
incidentAngle = [-50;30];
x = collector(x.',incidentAngle);
sigma = 0.2;
noise = sigma*randn(size(x));
rx = x + noise;
% 使用延时波束形成器对入射信号进行波束成形
beamformer = phased.TimeDelayBeamformer(SensorArray=array,...
    SampleRate=fs, PropagationSpeed=c,...
    Direction=incidentAngle);
y = beamformer(rx);
% 将波束形成的信号与阵列中间传感器的入射信号进行绘制。
plot(t,rx(:,6),"r:",t,y)
xlabel("Time (sec)")
ylabel("Amplitude")
legend("Original","Beamformed")
