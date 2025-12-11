% 将延时 LCMV 波束形成器应用于 11 元件声学 ULA 阵列。这些元件是全向麦克风。
% 信号的入射角方位角为-50度，仰角为30度。入射信号是带宽为 500 Hz 的 FM 线性调频。
% 传播速度是空气中的典型声速，340 m/s。
nElem = 11;
microphone = phased.OmnidirectionalMicrophoneElement(...
    'FrequencyRange',[20 20000]);
array = phased.ULA('Element',microphone,'NumElements',nElem,'ElementSpacing',0.04);
fs = 8000;
t = 0:1/fs:0.3;
x = chirp(t,0,1,500);
c = 340;
collector = phased.WidebandCollector('Sensor',array,...
    'PropagationSpeed',c,'SampleRate',fs,...
    'ModulatedInput',false);
incidentAngle = [-50;30];
x = collector(x.',incidentAngle);
noise = 0.2*randn(size(x));
rx = x + noise;
filterLength = 5;
constraintMatrix = kron(eye(filterLength),ones(nElem,1));
desiredResponseVector = eye(filterLength,1);
beamformer = phased.TimeDelayLCMVBeamformer('SensorArray',array,...
    'PropagationSpeed',c,'SampleRate',fs,'FilterLength',filterLength,...
    'Direction',incidentAngle,'Constraint',constraintMatrix,...
    'DesiredResponse',desiredResponseVector);
y = beamformer(rx);
plot(t,rx(:,6),'r:',t,y)
xlabel('Time')
ylabel('Amplitude')
legend('Original','Beamformed')