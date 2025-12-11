# Signal-Generator
# 该信号发生器是MATLAB的一个mex函数。它可用于生成混响环境中移动声源和接收器的响应。用户能够在每个离散时间点指定声源和接收器的位置。
# 生成的输出信号是通过将（无回声的）源信号与随时间变化的房间脉冲响应进行卷积计算得到的。
# 可以指定多个接收器位置，以同时生成多个响应。

The signal generator is a mex-function for MATLAB that can be used to generate the response of a moving sound source and receiver in a reverberant environment. 
The user can specify the position of the source and the receiver at each discrete time instance. 
The output signal is computed by convolving the (anechoic) source signal with the time-varying room impulse response. 
Multiple receiver positions can be specified to generate multiple responses simultaneously. 
The room impulse responses are generated using the image method, proposed by Allen and Berkley in 1979 [1]. 
The user can control the reverberation time (or reflection coefficients), reflection order, room dimension and microphone directivity in a way similar to the RIR generator. 

This package includes a MATLAB example, the mex-function, and the source code of the mex-function.

More information can be found [here](https://www.audiolabs-erlangen.de/fau/professor/habets/software/signal-generator).

[1] J.B. Allen and D.A. Berkley, "Image method for efficiently simulating small-room acoustics," Journal Acoustic Society of America, 65(4), April 1979, p 943.
