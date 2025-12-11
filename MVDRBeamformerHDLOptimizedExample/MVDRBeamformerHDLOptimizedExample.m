%% Fixed-Point HDL-Optimized Minimum-Variance Distortionless-Response (MVDR) Beamformer
% This example shows how to implement a fixed-point HDL-optimized
% minimum-variance distortionless-response (MVDR) beamformer, also known as
% Capon's method [5] and robust adaptive beamforming (RAB) [6].
% For more information on beamformers, see 
% <docid:phased_ug.example-ex37574528 Conventional and Adaptive Beamformers>.

% Copyright 2020-2021 The MathWorks, Inc.

%% MVDR Objective
% The MVDR beamformer preserves the gain in the
% direction of arrival of a desired signal and attenuates interference from
% other directions  [1], [2].
%
% Given readings from a sensor array, such as the uniform linear array
% (ULA) in the following diagram, form data matrix $A$ from samples of the
% array, where $a(t)$ is an $n$-by-1 column vector of readings from the
% array sampled at time $t$, and $a(t)^H$ is one row of matrix $A$.  Many
% more samples are taken than there are elements in the array. This results in the number of rows in $A$ being much greater than the number of columns. An estimate of the covariance matrix is $A^HA$,
% where $A^H$ is the Hermitian or complex-conjugate transpose of $A$.
%
% <<../Uniform_Linear_Array.png>>
%
% Compute the MVDR beamformer response by solving the following equation
% for $x$, where $b$ is a steering vector pointing in the direction of the
% desired signal.
%
% $$(A^HA)x = b$$
%
% The MVDR weight vector $w$ is computed from $x$ and $b$ using the following
% equation, which normalizes $x$ to preserve the gain in the direction of
% arrival of the desired signal.
%
% $$w = \frac{x}{b^H x}$$
%
% The MVDR system response is the inner product between the MVDR weight
% vector $w$ and a current sample from the sensor array $a(t)$.
%
% $$y = w^H a(t)$$
%% HDL-Optimized MVDR
% The three equations in the previous section are implemented by the three
% main blocks in the following model. The rate changes give the matrix
% solve additional clock cycles to update before the next input sample.
% The number of clock cycles between a valid input and when the complex
% matrix solve block is ready is its input wordlength to allow time for
% CORDIC iterations, plus 9 cycles for internal delays.
%
load_system('MVDRBeamformerHDLOptimizedModel');
open_system('MVDRBeamformerHDLOptimizedModel/MVDR - HDL Optimized');
%%
% Instead of forming data matrix $A$ and computing the Cholesky
% factorization of covariance matrix $A^HA$, the upper-triangular matrix of
% the QR decomposition of $A$ is computed directly and updated as each data
% vector $a(t)$ streams in from the sensor array.  Because the data is
% updated indefinitely, a forgetting factor is applied after each
% factorization. To integrate with an equivalent of a matrix of $m$ rows,
% the forgetting factor $\alpha$ should be set to
%
% $$
% \alpha = \exp(-1/(2m)).
% $$
%
% This example simulates the equivalent of a matrix with $m=300$ rows, so
% the forgetting factor is set to 0.9983.
%
% The *Complex Partial-Systolic Matrix Solve Using Q-less QR
% Decomposition with Forgetting Factor* block is implemented using the method
% found in [3].  The upper-triangular matrix $R$ from the QR decomposition
% of $A$ is identical to the Cholesky factorization of $A^HA$ except the
% signs of values on the diagonal. Solving the matrix equation $(A^HA)x = b$ by computing
% the Cholesky factorization of $A^HA$ is not as efficient or as
% numerically sound as computing the QR decomposition of $A$ directly [4].
%
%% Run Model
% Open and simulate the model.
open_system('MVDRBeamformerHDLOptimizedModel');
sim('MVDRBeamformerHDLOptimizedModel');
%%
% The desired rectangular pulse appears when the noise sources are nulled.
% This example simulates with the same latency as the hardware, so you can
% see the signal settle over time as the simulation starts and when the
% directions are changed.
%%
% When the signal direction and steering angle are aligned as indicated by
% the blue and green lines, you can see that the beam pattern has a gain of
% 0 dB. The noise sources are nulled as indicated by the red lines.
%%
% As the model is simulating, you can adjust the signal direction, steering
% angle and noise directions by dragging the sliders, or by editing the
% constant values.
%% Set Parameters
% The parameters for the beamformer are set in the model workspace.  You
% can modify the parameters by editing and running the |setMVDRExampleModelWorkspace| function.
%
%% References
%
% [1] V. Behar et al. "Parameter Optimization of the adaptive MVDR QR-based
% beamformer for jamming and multipath supression in GPS/GLONASS
% receivers". In: Proc. 16th Saint Petersburg International Conference on
% Integrated Navigation systems. Saint Petersburg, Russia, May 2009, pp.
% 325--334.
%
% [2] Jack Capon. "High-resolution frequency-wavenumber spectrum analysis".
% In: vol. 57. 1969, pp. 1408--1418.
%
% [3] C.M. Rader. "VLSI Systolic Arrays for Adaptive Nulling". In: IEEE
% Signal Processing Magazine (July 1996), pp. 29--49.
%
% [4] Charles F. Van Loan. Introduction to Scientific Computing: A
% Matrix-Vector Approach Using Matlab. Second edition. Prentice-Hall,
% 2000. isbn: 0-13-949157-0.
%
% [5] J. Capon, "High-resolution frequency-wavenumber spectrum analysis,"
% in Proceedings of the IEEE, vol. 57, no. 8, pp. 1408-1418, Aug. 1969,
% doi: 10.1109/PROC.1969.7278.
%
% [6] H. Cox, R. Zeskind and M. Owen, "Robust adaptive beamforming," in
% IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 35,
% no. 10, pp. 1365-1376, October 1987, doi: 10.1109/TASSP.1987.1165054.

