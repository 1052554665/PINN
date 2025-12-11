function varargout = setMVDRExampleModelWorkspace(model)
    %setMVDRExampleModelWorkspace Set model workspace for MVDR example.
    %   setMVDRExampleModelWorkspace sets the model workspace
    %   for the MVDR example.
    %
    %   setMVDRExampleModelWorkspace(model) sets the model workspace for model.
    %   This is useful if you copy the example and modify it.
    %
    %   Examples:
    %      setMVDRExampleModelWorkspace
    %
    %      model = 'MVDRBeamformerHDLOptimizedModel';
    %      setMVDRExampleModelWorkspace(model)
    %
    %   See also phased.MVDRBeamformer.

    %   Copyright 2020-2021 The MathWorks, Inc.
    if nargin<1 || isempty(model)
        model = 'MVDRBeamformerHDLOptimizedModel';
    end
    load_system(model);
    m = 300; % Effective number of samples in correlation matrix
    nArrayElements = 10;
    % Magnitude of the pulse, which is the signal of interest.
    pulseMagnitude = 1;
    % Thermal noise power
    thermalNoisePower = 1/db2pow(50);
    % Variance of the two noise signals
    noise1Variance = 100;
    noise2Variance = 100;
    paramBeamformerI = beamformerParameters(nArrayElements);
    samplesPerFrame = paramBeamformerI.samplesPerFrame;
    paramBeamformerI.samplesPerFrame  = 1;

    T = mvdr_types('Fixed',m,nArrayElements,pulseMagnitude,thermalNoisePower,noise1Variance,noise2Variance);
    M = T.M;
    forgettingFactor = T.ForgettingFactor;
    numberOfCORDICIterations = fixed.emblib.internal.cordicRowRotations.getMaxNIters(T.Input);
    qlessQRCyclesToReady = numberOfCORDICIterations + 10;

    OutputType = fixed.extractNumericType(T.Output);

    setModelWorkspace(model,...
        paramBeamformerI,...
        samplesPerFrame,...
        T,...
        M,...
        nArrayElements,...
        forgettingFactor,...
        pulseMagnitude,...
        thermalNoisePower,...
        noise1Variance,...
        noise2Variance,...
        qlessQRCyclesToReady,...
        OutputType);

    if nargout > 0
        varargout = {model,...
            paramBeamformerI,...
            samplesPerFrame,...
            T,...
            M,...
            nArrayElements,...
            forgettingFactor,...
            thermalNoisePower,...
            noise1Variance,...
            noise2Variance,...
            qlessQRCyclesToReady,...
            OutputType};
    end
end
function T = mvdr_types(dt,m,nArrayElements,pulseMagnitude,thermalNoisePower,noise1Variance,noise2Variance)
    % alpha = 0.99867003; From Rader paper
    alpha = exp(-1/(2*m)); % Forgetting factor equivalent to m samples
    % m = 300, alpha = 0.998334721450939
    switch dt
        case {'double','single'}
            T.Input = cast(0,dt);
            T.Output = cast(0,dt);
            T.SteeringVector = cast(0,dt);
            T.ForgettingFactor = cast(alpha,dt);
        case {'Fixed','ScaledDouble'}
            % The variance of the sum of the noise signals is the sum of
            % their variances.
            noiseStandardDeviation = sqrt(noise1Variance+noise2Variance);
            % The rows of A are the signals from the antenna array.
            % Estimate the sensor array maximum magnitude as the pulse
            % magnitude plus 4 standard deviations above the mean of the
            % noise.
            max_abs_A = pulseMagnitude + 4*noiseStandardDeviation;
            % B is a steering vector made up of complex exponentials with
            % magnitude 1.
            max_abs_B = 1;
            precisionBits = 24;
            Prototypes = fixed.complexQlessQRMatrixSolveFixedpointTypes(...
                m,...
                nArrayElements,...
                max_abs_A,...
                max_abs_B,...
                precisionBits,...
                sqrt(thermalNoisePower));

            F = fimath('RoundingMethod', 'Floor', ...
                'OverflowAction', 'Wrap');
            % Block parameters can't be empty
            T.Input = cast(0,'like',Prototypes.A);
            % Block parameters can't be empty
            T.Output = cast(0,'like',Prototypes.X);

            % Elements of the steering vector are
            % complex exponentials with unity gain
            % exp(1i*theta)
            wordLength = Prototypes.A.WordLength;
            T.SteeringVector = fi(0,1,wordLength,wordLength-2,F,'DataType',dt);

            T.ForgettingFactor = fi(alpha,1,wordLength,wordLength-1);

        otherwise
            error(['Data type ',dt', is not supported'])
    end
    T.M = m;
end
function paramBeamformerI = beamformerParameters(nArrayElements)
    % Environment
    prop_speed = physconst('LightSpeed');   % Propagation speed
    fc = 100e6;             % Operating frequency
    lambda = prop_speed/fc; % Wavelength
    paramBeamformerI.propSpeed = prop_speed;
    paramBeamformerI.fc = fc;

    % Antenna
    paramBeamformerI.Antenna = phased.ULA('NumElements',nArrayElements,...
        'ElementSpacing',0.5*lambda);

    % Pulse
    fs  = 1000; %1khz
    paramBeamformerI.fs = fs;
    prf = 1/0.3;
    paramBeamformerI.prf = prf;
    paramBeamformerI.samplesPerFrame = fs/prf;

    % LCMV Constraint Matrix
    steeringvec = phased.SteeringVector('SensorArray',paramBeamformerI.Antenna);
    paramBeamformerI.cMatrix = steeringvec(fc,[43 45 47]);
end

function mdlWks = setModelWorkspace(model,varargin)
    % https://www.mathworks.com/help/simulink/ug/change-model-workspace-data.html
    load_system(model)
    mdlWks = get_param(model,'ModelWorkspace');
    for k = 2:nargin
        assignin(mdlWks,inputname(k),varargin{k-1});
    end
end