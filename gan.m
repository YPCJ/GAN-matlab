clc, clear
addpath(genpath('gan')); rng('shuffle')

real = load('mnist.mat');
real = real.train_x;
real = double(real)/255;
noise = randn(size(real,1),128);

% GPU use if possible
num_gpu = 1;
gpu_try

opts.isPretrain         = 0;            % Pretraining opt. (0 new model, 2 previous model load)
opts.checker            = 10;           % During training, plot generated images for checker batch
opts.batch_size         = 128;
opts.max_epoch          = 100;
opts.learner            = 'adam';       % 'adam', 'rms', 'ada_grad', 'agd' optimizer 
opts.learner_scale      = 0.0002;       % Scale for optimizer
opts.save_model_path    = 'model/';     % model path to store the every epoch model
opts.initial            = 'he';         % initialization method 'xavier','he','random'
opt_config                              % etc option. (layer, units, cost function, ...)

[model_D, model_G, perf_D, perf_G] = funcTrain(noise, real, opts);