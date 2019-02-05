opts.epoch = 1;
if strcmp(opts.learner,'sgd')
    opts.sgd_learn_rate = linspace(0.0001, 0.001, opts.max_epoch); % linearly decreasing lrate for plain sgd
end

opts.initial_momentum = 0;
opts.final_momentum = 0;
opts.change_momentum_point = 0;

opts.isDropout = 0; % need dropout regularization?
opts.isDropoutInput = 0; % dropout inputs?
opts.drop_ratio = 0; % ratio of units to drop

opts.cost_function = 'gan';
if strcmp(opts.cost_function,'lsgan')
    opts.lsgan_a = 1;
    opts.lsgan_b = 0;
    opts.lsgan_c = 1;
end

opts.hid_struct_D = [128 ];
opts.hid_struct_G = [128 ];
opts.net_struct_G = [size(noise,2), opts.hid_struct_G, size(real, 2)];
opts.net_struct_D = [opts.net_struct_G(end), opts.hid_struct_D, 1];

opts.batchNormlization = 0;
opts.batchNorm_G = [1 0];
opts.batchNorm_D = [0 0];

opts.unit_type_output_D = 'sigm';
opts.unit_type_output_G = 'lin';
opts.unit_type_hidden_D = 'relu';
opts.unit_type_hidden_G = 'relu';

if strcmp(opts.cost_function,'gan')
    opts.unit_type_output_D = 'sigm';
elseif strcmp(opts.cost_function,'lsgan')
    opts.unit_type_output_D = 'lin';
end

if opts.batchNormlization && (numel(opts.net_struct_G)-1 ~= numel(opts.batchNorm_G))
    error('Batch norm check needed.')
end
if opts.batchNormlization && (numel(opts.net_struct_D)-1 ~= numel(opts.batchNorm_D))
    error('Batch norm check needed.')
end