function [net_iterative, net_weights_inc, net_grad_ssqr] = learner(net_iterative, momentum, net_weights_inc, net_grad_ssqr, net_grad, opts, epoch, bid, num_batch, GD)

num_net_layer = length(net_iterative);
net_ada_eta = net_weights_inc;

if strcmp(GD,'G')
   batchNorm = opts.batchNorm_G;
else
   batchNorm = opts.batchNorm_D;
end

    for ll = 1:num_net_layer
        switch opts.learner
            case 'sgd'
                net_weights_inc(ll).W = momentum*net_weights_inc(ll).W + opts.sgd_learn_rate(epoch)*net_grad(ll).W;
                net_weights_inc(ll).b = momentum*net_weights_inc(ll).b + opts.sgd_learn_rate(epoch)*net_grad(ll).b;
                if opts.batchNormlization && batchNorm(ll) == 1
                    net_weights_inc(ll).gamma = momentum*net_weights_inc(ll).gamma + opts.sgd_learn_rate(epoch)*net_grad(ll).gamma;
                    net_weights_inc(ll).beta  = momentum*net_weights_inc(ll).beta  + opts.sgd_learn_rate(epoch)*net_grad(ll).beta;
                end
            case 'ada_sgd'
                net_grad_ssqr(ll).W = net_grad_ssqr(ll).W + (net_grad(ll).W).^2;
                net_grad_ssqr(ll).b = net_grad_ssqr(ll).b + (net_grad(ll).b).^2;
                
                net_ada_eta(ll).W = opts.learner_scale./sqrt(net_grad_ssqr(ll).W + 10^-8);
                net_ada_eta(ll).b = opts.learner_scale./sqrt(net_grad_ssqr(ll).b + 10^-8);
                
                net_weights_inc(ll).W = momentum*net_weights_inc(ll).W + net_ada_eta(ll).W.*net_grad(ll).W;
                net_weights_inc(ll).b = momentum*net_weights_inc(ll).b + net_ada_eta(ll).b.*net_grad(ll).b;
                
                if opts.batchNormlization && batchNorm(ll) == 1
                    net_grad_ssqr(ll).gamma = net_grad_ssqr(ll).gamma + (net_grad(ll).gamma).^2;
                    net_grad_ssqr(ll).beta  = net_grad_ssqr(ll).beta  + (net_grad(ll).beta).^2;
                    net_ada_eta(ll).gamma = opts.learner_scale./sqrt(net_grad_ssqr(ll).gamma + 10^-8);
                    net_ada_eta(ll).beta  = opts.learner_scale./sqrt(net_grad_ssqr(ll).beta  + 10^-8);
                    net_weights_inc(ll).gamma = momentum*net_weights_inc(ll).gamma + net_ada_eta(ll).gamma.*net_grad(ll).gamma;
                    net_weights_inc(ll).beta  = momentum*net_weights_inc(ll).beta  + net_ada_eta(ll).beta.*net_grad(ll).beta;
                end
            case 'ada_delta'
                gamma = 0.9;
                net_grad_ssqr(ll).W = gamma*net_grad_ssqr(ll).W + (1-gamma)*(net_grad(ll).W).^2;
                net_grad_ssqr(ll).b = gamma*net_grad_ssqr(ll).b + (1-gamma)*(net_grad(ll).b).^2;
                net_ada_eta(ll).W = sqrt((net_grad_ssqr(ll).W2 + 10^-8)./(net_grad_ssqr(ll).W + 10^-8));
                net_ada_eta(ll).b = sqrt((net_grad_ssqr(ll).b2 + 10^-8)./(net_grad_ssqr(ll).b + 10^-8));
                
                net_weights_inc(ll).W = momentum*net_weights_inc(ll).W + net_ada_eta(ll).W.*net_grad(ll).W;
                net_weights_inc(ll).b = momentum*net_weights_inc(ll).b + net_ada_eta(ll).b.*net_grad(ll).b;
                
                net_grad_ssqr(ll).W2 = gamma*net_grad_ssqr(ll).W2 + (1-gamma)*(net_grad(ll).W).^2;
                net_grad_ssqr(ll).b2 = gamma*net_grad_ssqr(ll).b2 + (1-gamma)*(net_grad(ll).b).^2;
                
                if opts.batchNormlization && batchNorm(ll) == 1
                    net_grad_ssqr(ll).gamma = gamma*net_grad_ssqr(ll).gamma + (1-gamma)*(net_grad(ll).gamma).^2;
                    net_grad_ssqr(ll).beta  = gamma*net_grad_ssqr(ll).beta  + (1-gamma)*(net_grad(ll).beta).^2;
                    net_ada_eta(ll).gamma = sqrt((net_grad_ssqr(ll).gamma2 + 10^-8)./(net_grad_ssqr(ll).gamma + 10^-8));
                    net_ada_eta(ll).beta  = sqrt((net_grad_ssqr(ll).beta2  + 10^-8)./(net_grad_ssqr(ll).beta  + 10^-8));
                    
                    net_weights_inc(ll).gamma = momentum*net_weights_inc(ll).gamma + net_ada_eta(ll).gamma.*net_grad(ll).gamma;
                    net_weights_inc(ll).beta  = momentum*net_weights_inc(ll).beta  + net_ada_eta(ll).b.*net_grad(ll).beta;
                    
                    net_grad_ssqr(ll).gamma2 = gamma*net_grad_ssqr(ll).gamma2 + (1-gamma)*(net_grad(ll).gamma).^2;
                    net_grad_ssqr(ll).beta2  = gamma*net_grad_ssqr(ll).beta2  + (1-gamma)*(net_grad(ll).beta).^2;
                end
            case 'rms'
                gamma = 0.99;
                net_grad_ssqr(ll).W = gamma*net_grad_ssqr(ll).W + (1-gamma)*(net_grad(ll).W).^2;
                net_grad_ssqr(ll).b = gamma*net_grad_ssqr(ll).b + (1-gamma)*(net_grad(ll).b).^2;

                net_ada_eta(ll).W = opts.learner_scale./sqrt(net_grad_ssqr(ll).W + 10^-8);
                net_ada_eta(ll).b = opts.learner_scale./sqrt(net_grad_ssqr(ll).b + 10^-8);

                net_weights_inc(ll).W = momentum*net_weights_inc(ll).W + net_ada_eta(ll).W.*net_grad(ll).W;
                net_weights_inc(ll).b = momentum*net_weights_inc(ll).b + net_ada_eta(ll).b.*net_grad(ll).b;
                
                if opts.batchNormlization && batchNorm(ll) == 1
                    net_grad_ssqr(ll).gamma = gamma*net_grad_ssqr(ll).gamma + (1-gamma)*(net_grad(ll).gamma).^2;
                    net_grad_ssqr(ll).beta  = gamma*net_grad_ssqr(ll).beta  + (1-gamma)*(net_grad(ll).beta).^2;

                    net_ada_eta(ll).gamma = opts.learner_scale./sqrt(net_grad_ssqr(ll).gamma + 10^-8);
                    net_ada_eta(ll).beta  = opts.learner_scale./sqrt(net_grad_ssqr(ll).beta  + 10^-8);

                    net_weights_inc(ll).gamma = momentum*net_weights_inc(ll).gamma + net_ada_eta(ll).gamma.*net_grad(ll).gamma;
                    net_weights_inc(ll).beta  = momentum*net_weights_inc(ll).beta  + net_ada_eta(ll).beta .*net_grad(ll).beta;
                end
            case 'adam'
                beta1 = 0.9; beta2 = 0.999;
                timestamp = (epoch-1)*num_batch + bid;
                
                net_grad_ssqr(ll).W = beta1*net_grad_ssqr(ll).W + (1-beta1)*(net_grad(ll).W); % m
                net_grad_ssqr(ll).b = beta1*net_grad_ssqr(ll).b + (1-beta1)*(net_grad(ll).b);
                net_grad_ssqr(ll).W2 = beta2*net_grad_ssqr(ll).W2 + (1-beta2)*(net_grad(ll).W).^2; % v
                net_grad_ssqr(ll).b2 = beta2*net_grad_ssqr(ll).b2 + (1-beta2)*(net_grad(ll).b).^2;
                net_weights_inc(ll).W = momentum*net_weights_inc(ll).W + (opts.learner_scale*net_grad_ssqr(ll).W/(1-beta1.^(timestamp)))./(sqrt(net_grad_ssqr(ll).W2/(1-beta2.^(timestamp)) + 10^-8));
                net_weights_inc(ll).b = momentum*net_weights_inc(ll).b + (opts.learner_scale*net_grad_ssqr(ll).b/(1-beta1.^(timestamp)))./(sqrt(net_grad_ssqr(ll).b2/(1-beta2.^(timestamp)) + 10^-8));
                
                if opts.batchNormlization && batchNorm(ll) == 1
                    net_grad_ssqr(ll).gamma  = beta1*net_grad_ssqr(ll).gamma  + (1-beta1)*(net_grad(ll).gamma); % m
                    net_grad_ssqr(ll).beta   = beta1*net_grad_ssqr(ll).beta   + (1-beta1)*(net_grad(ll).beta);
                    net_grad_ssqr(ll).gamma2 = beta2*net_grad_ssqr(ll).gamma2 + (1-beta2)*(net_grad(ll).gamma).^2; % v
                    net_grad_ssqr(ll).beta2  = beta2*net_grad_ssqr(ll).beta2  + (1-beta2)*(net_grad(ll).beta).^2;
                    net_weights_inc(ll).gamma = momentum*net_weights_inc(ll).gamma + (opts.learner_scale*net_grad_ssqr(ll).gamma/(1-beta1.^(timestamp)))./(sqrt(net_grad_ssqr(ll).gamma2/(1-beta2.^(timestamp)) + 10^-8));
                    net_weights_inc(ll).beta  = momentum*net_weights_inc(ll).beta  + (opts.learner_scale*net_grad_ssqr(ll).beta /(1-beta1.^(timestamp)))./(sqrt(net_grad_ssqr(ll).beta2 /(1-beta2.^(timestamp)) + 10^-8));
                end
            case 'ams'
                beta1 = 0.9; beta2 = 0.999;
                net_grad_ssqr(ll).W = beta1*net_grad_ssqr(ll).W + (1-beta1)*(net_grad(ll).W); % m
                net_grad_ssqr(ll).b = beta1*net_grad_ssqr(ll).b + (1-beta1)*(net_grad(ll).b);
                net_grad_ssqr(ll).W2 = beta2*net_grad_ssqr(ll).W2 + (1-beta2)*(net_grad(ll).W).^2; % v
                net_grad_ssqr(ll).b2 = beta2*net_grad_ssqr(ll).b2 + (1-beta2)*(net_grad(ll).b).^2;
                
                net_grad_ssqr(ll).W3 = max(net_grad_ssqr(ll).W3, net_grad_ssqr(ll).W2); %v
                net_grad_ssqr(ll).b3 = max(net_grad_ssqr(ll).b3, net_grad_ssqr(ll).b2);
                net_weights_inc(ll).W = momentum*net_weights_inc(ll).W + (opts.learner_scale*net_grad_ssqr(ll).W)./(sqrt(net_grad_ssqr(ll).W3) + 10^-8);
                net_weights_inc(ll).b = momentum*net_weights_inc(ll).b + (opts.learner_scale*net_grad_ssqr(ll).b)./(sqrt(net_grad_ssqr(ll).b3) + 10^-8);
                if opts.batchNormlization && batchNorm(ll) == 1
                    net_grad_ssqr(ll).gamma = beta1*net_grad_ssqr(ll).gamma + (1-beta1)*(net_grad(ll).gamma); % m
                    net_grad_ssqr(ll).beta  = beta1*net_grad_ssqr(ll).beta  + (1-beta1)*(net_grad(ll).beta);
                    net_grad_ssqr(ll).gamma2 = beta2*net_grad_ssqr(ll).gamma2 + (1-beta2)*(net_grad(ll).gamma).^2; % v
                    net_grad_ssqr(ll).beta2  = beta2*net_grad_ssqr(ll).beta2  + (1-beta2)*(net_grad(ll).beta).^2;
                    
                    net_grad_ssqr(ll).gamma3 = max(net_grad_ssqr(ll).gamma3, net_grad_ssqr(ll).gamma2); %v
                    net_grad_ssqr(ll).beta3  = max(net_grad_ssqr(ll).beta3, net_grad_ssqr(ll).beta2);
                    net_weights_inc(ll).gamma = momentum*net_weights_inc(ll).gamma + (opts.learner_scale*net_grad_ssqr(ll).gamma)./(sqrt(net_grad_ssqr(ll).gamma3) + 10^-8);
                    net_weights_inc(ll).beta  = momentum*net_weights_inc(ll).beta  + (opts.learner_scale*net_grad_ssqr(ll).beta)./(sqrt(net_grad_ssqr(ll).beta3) + 10^-8);
                end
            otherwise
                error(['opts.learner : ' opts.learner])
        end

        net_iterative(ll).W = net_iterative(ll).W - net_weights_inc(ll).W;
        net_iterative(ll).b = net_iterative(ll).b - net_weights_inc(ll).b;
        if opts.batchNormlization && batchNorm(ll) == 1
            net_iterative(ll).gamma = net_iterative(ll).gamma - net_weights_inc(ll).gamma;
            net_iterative(ll).beta  = net_iterative(ll).beta  - net_weights_inc(ll).beta;
        end
    end
end