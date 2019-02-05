function [cost, GD_gradients, G_fake, D_fake, D_real, mu, istd] = computeNetGradient(net_G, net_D, noise, real, opts, GD)
    num_sample = size(noise,1);

    num_net_layer_G = length(net_G);
    num_net_layer_D = length(net_D);
    unit_type_output_G = opts.unit_type_output_G;
    unit_type_output_D = opts.unit_type_output_D;
    unit_type_hidden_G = opts.unit_type_hidden_G;
    unit_type_hidden_D = opts.unit_type_hidden_D;

    [forward_path_G_fake, drop_mask_G_fake, forward_path_G_fake_batch, mu, forward_path_G_fake_istd] = forwardPass(net_G, noise, opts, 'G');
    G_fake = forward_path_G_fake{num_net_layer_G+1}';

    D_real = [];
    if strcmp(GD,'D')
        [forward_path_D, drop_mask_D, forward_path_D_batch, mu, forward_path_D_istd] = forwardPass(net_D, [real;G_fake], opts ,'D');
        [forward_path_D_real, forward_path_D_fake] = real_fake(forward_path_D);
        D_real = forward_path_D_real{num_net_layer_D+1}';
        D_fake = forward_path_D_fake{num_net_layer_D+1}';
        
        [drop_mask_D_real, drop_mask_D_fake] = real_fake(drop_mask_D);
        [forward_path_D_real_batch, forward_path_D_fake_batch] = real_fake(forward_path_D_batch);
        
        switch opts.cost_function
            case 'gan'
                cost = -mean(log(D_real) + log(1-D_fake));
                if strcmp(unit_type_output_D,'sigm')
                    output_delta_real = -(1-D_real);
                    output_delta_fake = D_fake;
                else
                    output_delta_real = -(1-D_real)./(D_real.*(1-D_real)).*compute_unit_gradient(D_real,unit_type_output_D);
                    output_delta_fake = D_fake./(D_fake.*(1-D_fake)).*compute_unit_gradient(D_fake,unit_type_output_D);
                end
            case 'lsgan'
                cost = 0.5*sum(sum((D_real-opts.lsgan_a).^2 + (D_fake-opts.lsgan_b).^2))/num_sample;
                output_delta_real = +(D_real - opts.lsgan_a).*compute_unit_gradient(D_real,unit_type_output_D);
                output_delta_fake = +(D_fake - opts.lsgan_b).*compute_unit_gradient(D_fake,unit_type_output_D);
        end
    else
        [forward_path_D_fake, drop_mask_D_fake, forward_path_D_fake_batch, ~, forward_path_D_fake_istd] = forwardPass(net_D, G_fake, opts, 'D');
        D_fake = forward_path_D_fake{num_net_layer_D+1}';
        switch opts.cost_function
            case 'gan'
                cost = -mean(log(D_fake));
                if strcmp(unit_type_output_D,'sigm')
                    output_delta_fake = (D_fake-1);
                else
                    output_delta_fake = (D_fake-1)./(D_fake.*(1-D_fake)).*compute_unit_gradient(D_fake,unit_type_output_D);
                end
            case 'lsgan'
                cost = 0.5*sum(sum((D_fake-opts.lsgan_c).^2))/num_sample;
                output_delta_fake = (D_fake - opts.lsgan_c).*compute_unit_gradient(D_fake,unit_type_output_D);
        end
    end

    if strcmp(GD,'D')
        acf = struct('real',forward_path_D_real,'fake',forward_path_D_fake);
        x_hat = struct('real',forward_path_D_real_batch,'fake',forward_path_D_fake_batch);
        x_istd = forward_path_D_istd;
        
        dmask = struct('real',drop_mask_D_real,'fake',drop_mask_D_fake);
        fns = fieldnames(acf);
        
        net_gradients_D = zeroInitNet(opts.net_struct_D, opts.isGPU, 0, opts.batchNormlization, opts.batchNorm_D);
        GD_gradients = struct('real',net_gradients_D,'fake',net_gradients_D);
        GD_output_deltas = struct('real',output_delta_real,'fake',output_delta_fake);
    else
        acf = struct('fake',forward_path_D_fake);
        x_hat = struct('fake',forward_path_D_fake_batch);
        x_istd = forward_path_D_fake_istd;
        
        dmask = struct('fake',drop_mask_D_fake);
        fns = fieldnames(acf);
        
        net_gradients_G = zeroInitNet(opts.net_struct_G,opts.isGPU, 0, opts.batchNormlization, opts.batchNorm_G);
        GD_gradients = struct('fake',net_gradients_G);
        GD_output_deltas = struct('fake',output_delta_fake);
    end

    net = net_D;
    
    for k = 1:length(fns)
        forward_path = {acf.(fns{k})};
        drop_mask =  {dmask.(fns{k})};
        upper_layer_delta = GD_output_deltas.(fns{k}); % upper layer delta doesn't need {};
        
        h = {x_hat.(fns{k})};
        istd = x_istd;
        for ll = num_net_layer_D: -1: 1
            if opts.batchNormlization && opts.batchNorm_D(ll) == 1
                [upper_layer_delta, net_gradients_D(ll).gamma, net_gradients_D(ll).beta] =  batchNorm_backpass(h{ll}, istd{ll}, upper_layer_delta, net(ll).gamma);
            end
            net_gradients_D(ll).W = (forward_path{ll}*upper_layer_delta)'/num_sample; % Bishop (5.53) = (5.54)*(5.49)
            net_gradients_D(ll).b = mean(upper_layer_delta)';
            
            if ll == 1
                upper_layer_delta = ((upper_layer_delta*net(ll).W)'.*drop_mask{ll}.*compute_unit_gradient(forward_path{ll},unit_type_output_G))';
            else
                upper_layer_delta = ((upper_layer_delta*net(ll).W)'.*drop_mask{ll}.*compute_unit_gradient(forward_path{ll},unit_type_hidden_D))'; %Bishop (5.56)
            end
        end
        
        GD_gradients.(fns{k}) = net_gradients_D;
        
        if strcmp(GD,'G')
           net = net_G;
           forward_path = forward_path_G_fake;
           drop_mask = drop_mask_G_fake;
           h = forward_path_G_fake_batch;
           istd = forward_path_G_fake_istd;

           for ll = num_net_layer_G: -1: 1
               if opts.batchNormlization && opts.batchNorm_G(ll) == 1
                    [upper_layer_delta, net_gradients_G(ll).gamma, net_gradients_G(ll).beta] =  batchNorm_backpass(h{ll}, istd{ll}, upper_layer_delta, net(ll).gamma);
               end
               
               net_gradients_G(ll).W = (forward_path{ll}*upper_layer_delta)'/num_sample; % Bishop (5.53) = (5.54)*(5.49)
               net_gradients_G(ll).b = mean(upper_layer_delta)';
               
               upper_layer_delta = ((upper_layer_delta*net(ll).W)'.*drop_mask{ll}.*compute_unit_gradient(forward_path{ll},unit_type_hidden_G))'; %Bishop (5.56)
           end
           
           GD_gradients = net_gradients_G;
       end
    end
end
