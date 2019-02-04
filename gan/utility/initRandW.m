function W = initRandW(nhid,nvis,initMethod)

    switch initMethod
        case {'he','He'}
            W = randn(nhid, nvis,'single') * sqrt(2/nvis);
        case {'Xavier','xavier'}
            W = randn(nhid, nvis,'single') * sqrt(2/(nhid+nvis));
        otherwise
            r  = 6 / sqrt(nhid+nvis+1);  
            W  = rand(nhid, nvis,'single') * 8 * r - 4*r;
    end