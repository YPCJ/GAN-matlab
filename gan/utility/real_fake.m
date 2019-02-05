function [real, fake] = real_fake(input)
    num_cell = length(input);
    real = cell(num_cell,1);
    fake = cell(num_cell,1);
    
    for i=1:num_cell
        tmp = cell2mat(input(i));
        real(i) = {tmp(:,1:end/2)};
        fake(i) = {tmp(:,end/2+1:end)};
    end
end