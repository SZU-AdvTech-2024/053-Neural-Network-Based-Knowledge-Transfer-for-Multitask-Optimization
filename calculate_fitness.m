function fit = calculate_fitness(population,k)
N = length(population{k}); % N=100
fit = ones(1,N)*inf;
for i = 1:N
    p = population{k}(i).Objs;
    for j = 1:N
        q = population{k}(j).Objs;
        if i ~= j
            tmpfit =  sum(max(0,p-q).^2);
        end
    end
    fit(i)=min(fit(i),sqrt(tmpfit));
end
end