classdef MTEA_D_NN < Algorithm
    % <Multi-task> <Multi-objective> <None>

    %------------------------------- Reference --------------------------------
    % @Article{Wang2023MTEA-D-DN,
    %   title    = {Multiobjective Multitask Optimization - Neighborhood as a Bridge for Knowledge Transfer},
    %   author   = {Wang, Xianpeng and Dong, Zhiming and Tang, Lixin and Zhang, Qingfu},
    %   journal  = {IEEE Transactions on Evolutionary Computation},
    %   year     = {2023},
    %   number   = {1},
    %   pages    = {155-169},
    %   volume   = {27},
    %   doi      = {10.1109/TEVC.2022.3154416},
    % }
    %--------------------------------------------------------------------------

    %------------------------------- Copyright --------------------------------
    % Copyright (c) Yanchi Li. You are free to use the MToP for research
    % purposes. All publications which use this platform should acknowledge
    % the use of "MToP" or "MTO-Platform" and cite as "Y. Li, W. Gong, F. Ming,
    % T. Zhang, S. Li, and Q. Gu, MToP: A MATLAB Optimization Platform for
    % Evolutionary Multitasking, 2023, arXiv:2312.08134"
    %--------------------------------------------------------------------------

    properties (SetAccess = private)
        Beta = 0.4
        F = 0.5
        CR = 0.5
        MuM = 15
    end

    methods
        function Parameter = getParameter(Algo)
            Parameter = {'Beta: Probability of choosing parents locally', num2str(Algo.Beta), ...
                'F:Mutation Factor', num2str(Algo.F), ...
                'CR: Crossover Rate', num2str(Algo.CR), ...
                'MuM: Polynomial Mutation', num2str(Algo.MuM)};
        end

        
        function Algo = setParameter(Algo, Parameter)
            i = 1;
            Algo.Beta = str2double(Parameter{i}); i = i + 1;
            Algo.F = str2double(Parameter{i}); i = i + 1;
            Algo.CR = str2double(Parameter{i}); i = i + 1;
            Algo.MuM = str2double(Parameter{i}); i = i + 1;
        end

        function run(Algo, Prob)
            % Initialize
            
            for t = 1:Prob.T
                % Generate the weight vectors
                [W{t}, N{t}] = UniformPoint(Prob.N, Prob.M(t));
                DT{t} = ceil(N{t} / 20);

                % Detect the neighbours of each solution
                B{t} = pdist2(W{t}, W{t});
                [~, B{t}] = sort(B{t}, 2);
                B{t} = B{t}(:, 1:DT{t});
                population{t} = Initialization_One(Algo, Prob, t, Individual, N{t});

                Z{t} = min(population{t}.Objs, [], 1);
                if N{t} < Prob.N % Fill population
                    population{t}(N{t} + 1:Prob.N) = population{t}(1:Prob.N - N{t});
                end
            end

            gen = 0;
            NN_G = 50;

            ntask = Prob.T;
            dims = Prob.T(1);
            fit_list = cell(1,ntask);
            pop_list = cell(1,ntask);
            PopObj = cell(1,ntask);

            net = cell(ntask,1);
            for i =1:ntask
                net{i} = cell(1,1);% 初始化神经网络
            end
            

            while Algo.notTerminated(Prob, population)
                % Generation
                disp(['fes:', num2str(Algo.FE), '--------',' gen:', num2str(gen),'--------'])
                gen = gen + 1;
                kt_g = 5+5*(gen<=300);
                %% train neural network
                if mod(gen,NN_G)==1
                    
                    for i =1:ntask
                        fit_list{i}= calculate_fitness(population,i);
                        pop_list{i}=[];
                        PopObj{i} = [];
                        pop_len = length(population{i});
                        for j=1:pop_len
                            pop_list{i} = [pop_list{i};population{i}(j).Dec];
                            PopObj{i} = [PopObj{i};population{i}(j).Obj];
                        end
                       
                    end

                    % for i =1:ntask
                    %      fit_list{i}= calfitness(PopObj{i});
                    % end
                    for i = 1:ntask
                        for j = i+1:ntask
                            if length(fit_list{i}) < length(fit_list{j})
                                all_length{i} = length(fit_list{i});
                                all_length{j} = length(fit_list{i});
                            else
                                all_length{i} = length(fit_list{j});
                                all_length{j} = length(fit_list{j});
                            end
                            [fit_i,sort_index] = sort(fit_list{i},'ascend');
                            pop_i = pop_list{i}(sort_index,:);
                            fmax_i = fit_i(end);
                            fmin_i = fit_i(1);
                            [fit_j,sort_index] = sort(fit_list{j},'ascend');
                            pop_j = pop_list{j}(sort_index,:);
                            fmax_j = fit_j(end);
                            fmin_j = fit_j(1);

                            %j->i的神经网络学习
                            normalized_fit_i = [];
                            normalized_fit_j = [];
                            for k = 1:all_length{i}
                                normalized_fit_i(k) = (fit_i(k) - fmin_i)/(fmax_i-fmin_i+1E-10);
                            end
                            for k = 1:all_length{j}
                                normalized_fit_j(k) = (fit_j(k) - fmin_j)/(fmax_j-fmin_j+1E-10);
                            end

                            p_index = argmin_fun_2(normalized_fit_i, normalized_fit_j);
                            output = pop_i(1:all_length{i},:);
                            input = pop_j(p_index,:);
                            input = input';
                            output = output';

                            
                            input_d = input(:,1:ceil(N{i}*0.2));
                            output_d = output(:,1:ceil(N{i}*0.2));
                            hiddenLayerSize = 33;
                            net{i}{1} = feedforwardnet(hiddenLayerSize);
                            net{i}{1}.trainParam.showWindow = false;
                            net{i}{1} = train(net{i}{1},input_d,output_d);


                            %i->j的神经网络学习
                            p_index = argmin_fun_2(normalized_fit_j, normalized_fit_i);
                            output = pop_j(1:all_length{j},:);
                            input = pop_i(p_index,:);

                            input = input';
                            input_d = input(:,1:ceil(N{j}*0.2));
                            output = output';
                            output_d = output(:,1:ceil(N{j}*0.2));
                            
                            hiddenLayerSize = 33;
                            net{j}{1} = feedforwardnet(hiddenLayerSize);
                            net{j}{1}.trainParam.showWindow = false;
                            net{j}{1} = train(net{i}{1},input_d,output_d);
                        end

                    end
                end

                if gen>=1 &&mod(gen,kt_g)==1
                    for t=1:Prob.T
                        for i = 1:N{t}
                            

                            source_t = randi(ntask);
                            while(source_t == t)
                                source_t = randi(ntask);
                            end
                            inorder = randperm(N{t});
                            source_pop = population{source_t}(inorder(1));
                            source_pop_Dec = source_pop.Dec;

                            if rand()<0.5
                                
                                predict = net{t}{1}(source_pop_Dec');

                                predict(predict>1)=1;
                                predict(predict<0)=0;
                                Predict = population{t}(i);
                                Predict.Dec = predict';
                            else
                                Predict = source_pop;
                            end

                            if rand()<Algo.Beta
                                rndpm = randperm(length(B{t}(i,:)));
                                P = B{t}(i, :);
                                P1 = P(rndpm(1 + (i == rndpm(1))));

                                parent = [population{t}(i), population{t}(P1),Predict];
                                offspring = Algo.Generation(parent);
                                offspring = Algo.Evaluation(offspring, Prob, t);
                                Z{t} = min(Z{t}, offspring.Obj);
                                g_old = max(abs(population{t}(P).Objs - repmat(Z{t}, length(P), 1)) .* W{t}(P, :), [], 2);
                                g_new = max(repmat(abs(offspring.Obj - Z{t}), length(P), 1) .* W{t}(P, :), [], 2);
                                population{t}(P(g_old >= g_new)) = offspring;
                            else
                                P = randperm(N{t});
                                P1 = P(1+(i==P(1)));
                                parent = [population{t}(i), population{t}(P1),Predict];
                                offspring = Algo.Generation(parent);
                                offspring = Algo.Evaluation(offspring, Prob, t);
                                Z{t} = min(Z{t}, offspring.Obj);
                                g_old = max(abs(population{t}(P).Objs - repmat(Z{t}, length(P), 1)) .* W{t}(P, :), [], 2);
                                g_new = max(repmat(abs(offspring.Obj - Z{t}), length(P), 1) .* W{t}(P, :), [], 2);
                                population{t}(P(g_old >= g_new)) = offspring;
                            end
                        end
                    end
                else
                    for t = 1:Prob.T
                        for i = 1:N{t}
                            % Choose the parents
                            if rand()<Algo.Beta
                                rndpm = randperm(length(B{t}(i,:)));
                                P = B{t}(i, :);
                                ind = 1 + (i == rndpm(1));
                                P1 = P(rndpm(ind));
                                P2 = P(rndpm(ind+1 + (i == ind)));
                                parent = [population{t}(i), population{t}(P1),population{t}(P2)];
                                offspring = Algo.Generation(parent);
                                offspring = Algo.Evaluation(offspring, Prob, t);
                                Z{t} = min(Z{t}, offspring.Obj);
                                g_old = max(abs(population{t}(P).Objs - repmat(Z{t}, length(P), 1)) .* W{t}(P, :), [], 2);
                                g_new = max(repmat(abs(offspring.Obj - Z{t}), length(P), 1) .* W{t}(P, :), [], 2);
                                population{t}(P(g_old >= g_new)) = offspring;
                            else
                                P = randperm(N{t});
                                ind = 1+(i==P(1));
                                P1 = P(ind);
                                P2 = P(ind+1+(ind==i));

                                parent = [population{t}(i), population{t}(P1),population{t}(P2)];
                                offspring = Algo.Generation(parent);
                                offspring = Algo.Evaluation(offspring, Prob, t);
                                Z{t} = min(Z{t}, offspring.Obj);
                                g_old = max(abs(population{t}(P).Objs - repmat(Z{t}, length(P), 1)) .* W{t}(P, :), [], 2);
                                g_new = max(repmat(abs(offspring.Obj - Z{t}), length(P), 1) .* W{t}(P, :), [], 2);
                                population{t}(P(g_old >= g_new)) = offspring;
                            end
                        end
                    end


                    if N{t} < Prob.N % Fill population
                        population{t}(N{t} + 1:Prob.N) = population{t}(1:Prob.N - N{t});
                    end
                end
            end
        end

        function offspring = Generation(Algo, population)
            offspring = population(1);

            offspring.Dec = population(1).Dec + Algo.F * (population(2).Dec - population(3).Dec);
            offspring.Dec = DE_Crossover(offspring.Dec, population(1).Dec, Algo.CR);
            offspring.Dec = GA_Mutation(offspring.Dec, Algo.MuM);

            offspring.Dec(offspring.Dec > 1) = 1;
            offspring.Dec(offspring.Dec < 0) = 0;
        end
    end
end
