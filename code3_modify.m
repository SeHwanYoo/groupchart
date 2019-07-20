clear all
close all

[cancerInputs, cancerTargets] = cancer_dataset;
x = cancerInputs;
t = cancerTargets;
nodes = [2 8 32];
epochs = [4 8 16 32 64];

% classifier from 3 to 25, only odd numbers
clfs = [3:2:25];

% original test sets
avgs = zeros(length(clfs), length(nodes), length(epochs));
sigs = zeros(length(clfs), length(nodes), length(epochs));

% Recalculate training sets
trainNetwork = struct;
trainNetwork.avgs = zeros(length(clfs), length(nodes), length(epochs));
trainNetwork.sigs = zeros(length(clfs), length(nodes), length(epochs));

% Recaluculate test sets
testNetwork = struct;
testNetwork.avgs = zeros(length(clfs), length(nodes), length(epochs));
testNetwork.sigs = zeros(length(clfs), length(nodes), length(epochs));

for cc = 1:1:length(clfs)
    clf = clfs(cc); % Classifier
    
    for kk = 1:1:length(nodes)

        for k = 1:1:length(epochs)
            % 30 times
            percentErrors = zeros(1, 30);
            trainNetwork.errors = zeros(1, 30);
            testNetwork.errors = zeros(1, 30);
            
            for i = 1:1:30
                % trainFcn = 'trainscg';
                trainFcn = 'trainrp'; 
                hiddenLayerSize = nodes(kk);
                % hiddenLayerSize = 10;
                net = patternnet(hiddenLayerSize, trainFcn);
                net.trainParam.epochs = epochs(k);
                net.input.processFcns = {'removeconstantrows','mapminmax'}; 
                % Setup Division of Data for Training, Testing (No validation data)
                net.divideFcn = 'dividerand'; % Divide data randomly
                net.divideMode = 'sample'; % Divide up every sample
                                           % train and test ratio 50% each
                net.divideParam.trainRatio = 50/100;
                net.divideParam.valRatio = 0/100;
                net.divideParam.testRatio = 50/100;
                net.performFcn = 'crossentropy'; % Cross-Entropy
                net.plotFcns = {'plotperform','plottrainstate','ploterrhist', 'plotconfusion','plotroc'};
                clfs_errors = zeros(1, clf); 
                clfs_train_errors = zeros(1, clf); 
                clfs_test_errors = zeros(1, clf); 
                
                for c = 1:clf
                    [net,tr] = train(net, x, t);
                    y = net(x);
                    e = gsubtract(t,y);
                    performance = perform(net,t,y);
                    tind = vec2ind(t);
                    yind = vec2ind(y);
                    percentError = sum(tind ~= yind)/numel(tind);
                    clfs_errors(c) = percentError; 
                    
                    % training recalculate
                    trainNetwork.targets = t .* tr.trainMask{1};
                    trainNetwork.targets = trainNetwork.targets(~isnan(trainNetwork.targets))';
                    trainNetwork.targets = reshape(trainNetwork.targets, [2, length(trainNetwork.targets)/2]);
                    trainNetwork.output = y.*tr.trainMask{1};
                    trainNetwork.output = trainNetwork.output(~isnan(trainNetwork.output))';
                    trainNetwork.output = reshape(trainNetwork.output, [2, length(trainNetwork.output)/2]);
                    trainNetwork.tind = vec2ind(trainNetwork.targets);
                    trainNetwork.yind = vec2ind(trainNetwork.output);
                    trainNetwork.percentErrors = sum(trainNetwork.tind ~= trainNetwork.yind)/numel(trainNetwork.tind);
                    clfs_train_errors(c) = trainNetwork.percentErrors;
                    
                    % test recalculate
                    testNetwork.targets = t .* tr.testMask{1};
                    testNetwork.targets = testNetwork.targets(~isnan(testNetwork.targets))';
                    testNetwork.targets = reshape(testNetwork.targets, [2, length(testNetwork.targets)/2]);
                    testNetwork.output = y.*tr.testMask{1};
                    testNetwork.output = testNetwork.output(~isnan(testNetwork.output))';
                    testNetwork.output = reshape(testNetwork.output, [2, length(testNetwork.output)/2]);
                    testNetwork.tind = vec2ind(testNetwork.targets);
                    testNetwork.yind = vec2ind(testNetwork.output);
                    testNetwork.percentErrors = sum(testNetwork.tind ~= testNetwork.yind)/numel(testNetwork.tind);
                    clfs_test_errors(c) = testNetwork.percentErrors;
                    
                end 
                
                rand_num = rand(1, clfs(cc)); 
                rand_sum = sum(rand_num); 
                
                clfs_errors = (rand_num / rand_sum) * clfs_errors(1, :)';
                clfs_train_errors = (rand_num / rand_sum) * clfs_train_errors(1, :)';
                clfs_test_errors = (rand_num / rand_sum) * clfs_test_errors(1, :)';
                
                % percentErrors(1, i)
                % clfs_errors'
                
                percentErrors(1, i) = clfs_errors'; 
                trainNetwork.errors(1, i) = clfs_train_errors'; 
                testNetwork.errors(1, i) = clfs_test_errors'; 
            end
            
            avgs(cc, kk, k) = sum(percentErrors(1, :)) / 30;
            sigs(cc, kk, k) = sqrt(sum(((percentErrors(1, :) - avgs(cc, kk, k)).^2))/30); 
            
            % trainErrors = [trainErrors trainnetwork.errors];
            trainNetwork.avgs(cc, kk, k) = sum(trainNetwork.errors(1,:)) / 30;
            trainNetwork.sigs(cc, kk, k) = sqrt(sum(((trainNetwork.errors(1, :) - trainNetwork.avgs(cc, kk, k)).^2)) / 30);
            
            % testErrors = [testErrors testNetwork.errors];
            testNetwork.avgs(cc, kk, k) = sum(testNetwork.errors(1,:)) / 30;
            testNetwork.sigs(cc, kk, k) = sqrt(sum(((testNetwork.errors(1, :) - testNetwork.avgs(cc, kk,k)).^2)) / 30);
        end
    
    end

end 


figure(1)
subplot(2, 1, 1);
plot(epochs, trainNetwork.avgs(1, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.avgs(1, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.avgs(1, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.avgs(1, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.avgs(1, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.avgs(1, 3, :), 'b--s')
hold on
title('Error rate with Epoches (3 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

subplot(2, 1, 2);
plot(epochs, trainNetwork.sigs(1, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.sigs(1, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.sigs(1, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.sigs(1, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.sigs(1, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.sigs(1, 3, :), 'b--s')
hold on
title('STD with Epoches (3 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

figure(2)
subplot(2, 1, 1);
plot(epochs, trainNetwork.avgs(2, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.avgs(2, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.avgs(2, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.avgs(2, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.avgs(2, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.avgs(2, 3, :), 'b--s')
hold on
title('Error rate with Epoches (5 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2 1','node=8','node=32')
hold off 

subplot(2, 1, 2);
plot(epochs, trainNetwork.sigs(2, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.sigs(2, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.sigs(2, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.sigs(2, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.sigs(2, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.sigs(2, 3, :), 'b--s')
hold on
title('STD with Epoches (5 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2 1','node=8','node=32')
hold off 

figure(3)
subplot(2, 1, 1);
plot(epochs, trainNetwork.avgs(3, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.avgs(3, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.avgs(3, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.avgs(3, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.avgs(3, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.avgs(3, 3, :), 'b--s')
hold on
title('Error rate with Epoches (7 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

subplot(2, 1, 2);
plot(epochs, trainNetwork.sigs(3, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.sigs(3, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.sigs(3, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.sigs(3, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.sigs(3, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.sigs(3, 3, :), 'b--s')
hold on
title('STD with Epoches (7 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

figure(4)
subplot(2, 1, 1);
plot(epochs, trainNetwork.avgs(4, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.avgs(4, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.avgs(4, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.avgs(4, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.avgs(4, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.avgs(4, 3, :), 'b--s')
hold on
title('Error rate with Epoches (9 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

subplot(2, 1, 2);
plot(epochs, trainNetwork.sigs(4, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.sigs(4, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.sigs(4, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.sigs(4, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.sigs(4, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.sigs(4, 3, :), 'b--s')
hold on
title('STD with Epoches (9 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

figure(5)
subplot(2, 1, 1);
plot(epochs, trainNetwork.avgs(5, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.avgs(5, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.avgs(5, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.avgs(5, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.avgs(5, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.avgs(5, 3, :), 'b--s')
hold on
title('Error rate with Epoches (11 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

subplot(2, 1, 2);
plot(epochs, trainNetwork.sigs(5, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.sigs(5, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.sigs(5, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.sigs(5, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.sigs(5, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.sigs(5, 3, :), 'b--s')
hold on
title('STD with Epoches (11 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

figure(6)
subplot(2, 1, 1);
plot(epochs, trainNetwork.avgs(6, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.avgs(6, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.avgs(6, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.avgs(6, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.avgs(6, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.avgs(6, 3, :), 'b--s')
hold on
title('Error rate with Epoches (13 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

subplot(2, 1, 2);
plot(epochs, trainNetwork.sigs(6, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.sigs(6, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.sigs(6, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.sigs(6, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.sigs(6, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.sigs(6, 3, :), 'b--s')
hold on
title('STD with Epoches (13 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

figure(7)
subplot(2, 1, 1);
plot(epochs, trainNetwork.avgs(7, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.avgs(7, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.avgs(7, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.avgs(7, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.avgs(7, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.avgs(7, 3, :), 'b--s')
hold on
title('Error rate with Epoches (15 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

subplot(2, 1, 2);
plot(epochs, trainNetwork.sigs(7, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.sigs(7, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.sigs(7, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.sigs(7, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.sigs(7, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.sigs(7, 3, :), 'b--s')
hold on
title('STD with Epoches (15 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

figure(8)
subplot(2, 1, 1);
plot(epochs, trainNetwork.avgs(8, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.avgs(8, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.avgs(8, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.avgs(8, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.avgs(8, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.avgs(8, 3, :), 'b--s')
hold on
title('Error rate with Epoches (17 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

subplot(2, 1, 2);
plot(epochs, trainNetwork.sigs(8, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.sigs(8, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.sigs(8, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.sigs(8, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.sigs(8, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.sigs(8, 3, :), 'b--s')
hold on
title('STD with Epoches (17 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

figure(9)
subplot(2, 1, 1);
plot(epochs, trainNetwork.avgs(9, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.avgs(9, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.avgs(9, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.avgs(9, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.avgs(9, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.avgs(9, 3, :), 'b--s')
hold on
title('Error rate with Epoches (19 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

subplot(2, 1, 2);
plot(epochs, trainNetwork.sigs(9, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.sigs(9, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.sigs(9, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.sigs(9, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.sigs(9, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.sigs(9, 3, :), 'b--s')
hold on
title('STD with Epoches (19 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

figure(10)
subplot(2, 1, 1);
plot(epochs, trainNetwork.avgs(10, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.avgs(10, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.avgs(10, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.avgs(10, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.avgs(10, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.avgs(10, 3,:), 'b--s')
hold on
title('Error rate with Epoches (21 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

subplot(2, 1, 2);
plot(epochs, trainNetwork.sigs(10, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.sigs(10, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.sigs(10, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.sigs(10, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.sigs(10, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.sigs(10, 3,:), 'b--s')
hold on
title('STD with Epoches (21 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

figure(11)
subplot(2, 1, 1);
plot(epochs, trainNetwork.avgs(11, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.avgs(11, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.avgs(11, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.avgs(11, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.avgs(11, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.avgs(11, 3, :), 'b--s')
hold on
title('Error rate with Epoches (23 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

subplot(2, 1, 2);
plot(epochs, trainNetwork.sigs(11, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.sigs(11, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.sigs(11, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.sigs(11, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.sigs(11, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.sigs(11, 3, :), 'b--s')
hold on
title('STD with Epoches (23 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

figure(12)
subplot(2, 1, 1);
plot(epochs, trainNetwork.avgs(12, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.avgs(12, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.avgs(12, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.avgs(12, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.avgs(12, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.avgs(12, 3, :), 'b--s')
hold on
title('Error rate with Epoches (25 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

subplot(2, 1, 2);
plot(epochs, trainNetwork.sigs(12, 1, :), 'r-s')
hold on
plot(epochs, trainNetwork.sigs(12, 2, :), 'g-s')
hold on
plot(epochs, trainNetwork.sigs(12, 3, :), 'b-s')
hold on
plot(epochs, testNetwork.sigs(12, 1, :), 'r--s')
hold on
plot(epochs, testNetwork.sigs(12, 2, :), 'g--s')
hold on
plot(epochs, testNetwork.sigs(12, 3, :), 'b--s')
hold on
title('STD with Epoches (25 Classifiers)') ;
xlabel('epochs');
ylabel('error rate');
legend('node=2','node=8','node=32','node=2','node=8','node=32')
hold off

% =====================compare===============%
% figure(13)
% plot(, trainNetwork.avgs(
% load('E1.mat');
% train_errors=[result(2,3,4),result1(2,3,4),result2(2,3,4),result3(2,3,4)];
% train_std=[result(5,3,4),result1(5,3,4),result2(5,3,4),result3(5,3,4)];
% test_errors=[result(3,3,4),result1(3,3,4),result2(3,3,4),result3(3,3,4)];
% test_std=[result(6,3,4),result1(6,3,4),result2(6,3,4),result3(6,3,4)];
% nic=[1,3,15,25];
% plot(nic,train_errors,'r-s');
% hold on
% plot(nic,test_errors,'b-s');
% hold on
% title('Individual Classifier Accuracy VS Ensenmble Accuracy') ;
% xlabel('Number of individual classifiers'),ylabel('Error rate');
% legend('Train error rate','Test error rate')
% hold off
% figure(15)
% plot(nic,train_std,'r-s');
% hold on
% plot(nic,test_std,'b-s');
% hold on
% title('Individual Classifier Accuracy VS Ensenmble Accuracy') ;
% xlabel('Number of individual classifiers'),ylabel('Std');
% legend('Train std','Test std')
% hold off