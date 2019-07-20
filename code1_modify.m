clear all
close all

[cancerInputs, cancerTargets] = cancer_dataset;
x = cancerInputs;
t = cancerTargets;
nodes = [2 8 32];
epochs = [4 8 16 32 64];

% original test sets
avgs = zeros(length(nodes), length(epochs));
sigs = zeros(length(nodes), length(epochs));

% Recalculate training sets
trainNetwork = struct;
trainNetwork.avgs = zeros(length(nodes), length(epochs));
trainNetwork.sigs = zeros(length(nodes), length(epochs));

% Recaluculate test sets
testNetwork = struct;
testNetwork.avgs = zeros(length(nodes), length(epochs));
testNetwork.sigs = zeros(length(nodes), length(epochs));

for kk = 1:1:length(nodes)

    for k = 1:1:length(epochs)
        percentErrors = zeros(1, 30);
        trainNetwork.errors = zeros(1, 30);
        testNetwork.errors = zeros(1, 30);

        % Train the Network
        % 30 times
        for i = 1:1:30
        trainFcn = 'trainscg'; % Scaled conjugate gradient backpropagation.
        hiddenLayerSize = nodes(kk);
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

            [net,tr] = train(net, x, t);
            % Test the Network
            y = net(x);
            e = gsubtract(t,y);
            performance = perform(net,t,y);
            tind = vec2ind(t);
            yind = vec2ind(y);
            percentError = sum(tind ~= yind)/numel(tind);
            % percentErrors(i) = percentError;
            percentErrors(1, i) = percentError;

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
            trainNetwork.errors(1, i) = trainNetwork.percentErrors;

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
            testNetwork.errors(1, i) = testNetwork.percentErrors;

        end

        % avg : average / standard deviation
        avgs(kk, k) = sum(percentErrors(1, :)) / 30;
        sigs(kk, k) = sqrt(sum(((percentErrors(1, :) - avgs(kk, k)).^2))/30);

        trainNetwork.avgs(kk, k) = sum(trainNetwork.errors(1,:)) / 30;
        trainNetwork.sigs(kk, k) = sqrt(sum(((trainNetwork.errors(1, :) -trainNetwork.avgs(kk, k)).^2)) / 30);

        testNetwork.avgs(kk, k) = sum(testNetwork.errors(1,:)) / 30;
        testNetwork.sigs(kk, k) = sqrt(sum(((testNetwork.errors(1, :) - testNetwork.avgs(kk,k)).^2)) / 30);
    end

end

15

figure(1) 
plot(epochs, trainNetwork.avgs(1, :), 'r-s'); 
hold on
plot(epochs, trainNetwork.avgs(2, :), 'g-s'); 
hold on
plot(epochs, trainNetwork.avgs(3, :), 'b-s'); 
hold on
plot(epochs, testNetwork.avgs(1, :), 'r--s'); 
hold on
plot(epochs, testNetwork.avgs(2, :), 'g--s'); 
hold on
plot(epochs, testNetwork.avgs(3, :), 'b--s'); 
hold on
title('Error rates with  epoches')
xlabel('epochs')
ylabel('error rate');
legend('train node=2','train node=8','train node=32','test node=2','test node=8','test node=32')
hold off

figure(2) 
plot(epochs, trainNetwork.sigs(1, :), 'r-s'); 
hold on
plot(epochs, trainNetwork.sigs(2, :), 'g-s'); 
hold on
plot(epochs, trainNetwork.sigs(3, :), 'b-s'); 
hold on
plot(epochs, testNetwork.sigs(1, :), 'r--s'); 
hold on
plot(epochs, testNetwork.sigs(2, :), 'g--s'); 
hold on
plot(epochs, testNetwork.sigs(3, :), 'b--s'); 
hold on
% title('Error rate VS Epoches')
title('Error rates with  epoches')
xlabel('epochs')
ylabel('error rate');
legend('train node=2','train node=8','train node=32','test node=2','test node=8','test node=32')
hold off
% figure(1)
% plot(epochs, trainNetwork.avgs(:,1), 'r', epochs, trainNetwork.avgs(:, 2), 'b', epochs,
% trainNetwork.avgs(:, 3), 'g');
% legend('Performance of 2 Node', 'Performance of 8 node', 'Performance of 32 Node');
% title('Error rates using training sets');
% xlabel('Epochs');
% ylabel('Mean Squared Error (MSE)');

% figure(2)
% plot(epochs, trainNetwork.sigs(:,1), 'r', epochs, trainNetwork.sigs(:, 2), 'b', epochs,
% trainNetwork.sigs(:, 3), 'g');
% legend('Performance of 2 Node', 'Performance of 8 node', 'Performance of 32 Node');
% title('Standard deviation using training sets');
% xlabel('Epochs');
% ylabel('Standard deviation');

% figure(3)
% % subplot(2, 1, 1);
% plot(epochs, testNetwork.avgs(:, 1), 'r', epochs, testNetwork.avgs(:, 2), 'b', epochs,
% testNetwork.avgs(:, 3), 'g');
% legend('Performance of 2 Node', 'Performance of 8 node', 'Performance of 32 Node');
% title('Error rates using test sets');
% xlabel('Epochs');
% ylabel('Mean Squared Error (MSE)');

% figure(4)
% plot(epochs, testNetwork.sigs(:,1), 'r', epochs, testNetwork.sigs(:, 2), 'b', epochs,
% testNetwork.sigs(:, 3), 'g');
% legend('Performance of 2 Node', 'Performance of 8 node', 'Performance of 32 Node');
% title('Standard deviation using test sets');
% xlabel('Epochs');
% ylabel('Standard deviation');