 %clear all;
 %clc;
 %close all;

data = load('labels_final.mat');

%%
unzip vehicleDatasetImages.zip
data = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;

% Add the full path to the local vehicle data folder.
vehicleDataset.imageFilename = fullfile(pwd,vehicleDataset.imageFilename);
%%
DisplasiaDataset = data.gTruth.LabelData; 

% Add the full path to the local vehicle data folder.
%directory = 'com_displasia\';
DisplasiaDataset.filename = data.gTruth.DataSource.Source;

%Training set
rng(0);
shuffledIndices = randperm(height(DisplasiaDataset));
idx = floor(0.6 * length(shuffledIndices));
trainingDataTbl = DisplasiaDataset(shuffledIndices(1:idx), :);
testDataTbl = DisplasiaDataset(shuffledIndices(idx+1:end), :);

%Imagem Datastore
imdsTrain = imageDatastore(trainingDataTbl.filename);
imdsTest = imageDatastore(testDataTbl.filename);

%Datastore para box labels
bldsTrain = boxLabelDatastore(trainingDataTbl(:, 2:end));
bldsTest = boxLabelDatastore(testDataTbl(:, 2:end));


% Este excerto causa erro no data aug
  miniBatchSize = 8;
    imdsTrain.ReadSize = miniBatchSize;
  bldsTrain.ReadSize = miniBatchSize;
% --------------------------------------

trainingData = combine(imdsTrain, bldsTrain);
testData = combine(imdsTest, bldsTest);

%Data Augmentation
augmentedTrainingData = transform(trainingData,@augmentData);

% Visualize the augmented images.
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1,1}, 'Rectangle', data{1,2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData, 'BorderSize', 10)

%---------------------------------
%Preprocess training data--------------
%--------------------------------
    
networkInputSize = [227 227 3];
preprocessedTrainingData = transform(augmentedTrainingData, @(data)preprocessData(data, networkInputSize)); %erro aqui
data = read(preprocessedTrainingData);

I = data{1,1};
bbox = data{1,2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%Definir YOLO network

trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,networkInputSize));
numAnchors = 6;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors)

area = anchorBoxes(:, 1).*anchorBoxes(:, 2);
[~, idx] = sort(area, 'descend');
anchorBoxes = anchorBoxes(idx, :);
anchorBoxMasks = {[1,2,3]
    [4,5,6]
    };

baseNetwork = squeezenet;
lgraph = squeezenetFeatureExtractor(baseNetwork, networkInputSize);

classNames = trainingDataTbl.Properties.VariableNames(1:2);
numClasses = size(classNames, 2);
numPredictorsPerAnchor = 5 + numClasses;
lgraph = addFirstDetectionHead(lgraph, anchorBoxMasks{1}, numPredictorsPerAnchor);
lgraph = addSecondDetectionHead(lgraph, anchorBoxMasks{2}, numPredictorsPerAnchor);

lgraph = connectLayers(lgraph, 'fire9-concat', 'conv1Detection1');
lgraph = connectLayers(lgraph, 'relu1Detection1', 'upsample1Detection2');
lgraph = connectLayers(lgraph, 'fire5-concat', 'depthConcat1Detection2/in2');

networkOutputs = ["conv2Detection1"
    "conv2Detection2"
    ];

%Especificar opcoes de treino
numIterations = 140;
learningRate = 0.001; %era 0.001
warmupPeriod = 2000; 
l2Regularization = 0.0005; %Adam?
penaltyThreshold = 0.5;
velocity = [];

executionEnvironment = "gpu";

    % Convert layer graph to dlnetwork.
    net = dlnetwork(lgraph);
    
    % Create subplots for the learning rate and mini-batch loss.
    fig = figure;
    [lossPlotter, learningRatePlotter] = configureTrainingProgressPlotter(fig);
    
    % Custom training loop.
    for iteration = 1:numIterations
        
        % Reset datastore.
        if ~hasdata(preprocessedTrainingData)
            reset(preprocessedTrainingData);
        end
        
        % Read batch of data and create batch of images and
        % ground truths.
        data = read(preprocessedTrainingData);
        [XTrain,YTrain] = createBatchData(data, classNames);
        
        % Convert mini-batch of data to dlarray.
        XTrain = dlarray(single(XTrain),'SSCB');
        
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            XTrain = gpuArray(XTrain);
        end
        
        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients function.
        [gradients,loss,state] = dlfeval(@modelGradients, net, XTrain, YTrain, anchorBoxes, anchorBoxMasks, penaltyThreshold, networkOutputs);
        
        % Apply L2 regularization.
        gradients = dlupdate(@(g,w) g + l2Regularization*w, gradients, net.Learnables);
        
        % Determine the current learning rate value.
        currentLR = piecewiseLearningRateWithWarmup(iteration, learningRate, warmupPeriod, numIterations);
        
        % Update the network learnable parameters using the SGDM optimizer.
        [net, velocity] = sgdmupdate(net, gradients, velocity, currentLR);
        
        % Update the state parameters of dlnetwork.
        net.State = state;
        
        % Update training plot with new points.
        addpoints(lossPlotter, iteration, double(gather(extractdata(loss))));
        addpoints(learningRatePlotter, iteration, currentLR);
        drawnow;  
    end
%%
%Evaluate Model
confidenceThreshold = 0.5;
overlapThreshold = 0.5;

% Create the test datastore.
preprocessedTestData = transform(testData,@(data)preprocessData(data,networkInputSize));

% Create a table to hold the bounding boxes, scores, and labels returned by
% the detector. 
numImages = size(testDataTbl,1);
results = table('Size',[numImages 3],...
    'VariableTypes',{'cell','cell','cell'},...
    'VariableNames',{'Boxes','Scores','Labels'});

% Run detector on each image in the test set and collect results.
for i = 1:numImages
    
    % Read the datastore and get the image.
    data = read(preprocessedTestData);
    I = data{1};
    
    % Convert to dlarray. If GPU is available, then convert data to gpuArray.
    XTest = dlarray(I,'SSCB');
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        XTest = gpuArray(XTest);
    end
    
    % Run the detector.
    [bboxes, scores, labels] = yolov3Detect(net, XTest, networkOutputs, anchorBoxes, anchorBoxMasks, confidenceThreshold, overlapThreshold, classNames);
    
    % Collect the results.
    results.Boxes{i} = bboxes;
    results.Scores{i} = scores;
    results.Labels{i} = labels;
end

% Evaluate the object detector using Average Precision metric.
[ap, recall, precision] = evaluateDetectionPrecision(results, preprocessedTestData);

% Plot precision-recall curve.
figure
plot(recall, precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))

%%
%Predição nas imagens correspondentes
  if ~hasdata(preprocessedTrainingData)
            reset(preprocessedTrainingData);
        end

for i=1:numel(data)
II = data{i};  %data{i,1};
bbox_I = results.Boxes{i,1};
label = results.Scores{i,1};
ident = results.Labels{i,1};
%PredictedImage = insertShape(II,'Rectangle',bbox_I);
PredictedImage = insertObjectAnnotation(II, 'rectangle',bbox_I, label);
PredictedImage = imresize(PredictedImage,2);
figure
imshow(PredictedImage)
pause
end

%%
%Calculo da matriz confusao

[XTest,YTest] = digitTest4DArrayData;
YPredicted = classify(net,XTest);

%plotconfusion(results, preprocessedTestData)

%%
%Average Miss Rate
[am, fppi, missRate] = evaluateDetectionMissRate(results, preprocessedTestData);
% figure;
% loglog(fppi{1,1}, missRate{1,1});
% grid on
% title(sprintf('Log Average Miss Rate = %.1f', am))

figure(2);
loglog(fppi{2,1}, missRate{2,1});
xlabel('False Positives (per image)')
ylabel('Miss Rate')
grid on
title(sprintf('Log Average Miss Rate = %.1f', am(2,1)))
%%
%[am, fppi, missRate] = evaluateDetectionMissRate(results, preprocessedTestData);
%[ap, recall, precision] = evaluateDetectionPrecision(results, preprocessedTestData);

recall_score = mean(recall{2})
precision_score = mean(precision{2})

%%
%Extrair True Positives(TP) atraves dos dados obtidos pelas funcs matlab
TP= [];

tp_1 = (precision{1} .* fppi{1})./(1-precision{1});
tp_2 = (precision{2} .* fppi{2})./(1-precision{2});

TP = [TP;tp_1];
TP = [TP;tp_2];

%%
%F1- Score

F1_score_1 = 2*((precision{1}.*recall{1})/(precision{1}+recall{1}))
F1_score_2 = 2*((precision{2}.*recall{2})/(precision{2}+recall{2}))
F1_score_mean = mean(F1_score_1 (:,30))
F2_score_mean = mean(F1_score_2 (:,37))


%%
%Extrair False Negatives (FN) atraves dos dados obtidos

FN = [];
fn_1 = ((tp_1 - recall{1}.*tp_1) ./recall{1})
fn_2 = ((tp_2 - recall{2}.*tp_2) ./recall{2})

FN = [FN;fn_1];
FN = [FN;fn_2];

%%
%Plot grafico recall/precision

figure
plot(recall{2}, precision{2})
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap(2,1)))



%%
%Detectar objectos a usar o YOLO
% Read the datastore.
reset(preprocessedTestData)
data = read(preprocessedTestData);

% Get the image.
I = data{1};

% Convert to dlarray.
XTest = dlarray(I,'SSCB');

% If GPU is available, then convert data to gpuArray.
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    XTest = gpuArray(XTest);
end

[bboxes, scores, labels] = yolov3Detect(net, XTest, networkOutputs, anchorBoxes, anchorBoxMasks, confidenceThreshold, overlapThreshold, classNames);

% Display the detections on image.
if ~isempty(scores)
    I = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
end
figure
imshow(I)

%-------------------------------------------------------------------
%FUNCOES SUPORTE-------------------------------------------------
%-------------------------------------------------------------------

function data = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.

data = cell(size(A));
for ii = 1:size(A,1)
    I = A{ii,1};
    bboxes = A{ii,2};
    labels = A{ii,3};
    sz = size(I);
    if numel(sz)==3 && sz(3) == 3
        I = jitterColorHSV(I,...
            'Contrast',0.0,...
            'Hue',0.1,...
            'Saturation',0.2,...
            'Brightness',0.2);
    end
    
    % Randomly flip image.
    tform = randomAffine2d('XReflection',true,'Scale',[1 1.1]);
    rout = affineOutputView(sz,tform,'BoundsStyle','centerOutput');
    I = imwarp(I,tform,'OutputView',rout);
    
    % Apply same transform to boxes.
    [bboxes,indices] = bboxwarp(bboxes,tform,rout,'OverlapThreshold',0.25);
    labels = labels(indices);
    
    % Return original data only when all boxes are removed by warping.
    if isempty(indices)
        data = A(ii,:);
    else
        data(ii,:) = {I, bboxes, labels};
    end
end
end

function data = preprocessData(data, targetSize)
% Resize the images and scale the pixels to between 0 and 1. Also scale the
% corresponding bounding boxes.

for ii = 1:size(data,1)
    I = data{ii,1};
    imgSize = size(I);
    
    % Convert an input image with single channel to 3 channels.
    if numel(imgSize) == 1 
        I = repmat(I,1,1,3);
    end
    bboxes = data{ii,2};
    I = im2single(imresize(I,targetSize(1:2)));
    scale = targetSize(1:2)./imgSize(1:2);
    bboxes = bboxresize(bboxes,scale);
    data(ii,1:2) = {I, bboxes};
end
end

function [x,y] = createBatchData(data, classNames)
% The createBatchData function creates a batch of images and ground truths
% from input data, which is a [Nx3] cell array returned by the transformed
% datastore for YOLO v3. It creates two 4-D arrays by concatenating all the
% images and ground truth boxes along the batch dimension. The function
% performs these operations on the bounding boxes before concatenating
% along the fourth dimension:
% * Convert the class names to numeric class IDs based on their position in
%   the class names.
% * Combine the ground truth boxes, class IDs and network input size into
%   single cell array.
% * Pad with zeros to make the number of ground truths consistent across
%   a mini-batch.

% Concatenate images along the batch dimension.
x = cat(4,data{:,1});

% Get class IDs from the class names.
groundTruthClasses = data(:,3);
classNames = repmat({categorical(classNames')},size(groundTruthClasses));
[~,classIndices] = cellfun(@(a,b)ismember(a,b),groundTruthClasses,classNames,'UniformOutput',false);

% Append the label indexes and training image size to scaled bounding boxes
% and create a single cell array of responses.
groundTruthBoxes = data(:,2);
combinedResponses = cellfun(@(bbox,classid)[bbox,classid],groundTruthBoxes,classIndices,'UniformOutput',false);
len = max( cellfun(@(x)size(x,1), combinedResponses ) );
paddedBBoxes = cellfun( @(v) padarray(v,[len-size(v,1),0],0,'post'), combinedResponses, 'UniformOutput',false);
y = cat(4,paddedBBoxes{:,1});
end

%Network create functions-----------------------
function lgraph = squeezenetFeatureExtractor(net, imageInputSize)
% The squeezenetFeatureExtractor function removes the layers after 'fire9-concat'
% in SqueezeNet and also removes any data normalization used by the image input layer.

% Convert to layerGraph.
lgraph = layerGraph(net);

lgraph = removeLayers(lgraph, {'drop9' 'conv10' 'relu_conv10' 'pool10' 'prob' 'ClassificationLayer_predictions'});
inputLayer = imageInputLayer(imageInputSize,'Normalization','none','Name','data');
lgraph = replaceLayer(lgraph,'data',inputLayer);
end

function lgraph = addFirstDetectionHead(lgraph,anchorBoxMasks,numPredictorsPerAnchor)
% The addFirstDetectionHead function adds the first detection head.

numAnchorsScale1 = size(anchorBoxMasks, 2);
% Compute the number of filters for last convolution layer.
numFilters = numAnchorsScale1*numPredictorsPerAnchor;
firstDetectionSubNetwork = [
    convolution2dLayer(3,256,'Padding','same','Name','conv1Detection1','WeightsInitializer','he')
    reluLayer('Name','relu1Detection1')
    convolution2dLayer(1,numFilters,'Padding','same','Name','conv2Detection1','WeightsInitializer','he')
    ];
lgraph = addLayers(lgraph,firstDetectionSubNetwork);
end

function lgraph = addSecondDetectionHead(lgraph,anchorBoxMasks,numPredictorsPerAnchor)
% The addSecondDetectionHead function adds the second detection head.

numAnchorsScale2 = size(anchorBoxMasks, 2);
% Compute the number of filters for the last convolution layer.
numFilters = numAnchorsScale2*numPredictorsPerAnchor;
secondDetectionSubNetwork = [
    upsampleLayer(2,'upsample1Detection2')
    depthConcatenationLayer(2,'Name','depthConcat1Detection2');
    convolution2dLayer(3,128,'Padding','same','Name','conv1Detection2','WeightsInitializer','he')
    reluLayer('Name','relu1Detection2')
    convolution2dLayer(1,numFilters,'Padding','same','Name','conv2Detection2','WeightsInitializer','he')
    ];
lgraph = addLayers(lgraph,secondDetectionSubNetwork);
end

%Learning rate schedule function --------------------------------------
function currentLR = piecewiseLearningRateWithWarmup(iteration, learningRate, warmupPeriod, numIterations)
% The piecewiseLearningRateWithWarmup function computes the current
% learning rate based on the iteration number.

if iteration <= warmupPeriod
    % Increase the learning rate for number of iterations in warmup period.
    currentLR = learningRate * ((iteration/warmupPeriod)^4);
    
elseif iteration >= warmupPeriod && iteration < warmupPeriod+floor(0.6*(numIterations-warmupPeriod))
    % After warm up period, keep the learning rate constant if the remaining number of iteration is less than 60 percent. 
    currentLR = learningRate;
    
elseif iteration >= warmupPeriod+floor(0.6*(numIterations-warmupPeriod)) && iteration < warmupPeriod+floor(0.9*(numIterations-warmupPeriod))
    % If the remaining number of iteration is more than 60 percent but less
    % than 90 percent multiply the learning rate by 0.1.
    currentLR = learningRate*0.1;
    
else
    % If remaining iteration is more than 90 percent multiply the learning
    % rate by 0.01.
    currentLR = learningRate*0.01;
end
end

%Predict functions----------------------------------------------
function [bboxes,scores,labels] = yolov3Detect(net, XTest, networkOutputs, anchors, anchorBoxMask, confidenceThreshold, overlapThreshold, classes)
% The yolov3Detect function detects the bounding boxes, scores, and labels in an image.

imageSize = size(XTest,[1,2]);

% Find the input image layer and get the network input size.
networkInputIdx = arrayfun( @(x)isa(x,'nnet.cnn.layer.ImageInputLayer'), net.Layers);
networkInputSize = net.Layers(networkInputIdx).InputSize;

% Predict and filter the detections based on confidence threshold.
predictions = yolov3Predict(net,XTest,networkOutputs,anchorBoxMask);
predictions = cellfun(@ gather, predictions,'UniformOutput',false);
predictions = cellfun(@ extractdata, predictions, 'UniformOutput', false);
tiledAnchors = generateTiledAnchors(predictions(:,2:5),anchors,anchorBoxMask);
predictions(:,2:5) = applyAnchorBoxOffsets(tiledAnchors, predictions(:,2:5), networkInputSize);
[bboxes,scores,labels] = generateYOLOv3Detections(predictions, confidenceThreshold, imageSize, classes);

% Apply suppression to the detections to filter out multiple overlapping
% detections.
if ~isempty(scores)
    [bboxes, scores, labels] = selectStrongestBboxMulticlass(bboxes, scores, labels ,...
        'RatioType', 'Union', 'OverlapThreshold', overlapThreshold);
end
end

function YPredCell = yolov3Predict(net,XTrain,networkOutputs,anchorBoxMask)
% Predict the output of network and extract the confidence, x, y,
% width, height, and class.
YPredictions = cell(size(networkOutputs));
[YPredictions{:}] = predict(net, XTrain);
YPredCell = extractPredictions(YPredictions, anchorBoxMask);

% Apply activation to the predicted cell array.
YPredCell = applyActivations(YPredCell);
end

%Model Gradient Functions------------------------------
function [gradients, totalLoss, state] = modelGradients(net, XTrain, YTrain, anchors, mask, penaltyThreshold, networkOutputs)
inputImageSize = size(XTrain,1:2);

% Extract the predictions from the network.
[YPredCell, state] = yolov3Forward(net,XTrain,networkOutputs,mask);

% Gather the activations in the CPU for post processing and extract dlarray data. 
gatheredPredictions = cellfun(@ gather, YPredCell(:,1:6),'UniformOutput',false); 
gatheredPredictions = cellfun(@ extractdata, gatheredPredictions, 'UniformOutput', false);

% Convert predictions from grid cell coordinates to box coordinates.
tiledAnchors = generateTiledAnchors(gatheredPredictions(:,2:5),anchors,mask);
gatheredPredictions(:,2:5) = applyAnchorBoxOffsets(tiledAnchors, gatheredPredictions(:,2:5), inputImageSize);

% Generate target for predictions from the ground truth data.
[boxTarget, objectnessTarget, classTarget, objectMaskTarget, boxErrorScale] = generateTargets(gatheredPredictions, YTrain, inputImageSize, anchors, mask, penaltyThreshold);

% Compute the loss.
boxLoss = bboxOffsetLoss(YPredCell(:,[2 3 7 8]),boxTarget,objectMaskTarget,boxErrorScale);
objLoss = objectnessLoss(YPredCell(:,1),objectnessTarget,objectMaskTarget);
clsLoss = classConfidenceLoss(YPredCell(:,6),classTarget,objectMaskTarget);
totalLoss = boxLoss + objLoss + clsLoss;

% Compute gradients of learnables with regard to loss.
gradients = dlgradient(totalLoss, net.Learnables);
end

function [YPredCell, state] = yolov3Forward(net, XTrain, networkOutputs, anchorBoxMask)
% Predict the output of network and extract the confidence score, x, y,
% width, height, and class.
YPredictions = cell(size(networkOutputs));
[YPredictions{:}, state] = forward(net, XTrain, 'Outputs', networkOutputs);
YPredCell = extractPredictions(YPredictions, anchorBoxMask);

% Append predicted width and height to the end as they are required
% for computing the loss.
YPredCell(:,7:8) = YPredCell(:,4:5);

% Apply sigmoid and exponential activation.
YPredCell(:,1:6) = applyActivations(YPredCell(:,1:6));
end

function boxLoss = bboxOffsetLoss(boxPredCell, boxDeltaTarget, boxMaskTarget, boxErrorScaleTarget)
% Mean squared error for bounding box position.
lossX = sum(cellfun(@(a,b,c,d) mse(a.*c.*d,b.*c.*d),boxPredCell(:,1),boxDeltaTarget(:,1),boxMaskTarget(:,1),boxErrorScaleTarget));
lossY = sum(cellfun(@(a,b,c,d) mse(a.*c.*d,b.*c.*d),boxPredCell(:,2),boxDeltaTarget(:,2),boxMaskTarget(:,1),boxErrorScaleTarget));
lossW = sum(cellfun(@(a,b,c,d) mse(a.*c.*d,b.*c.*d),boxPredCell(:,3),boxDeltaTarget(:,3),boxMaskTarget(:,1),boxErrorScaleTarget));
lossH = sum(cellfun(@(a,b,c,d) mse(a.*c.*d,b.*c.*d),boxPredCell(:,4),boxDeltaTarget(:,4),boxMaskTarget(:,1),boxErrorScaleTarget));
boxLoss = lossX+lossY+lossW+lossH;
end

function objLoss = objectnessLoss(objectnessPredCell, objectnessDeltaTarget, boxMaskTarget)
% Binary cross-entropy loss for objectness score.
objLoss = sum(cellfun(@(a,b,c) crossentropy(a.*c,b.*c,'TargetCategories','independent'),objectnessPredCell,objectnessDeltaTarget,boxMaskTarget(:,2)));
end

function clsLoss = classConfidenceLoss(classPredCell, classTarget, boxMaskTarget)
% Binary cross-entropy loss for class confidence score.
clsLoss = sum(cellfun(@(a,b,c) crossentropy(a.*c,b.*c,'TargetCategories','independent'),classPredCell,classTarget,boxMaskTarget(:,3)));
end

%Utility functions
function YPredCell = applyActivations(YPredCell)
YPredCell(:,1:3) = cellfun(@ sigmoid ,YPredCell(:,1:3),'UniformOutput',false);
YPredCell(:,4:5) = cellfun(@ exp,YPredCell(:,4:5),'UniformOutput',false);
YPredCell(:,6) = cellfun(@ sigmoid ,YPredCell(:,6),'UniformOutput',false);
end

function predictions = extractPredictions(YPredictions, anchorBoxMask)
predictions = cell(size(YPredictions, 1),6);
for ii = 1:size(YPredictions, 1)
    numAnchors = size(anchorBoxMask{ii},2);
    % Confidence scores.
    startIdx = 1;
    endIdx = numAnchors;
    predictions{ii,1} = YPredictions{ii}(:,:,startIdx:endIdx,:);
    
    % X positions.
    startIdx = startIdx + numAnchors;
    endIdx = endIdx+numAnchors;
    predictions{ii,2} = YPredictions{ii}(:,:,startIdx:endIdx,:);
    
    % Y positions.
    startIdx = startIdx + numAnchors;
    endIdx = endIdx+numAnchors;
    predictions{ii,3} = YPredictions{ii}(:,:,startIdx:endIdx,:);
    
    % Width.
    startIdx = startIdx + numAnchors;
    endIdx = endIdx+numAnchors;
    predictions{ii,4} = YPredictions{ii}(:,:,startIdx:endIdx,:);
    
    % Height.
    startIdx = startIdx + numAnchors;
    endIdx = endIdx+numAnchors;
    predictions{ii,5} = YPredictions{ii}(:,:,startIdx:endIdx,:);
    
    % Class probabilities.
    startIdx = startIdx + numAnchors;
    predictions{ii,6} = YPredictions{ii}(:,:,startIdx:end,:);
end
end

function tiledAnchors = generateTiledAnchors(YPredCell,anchorBoxes,anchorBoxMask)
% Generate tiled anchor offset.
tiledAnchors = cell(size(YPredCell));
for i=1:size(YPredCell,1)
    anchors = anchorBoxes(anchorBoxMask{i}, :);
    [h,w,~,n] = size(YPredCell{i,1});
    [tiledAnchors{i,2}, tiledAnchors{i,1}] = ndgrid(0:h-1,0:w-1,1:size(anchors,1),1:n);
    [~,~,tiledAnchors{i,3}] = ndgrid(0:h-1,0:w-1,anchors(:,2),1:n);
    [~,~,tiledAnchors{i,4}] = ndgrid(0:h-1,0:w-1,anchors(:,1),1:n);
end
end

function tiledAnchors = applyAnchorBoxOffsets(tiledAnchors,YPredCell,inputImageSize)
% Convert grid cell coordinates to box coordinates.
for i=1:size(YPredCell,1)
    [h,w,~,~] = size(YPredCell{i,1});  
    tiledAnchors{i,1} = (tiledAnchors{i,1}+YPredCell{i,1})./w;
    tiledAnchors{i,2} = (tiledAnchors{i,2}+YPredCell{i,2})./h;
    tiledAnchors{i,3} = (tiledAnchors{i,3}.*YPredCell{i,3})./inputImageSize(2);
    tiledAnchors{i,4} = (tiledAnchors{i,4}.*YPredCell{i,4})./inputImageSize(1);
end
end

function [lossPlotter, learningRatePlotter] = configureTrainingProgressPlotter(f)
% Create the subplots to display the loss and learning rate.
figure(f);
clf
subplot(2,1,1);
ylabel('Learning Rate');
xlabel('Iteration');
learningRatePlotter = animatedline;
subplot(2,1,2);
ylabel('Total Loss');
xlabel('Iteration');
lossPlotter = animatedline;
end
