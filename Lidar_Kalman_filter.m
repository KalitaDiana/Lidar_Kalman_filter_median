%% Moving object detection based on Kalman filter and median filter
%% input data 
m = [u, v];
M = [1,1,1];
S = 1;
R* = 0.5;
W = [0.5 0.5 1
    0 0.5 1
    0 0 1];
L = W*[r1 r2 t];
E = 1/W'*1/W;
r1 = 0.5;
r2 = 0.5;
t = 1;
x0 = 1;
P0 = [1 0 0
    0 1 0
    0 0 1];
A = [1.1269 -0.4940 0.1129
    1.0000 0 0
    0 1.0000 0];
B = [-0.3832
    0.5919
    0.5191];
C = [1 0 0];
D = 0;
H =
Ts = -1;
Q = 2.3; 
R = 1;
%% Calibration 
imagePath = fullfile(toolboxdir('lidar'), 'lidardata', 'lcc', 'HDL64', 'images');
ptCloudPath = fullfile(toolboxdir('lidar'), 'lidardata', 'lcc', 'HDL64', 'pointCloud');
cameraParamsPath = fullfile(imagePath, 'calibration.mat');

[tform, errors] = estimateLidarCameraTransform(lidarCheckerboardPlanes, ...
    imageCorners3d, 'CameraIntrinsic', intrinsic.cameraParams);

intrinsic = load(cameraParamsPath);
imds = imageDatastore(imagePath); 
pcds = fileDatastore(ptCloudPath, 'ReadFcn', @pcread); 

imageFileNames = imds.Files;
ptCloudFileNames = pcds.Files;

squareSize = 200; 
sys = ss(A,[B B],C,D,Ts,'InputName',{'u' 'w'},'OutputName','y', 'x*');
rng('default'); 

%% Calculation of median filtering values and Kalman filtering
x* = median(D)
[kalmf,L,~,Mx,Z] = kalman(sys,Q,R);
kalmf = kalmf(1,:);
sys.InputName = {'u','w'};
sys.OutputName = {'yt'};
vIn = sumblk('y=yt+v');
 
kalmf.InputName = {'u','y'};
kalmf.OutputName = 'ye';

SimModel = connect(sys,vIn,kalmf,{'u','w','v'}, {'yt','ye'});
t = (0:100)';
u = sin(t/5);
rng(10,'twister');
w = sqrt(Q)*randn(length(t),1); v = sqrt(R)*randn(length(t),1);
out = lsim(SimModel,[u,w,v]);
yt = lsim(sys,[u w]);  
y = yt + v; 
P = B*Q*B'; 
x = zeros(3,1); 

ye = zeros(length(t),1); 
ycov = zeros(length(t),1);  
errcov = zeros(length(t),1);  
Su = C*P*C';
N0 =1;
Ss0 = Su+R;
result = 1;
for i = 0:20
    F = 2 - Ss0;
    result = result*F;
    Ss0 = Ss0*F;
end
for i=1:length(t)  
 Mxn = P*C'*result; 
 x = x + Mxn*(y(i)-C*x); 
 P = (eye(3)-Mxn*C)*P; 
 ye(i) = C*x;  
 errcov(i) = C*P*C';  
 x = A*x + B*u(i); 
 P = A*P*A' + B*Q*B'; 
end
function PedestrianTrackingFromMovingCameraExample()

videoFile       = 'D:\car.mp4';
scaleDataFile   = 'pedScaleTable.mat'; 
obj = setupSystemObjects(videoFile, scaleDataFile);

detector = peopleDetectorACF('caltech');

tracks = initializeTracks(); 

nextId = 1; 
option.ROI                  = [180 370 700 140];  
option.scThresh             = 0.5;               
option.gatingThresh         = 0.3;             
option.gatingCost           = 100;             
option.costOfNonAssignment  = 10;               
option.timeWindowSize       = 16;               
option.confidenceThresh     = 2;                
option.ageThresh            = 8;                
option.visThresh            = 0.6;             
stopFrame = 1629;
for fNum = 1:stopFrame
    frame   = readFrame(obj.reader);
 
    [centroids, bboxes, scores] = detectPeople();
    
    predictNewLocationsOfTracks();    
    
    [assignments, unassignedTracks] = ...
        detectionToTrackAssignment();
    
    updateAssignedTracks();    
    updateUnassignedTracks();    
    deleteLostTracks();    
    createNewTracks();
    
    displayTrackingResults();

    if ~isOpen(obj.videoPlayer)
        break;
    end
end

%%
    function obj = setupSystemObjects(videoFile,scaleDataFile)
       
        obj.reader = VideoReader(videoFile);
       
        obj.videoPlayer = vision.VideoPlayer('Position', [29, 597, 643, 386]);                
                                       
        ld = load(scaleDataFile, 'pedScaleTable');
        obj.pedScaleTable = ld.pedScaleTable;
    end


%% Initialize 
    function tracks = initializeTracks()
        tracks = struct(...
            'id', {}, ...
            'color', {}, ...
            'bboxes', {}, ...
            'scores', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'confidence', {}, ...            
            'predPosition', {});
    end

%% Detect 
    function [centroids, bboxes, scores] = detectPeople()
        
        resizeRatio = 1.5;
        frame = imresize(frame, resizeRatio, 'Antialiasing',false);
        
        [bboxes, scores] = detect(detector, frame, option.ROI, ...
            'WindowStride', 2,...
            'NumScaleLevels', 4, ...
            'SelectStrongest', false);
       
        height = bboxes(:, 4) / resizeRatio;
        y = (bboxes(:,2)-1) / resizeRatio + 1;        
        yfoot = min(length(obj.pedScaleTable), round(y + height));
        estHeight = obj.pedScaleTable(yfoot); 
        invalid = abs(estHeight-height)>estHeight*option.scThresh;        
        bboxes(invalid, :) = [];
        scores(invalid, :) = [];

        [bboxes, scores] = selectStrongestBbox(bboxes, scores, ...
                            'RatioType', 'Min', 'OverlapThreshold', 0.6);                               
        
        if isempty(bboxes)
            centroids = [];
        else
            centroids = [(bboxes(:, 1) + bboxes(:, 3) / 2), ...
                (bboxes(:, 2) + bboxes(:, 4) / 2)];
        end
    end

%% Predict New Locations 

    function predictNewLocationsOfTracks()
        for i = 1:length(tracks)

            bbox = tracks(i).bboxes(end, :);
            
            predictedCentroid = predict(tracks(i).kalmanFilter);
            
            tracks(i).predPosition = [predictedCentroid - bbox(3:4)/2, bbox(3:4)];
        end
    end

%%

   function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()
        
        predBboxes = reshape([tracks(:).predPosition], 4, [])';
        cost = 1 - bboxOverlapRatio(predBboxes, bboxes);
        cost(cost > option.gatingThresh) = 1 + option.gatingCost;

        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, option.costOfNonAssignment);
    end

%% Update

    function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);

            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);
            
            correct(tracks(trackIdx).kalmanFilter, centroid);
             
            T = min(size(tracks(trackIdx).bboxes,1), 4);
            w = mean([tracks(trackIdx).bboxes(end-T+1:end, 3); bbox(3)]);
            h = mean([tracks(trackIdx).bboxes(end-T+1:end, 4); bbox(4)]);
            tracks(trackIdx).bboxes(end+1, :) = [centroid - [w, h]/2, w, h];
            
            tracks(trackIdx).age = tracks(trackIdx).age + 1;
            
            tracks(trackIdx).scores = [tracks(trackIdx).scores; scores(detectionIdx)];
           
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
           
            T = min(option.timeWindowSize, length(tracks(trackIdx).scores));
            score = tracks(trackIdx).scores(end-T+1:end);
            tracks(trackIdx).confidence = [max(score), mean(score)];
        end
    end

%% 
    function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            idx = unassignedTracks(i);
            tracks(idx).age = tracks(idx).age + 1;
            tracks(idx).bboxes = [tracks(idx).bboxes; tracks(idx).predPosition];
            tracks(idx).scores = [tracks(idx).scores; 0];
            T = min(option.timeWindowSize, length(tracks(idx).scores));
            score = tracks(idx).scores(end-T+1:end);
            tracks(idx).confidence = [max(score), mean(score)];
        end
    end

%% 

    function deleteLostTracks()
        if isempty(tracks)
            return;
        end        
     
        ages = [tracks(:).age]';
        totalVisibleCounts = [tracks(:).totalVisibleCount]';
        visibility = totalVisibleCounts ./ ages;
       
        confidence = reshape([tracks(:).confidence], 2, [])';
        maxConfidence = confidence(:, 1);
        lostInds = (ages <= option.ageThresh & visibility <= option.visThresh) | ...
             (maxConfidence <= option.confidenceThresh);
        tracks = tracks(~lostInds);
    end

%%
 
        for i = 1:size(unassignedBboxes, 1)            
            centroid = unassignedCentroids(i,:);
            bbox = unassignedBboxes(i, :);
            score = unassignedScores(i);
            
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [2, 1], [5, 5], 100);
          newTrack = struct(...
                'id', nextId, ...
                'color', 255*rand(1,3), ...
                'bboxes', bbox, ...
                'scores', score, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'confidence', [score, score], ...
                'predPosition', bbox);
            
            tracks(end + 1) = newTrack;
            
            nextId = nextId + 1;
        end
    end

%% 
    function displayTrackingResults()

        displayRatio = 4/3;
        frame = imresize(frame, displayRatio);
        
        if ~isempty(tracks)
            ages = [tracks(:).age]';        
            confidence = reshape([tracks(:).confidence], 2, [])';
            maxConfidence = confidence(:, 1);
            avgConfidence = confidence(:, 2);
            opacity = min(0.5,max(0.1,avgConfidence/3));
            noDispInds = (ages < option.ageThresh & maxConfidence < option.confidenceThresh) | ...
                       (ages < option.ageThresh / 2);
                   
            for i = 1:length(tracks)
                if ~noDispInds()
                    bb = tracks(i).bboxes(end, :);
                    bb(:,1:2) = (bb(:,1:2)-1)*displayRatio + 1;
                    bb(:,3:4) = bb(:,3:4) * displayRatio;
                    
                    
                    frame = insertShape(frame, ...
                                            'FilledRectangle', bb, ...
                                            'Color', tracks(i).color, ...
                                            'Opacity', opacity(i));
                    frame = insertObjectAnnotation(frame, ...
                                            'rectangle', bb, ...
                                            num2str(avgConfidence(i)), ...
                                            'Color', tracks(i).color);
                end
            end
        end
        
        frame = insertShape(frame, 'Rectangle', option.ROI * displayRatio, ...
                                'Color', [255, 0, 0], 'LineWidth', 3);
                            
        step(obj.videoPlayer, frame);
        
    end

%%
end
