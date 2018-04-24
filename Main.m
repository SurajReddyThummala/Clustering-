clear
clc
close all

% Read in data
data = dlmread('data1.csv');

% Make similarity matrix
simMat = 1-squareform(pdist(data));

% Set preferences (diagonals) to median of each row
for k = 1:size(simMat,1)
    simMat(k,k) = median(simMat(k));
end

% Initialize messages (availabilities and responsibilities)
dataSize = size(simMat,1); 
avails = zeros(dataSize,dataSize);
resps = zeros(dataSize,dataSize);

% Remove degeneracies from similarity matrix
simMat = simMat + 1e-12 * randn(dataSize,dataSize) * (max(simMat(:)) - min(simMat(:)));

% Set damping factor
lambda = 0.5;

% Number of iterations
numIters = 100;

for iter = 1:numIters
    
    % Compute responsibilities
    resps_old = resps;
    availsPlusSimMat = avails + simMat;
    [vals,exemplars] = max(availsPlusSimMat,[],2);
    
    for ii = 1:dataSize 
        availsPlusSimMat(ii,exemplars(ii)) = -realmax; 
    end
    
    vals2 = max(availsPlusSimMat,[],2);
    resps = simMat - repmat(vals,[1,dataSize]);
    
    for ii = 1:dataSize
        resps(ii,exemplars(ii)) = simMat(ii,exemplars(ii)) - vals2(ii);
    end
    
    % Dampen responsibilities
    resps = (1 - lambda) * resps + lambda * resps_old;

    % Compute availabilities
    avails_old = avails;
    maxResps = max(resps,0);
    
    for k = 1:dataSize 
        maxResps(k,k) = resps(k,k); 
    end
    
    avails = repmat(sum(maxResps,1),[dataSize,1]) - maxResps;
    diagAvails = diag(avails);
    avails = min(avails,0); 
    
    for k = 1:dataSize 
        avails(k,k) = diagAvails(k); 
    end
    
    % Dampen availabilities
    avails = (1 - lambda) * avails + lambda * avails_old;
end

% Finding exemplars
pseuds = resps + avails;
exemplars = find(diag(pseuds) > 0);

% Indices of exemplars
numExemps = length(exemplars);
[~, clusters] = max(simMat(:,exemplars),[],2);
clusters(exemplars) = 1:numExemps;
indices = exemplars(clusters);

% Printing results

disp(['Number of clusters: ',num2str(numExemps)]);

for ii = 1:numExemps
    
    disp('-----------------------------------------')
    
    disp(['Exemplar for cluster ' num2str(ii) ': ' num2str(exemplars(ii))])
    
    disp('Cluster members:')
    disp(find(indices == exemplars(ii)));
end

% Plot clusters
figure(1)
subplot(1,2,1)
plot(data(:,1),data(:,2),'x')
title('before clustering')
for ii = 1:numExemps
    subplot(1,2,2)
    hold on
    plot(data(indices == exemplars(ii),1),data(indices == exemplars(ii),2),'x')
end
title('after clustering')

figure(2)
% Points
for ii = 1:numExemps
   plot(data(indices == exemplars(ii),1),data(indices == exemplars(ii),2),'go','MarkerFaceColor', 'g')
   hold on
   plot(data(exemplars(ii),1),data(exemplars(ii),2),'ro','MarkerFaceColor', 'r')
end

% Arrows
axis(axis)
for ii = 1:numExemps
   x = data(indices == exemplars(ii),1);
   y = data(indices == exemplars(ii),2);
   for jj = 1:length(x)
       if x(jj) ~= data(exemplars(ii),1) || y(jj) ~= data(exemplars(ii),2)
           plot([x(jj),data(exemplars(ii),1)],[y(jj),data(exemplars(ii),2)],'k');
           
           % Make arrow head 1/3 of the way to point
           xpoint = x(jj) + (data(exemplars(ii),1) - x(jj))/2;
           ypoint = y(jj) + (data(exemplars(ii),2) - y(jj))/2;
           arrow([x(jj),y(jj)],[xpoint,ypoint],'Length',5,'TipAngle',20);
       end
   end
end
title('Points belonging to each cluster')