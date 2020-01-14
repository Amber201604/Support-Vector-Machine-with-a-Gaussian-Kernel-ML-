function a4q1
% A4Q1: function for CISC371, Fall 2019, Assignment #4, Question #1

% Get the current figure; will restore this at the end of the function
hfig = gcf;

% Load the data into a matrix
dmat = csvread('collegenum.csv', 1, 1);

% Extract the independent data from the spreadsheet; scale to work with
% the provided plotting code
xmat = 0.001*dmat(:,2:end);
% Problem size
M = size(xmat, 1);
% Extract the labels as {-1,+1}
yvec = dmat(:, 1);

% Reduce the dimensionality to 2
% % 
% % STUDENT CODE: USE SVD, PCA, TSNE, ETC. IN PLACE OF THIS LINE
% %
[V,U] = pca(xmat);
%xmat = (zscore(U(:,1:2))/sqrt(size(U,1)-1));
xmat = U(:,1:2);
% Problem size
M = size(xmat,1);


x2mat = xmat(:, [1 end]);
%size(x2mat,1)
% Classify using a linear SVM
% % 
% % STUDENT CODE: USE SOFT MARGINS; CONSIDER STANDARDIZATION
% %
linmodel = fitcsvm(x2mat, yvec, 'Standardize', 'off','Solver', 'L1QP');
wvec = [linmodel.Beta ; linmodel.Bias];

% Use 5-fold cross-validation and find the accuracy
% % 
% % STUDENT CODE: CROSS-VALIDATION GOES HERE
% %
CVSVMModel = crossval(linmodel,'KFold',5);
FirstModel = CVSVMModel.Trained{1};

acc = 1 - kfoldLoss(CVSVMModel, 'lossfun', 'classiferror');


% Plot the data and the decision line
figure(1);
plotclass2d(x2mat', yvec);
plotline(wvec, 'k');
xlabel('PCA 1', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('PCA 2', 'Interpreter', 'latex');
title('\bf Reduced data and separating line', ...
    'Interpreter', 'Latex', 'FontSize', 14);

% Compute the classifications and the scores of the data
% % 
% % STUDENT CODE: PREDICT THE CLASSIFICATION FROM "LINMODEL"
% %
%[label,scorePred] = kfoldPredict(CVSVMModel);
% Extract trained, compact classifier
%CompactSVMModel = linmodel.Trained{1};
% testInds = test(CVSVMModel.Partition,1);   % Extract the test indices
% XTest = x2mat(testInds,:);
% [label,scorePred] = predict(FirstModel, XTest);
% size(scorePred);

[label,scorePred] = predict(linmodel, x2mat);

% Compute the confusion matrix
% % 
% % STUDENT CODE: REPLACE THIS LINE WITH THE CONFUSION MATRIX
% %
%size(linmodel.SupportVectorLabels,1) 341?
% sizf = size(label);
% cmat = confusionmat(yvec(testInds),label,'Order', [1 -1]);
cmat = confusionmat(yvec, label, 'Order', [1 -1]);
% Compute the ROC curve and its AUC
% % 
% % STUDENT CODE: USE "PERFCURVE" TO GENERATE THESE VARIABLES
% %
% xroc = linspace(0, 1, size(yvec,1) + 1)';
% yroc = linspace(0, 1, size(yvec,1) + 1)';
% auc = 0.5;??
[xroc,yroc,~,auc] = perfcurve(yvec,scorePred(:,2),1);


disp(sprintf('\nAccuracy = % 3.2f, confusion matrix is:', acc));
disp(cmat);

% Plot the ROC curve
figure(2);
plot(xroc, yroc, 'b-', 'LineWidth', 2);
xlabel('1 - Sensitivity', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Specificity', 'Interpreter', 'latex', 'FontSize', 14);
title(sprintf('\\bf ROC curve with AUC=%0.2f', auc), ...
    'Interpreter', 'latex', 'FontSize', 14);


% Restore the user's figure; end
figure(hfig);

end

function [w_cls, lambda] = cls(xmat, yvec,theta)
% [WCLS,LAMBDA]=CLS(XMAT,YVEC,THETA) solves constrained
% least squares of a linear regression of YVEC to XMAT, with
% a solution tolerance of NORM(WCLS)^2<=THETA. WCLS is
% the constrained weight vector and LAMBDA is the Lagrange
% multiplier for the solution
%
% INPUTS:
%         XMAT   - MxN design matrix
%         YVEC   - Mx1 data vector
%         THETA  - positive scalar, solution threshold
% OUTPUTS:
%         WCLS   - solution coefficients
%         LAMBDA - Lagrange coefficient

% Return immediately if the threshold is invalid
if theta<0
    w_cls = [];
    lambda = [];
    return;
end

% Set up the problem as xmat*w=yvec
Im = eye(size(xmat, 2));
wfun =@(l) inv(xmat'*xmat + l*Im)*xmat'*yvec;
gfun =@(l) wfun(l)'*wfun(l) - theta;
% %
% % STUDENT CODE GOES HERE: define "w" and "g" functions from class notes
% %
% wfun =@(lval) 0;
% gfun =@(lval) 0;

% OLS solution: use pseudo-inverse for ill conditioned matrix
if cond(xmat)<1e+8
    wls = xmat\yvec;
else
    wls = pinv(xmat)*yvec;
end

% The OLS solution is used if it is within the user's threshold
if norm(wls)^2<= theta | theta<=0
    w_cls = wls;
    lambda = 0;
else
    % %
    % % STUDENT CODE GOES HERE: you can use "fzero" to estimate lambda
    % %
    lambda = 0;
    lambda = fzero(gfun, lambda);
    % CLS is a simple closed-form solution
    %w_cls = zeros(size(xmat, 2), 1);
    w_cls = wfun(lambda);
end
end

function ph = plotclass2d(dmat, lvec, lw)
% PH=PLOTCLASS(DMAT,LVEC,LW) plots a 2d data set DMAT
% for binary classification using characters in LVEC
%
% INPUTS:
%        DMAT - 2xN data matrix
%        LVEC - Nx1 binary classification
%        LW   - optional scalar, line width for plotting symbols
% OUTPUT:
%        PH   - plot handle for the current figure
% SIDE EFFECTS:
%        Plot into the current window. Low-valued labels are shown
%        as red circles and high-valued labels are shown as
%        blue "+". Optionally, LW is the line width.
%        Axes are slightly adjusted to improve legibility.

% Set the line width
if nargin >= 3 & ~isempty(lw)
  lwid = lw;
else
  lwid = 2;
end

ph = gscatter(dmat(1,:),dmat(2,:),lvec, ...
    'rb', 'o+', [], 'off');
set(ph, 'LineWidth', lwid);
axisadjust(1.1);
end

function axisadjust(axisexpand)
% AXISADJUST(AXISEXPAND) multiplies the current plot
% ranges by AXISEXPAND.  To increase by 5%, use 1.05
%
% INPUTS:
%         AXISEXPAND - positive scalar multiplier
% OUTPUTS:
%         none
% SIDE EFFECTS:
%         Changes the current plot axis

axvec = axis();
axwdth = (axvec(2) - axvec(1))/2;
axhght = (axvec(4) - axvec(3))/2;
axmidw = mean(axvec([1 2]));
axmidh = mean(axvec([3 4]));
axis([axmidw-axisexpand*axwdth , axmidw+axisexpand*axwdth , ...
      axmidh-axisexpand*axhght , axmidh+axisexpand*axhght]);
end
function ph = plotline(vvec, color, lw, nv)
% PLOTLINE(VVEC,COLOR,LW) plots a separating line
% into an existing figure
% INPUTS:
%        VVEC   - (M+1) augmented weight vector
%        COLOR  - character, color to use in the plot
%        LW   - optional scalar, line width for plotting symbols
%        NV   - optional logical, plot the normal vector
% OUTPUT:
%        PH   - plot handle for the current figure
% SIDE EFFECTS:
%        Plot into the current window. 

% Set the line width
if nargin >= 3 & ~isempty(lw)
  lwid = lw;
else
  lwid = 2;
end

% Set the normal vector
if nargin >= 4 & ~isempty(nv)
  do_normal = true;
else
  do_normal = false;
end

% Scale factor for the normal vector
sval = 0.15;

% Current axis settings
axin = axis();

% Four corners of the current axis
ll = [axin(1) ; axin(3)];
lr = [axin(2) ; axin(3)];
ul = [axin(1) ; axin(4)];
ur = [axin(2) ; axin(4)];

% Normal vector, direction vector, hyperplane scalar
nlen = norm(vvec(1:2));
uvec = vvec/nlen;
nvec = uvec(1:2);
dvec = [-uvec(2) ; uvec(1)];
bval = uvec(3);

% A point on the hyperplane
pvec = -bval*nvec;

% Projections of the axis corners on the separating line
clist = dvec'*([ll lr ul ur] - pvec);
cmin = min(clist);
cmax = max(clist);

% Start and end are outside the current plot axis, no problem
pmin = pvec +cmin*dvec;
pmax = pvec +cmax*dvec;

% Create X and Y coordinates of a box for the current axis
xbox = [axin(1) axin(2) axin(2) axin(1) axin(1)];
ybox = [axin(3) axin(3) axin(4) axin(4) axin(3)];

% Intersections of the line and the box
[xi, yi] = polyxpoly([pmin(1) pmax(1)], [pmin(2) pmax(2)], xbox, ybox);

% Point midway between the intersections
pmid = [mean(xi) ; mean(yi)];

% Range of the intersection line
ilen = norm([(max(xi) - min(xi)) ; (max(yi) - min(yi))]);

% Plot the line and re-set the axis to its original
hold on;
ph = plot([pmin(1) pmax(1)], [pmin(2) pmax(2)], ...
    strcat(color, '-'), 'LineWidth', lwid);
if do_normal
    quiver(pmid(1), pmid(2), nvec(1)*ilen*sval, nvec(2)*ilen*sval, ...
        color, 'LineWidth', lwid, 'MaxHeadSize', ilen/2, ...
        'AutoScale', 'off');
end
hold off;
end
