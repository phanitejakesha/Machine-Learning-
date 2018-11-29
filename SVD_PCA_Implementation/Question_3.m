rand('seed', 1); %%just to make all random sequences on all computers the same.
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);

inds = randperm(size(images_test, 1));
images_test = images_test(inds, :);
labels_test = labels_test(inds, :);

%To use only 1000 data points.
images_train = images_train(1:1000, :);
labels_train = labels_train(1:1000, :);

model = svmtrain(labels_train,images_train , '-t 0');

[predictedlabel,accuracy,decisionvalues] = svmpredict(labels_test,images_test,model);
succ=0;
for i=1:10000
if predictedlabel(i,1)==labels_test(i,1)
    succ=succ+1;
end
end

acc=succ/100;