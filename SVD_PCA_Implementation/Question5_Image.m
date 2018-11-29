load MNIST_digit_data;
rand('seed', 1); %%just to make all random sequences on all computers the same.
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);

%To use only 1000 data points.
images_train = images_train(1:1000, :);
labels_train = labels_train(1:1000, :);
   
mean_train=mean(images_train,2);
X1=images_train;
X1=X1-(mean_train * ones(1,784));
[utrain,strain,vtrain]=svds(X1,dimensions(1,i));

for i=1:10
im = reshape(vtrain(:,i),[28,28]);
subplot(2,5,i);
imagesc(im);
end