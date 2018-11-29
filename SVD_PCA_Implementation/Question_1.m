clear
load MNIST_digit_data


%%% randomly permute data points
 
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
