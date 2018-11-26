clear
load MNIST_digit_data

i = 10;
close all
im = reshape(images_train(i, :), [28 28]);
imshow(im)
title(num2str(labels_train(i)));
