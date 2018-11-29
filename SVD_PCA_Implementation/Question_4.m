load MNIST_digit_data
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


model = svmtrain(labels_train, images_train, '-t 0');
    weights = model.SVs' * model.sv_coef;
    bias = -model.rho;

    X1 = images_train(1,:);
    primal_decision_values=(X1*weights + bias);

    [predicted_label, accuracy, decision_values] = svmpredict(labels_test(1,1), X1, model);
    display(predicted_label);
    meanvalue = mean(sign(primal_decision_values));
    display(meanvalue);