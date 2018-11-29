function [] = Question_SVD(trainImages)
    
    dimensions = round(logspace(1,log10(500),50));
    errorMatrix = zeros(1,50);

    x_data = trainImages;
    x_mean = mean(x_data,2);
    data_cols = size(x_data, 2);

    x_data = x_data - x_mean * ones(1, data_cols);

    for i = 1 : size(dimensions, 2)
        [U,S,V] = svds(x_data, dimensions(1,i));
        E = x_data * V;
        reconstructed = E * V';
        errorMatrix(1,i) = immse(x_data, reconstructed) ;
    end

    figure;

    plot(dimensions, errorMatrix);

    xlabel('number of dimensions');
    ylabel('mean square error');

    figure;
    title('Visualizing first 10 eigen vectors');
    for i = 1: 10
        im = reshape(V(:,i), [28 28]);
        subplot(2,5,i);
        imagesc(im);
    end
end