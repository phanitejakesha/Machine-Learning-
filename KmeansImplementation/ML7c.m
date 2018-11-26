clear
load MNIST_digit_data
images_test = images_test(1:1000, :);
acc_av=[];
k=[];

for j=30:1000:10030
succ=1000;
total_testdata=1000;
D = pdist2(images_train(1:j,:),images_test,'euclidean');
[D,I]=sort(D,1);
for i=1:1000
           answ=labels_train(I(1,i),1); 
            predicted_v=answ;
            original_v=labels_test(i,1);     
        if ne(predicted_v,original_v)
           succ=succ-1;
        end
end
acc_av=[acc_av,succ/total_testdata];
k=[k,j];
end