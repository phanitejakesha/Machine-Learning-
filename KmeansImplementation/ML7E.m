clear
load MNIST_digit_data
images1_train = images_train(1:1000, :);
labels1_train = labels_train(1:1000, :);
images1_test = images_train(1001:2000, :);
labels1_test = labels_train(1001:2000, :);

[acc_number1,acc_total1]=(kNN(images1_train,labels1_train,images1_test,labels1_test,1));
[acc_number2,acc_total2]=(kNN(images1_train,labels1_train,images1_test,labels1_test,2));
[acc_number3,acc_total3]=(kNN(images1_train,labels1_train,images1_test,labels1_test,3));
[acc_number4,acc_total4]=(kNN(images1_train,labels1_train,images1_test,labels1_test,5));
[acc_number5,acc_total5]=(kNN(images1_train,labels1_train,images1_test,labels1_test,10));

function [acc,acc_av]=kNN(images_train, labels_train, images_test, labels_test, k)
acc=[];
succ=1000;
total_testdata=1000;
numbers=zeros(10,2);

    D = pdist2(images_train,images_test,'euclidean');
    [D,I]=sort(D,1);
    for i=1:1000
        mean_k=[];
    for k=1:k
         mean_k=[mean_k,labels_train(I(k,i),1)];
    end
            answ=mode(mean_k); 
            predicted_v=answ;
            original_v=labels_test(i,1);     
        if ne(predicted_v,original_v)
           succ=succ-1;
            numbers(original_v+1,2)=numbers(original_v+1,2)+1;
        else
            numbers(original_v+1,1)=numbers(original_v+1,1)+1;
        end
    end   
 for i=1:10
     total_numbers=numbers(i,2)+numbers(i,1);
     if total_numbers~=0
         success=numbers(i,1)/total_numbers;
         acc=[acc,success];
     else
         acc=[acc,0];
     end
end
 acc_av=succ/total_testdata;
end
      