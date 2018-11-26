clear
load MNIST_digit_data

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