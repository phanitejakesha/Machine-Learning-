clear
load MNIST_digit_data
images_test = images_test(1:1000, :);


[acc_number1,acc_total1]=(kNN(images_train,labels_train,images_test,labels_test,1));
[acc_number2,acc_total2]=(kNN(images_train,labels_train,images_test,labels_test,2));
[acc_number3,acc_total3]=(kNN(images_train,labels_train,images_test,labels_test,3));
[acc_number4,acc_total4]=(kNN(images_train,labels_train,images_test,labels_test,5));
[acc_number5,acc_total5]=(kNN(images_train,labels_train,images_test,labels_test,10));
plot(acc_total1,'DisplayName','acc_total1');hold on;plot(acc_total2,'DisplayName','acc_total2');plot(acc_total3,'DisplayName','acc_total3');plot(acc_total4,'DisplayName','acc_total4');plot(acc_total5,'DisplayName','acc_total5');hold off;

function [acc,acc_av]=kNN(images_train, labels_train, images_test, labels_test, k)
acc=[];
acc_av=[];
for j=30:1000:10030
succ=1000;
total_testdata=1000;
D = pdist2(images_train(1:j,:),images_test,'euclidean');
[D,I]=sort(D,1);
numbers=zeros(10,2);
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
acc_av=[acc_av,succ/total_testdata];
end
end
