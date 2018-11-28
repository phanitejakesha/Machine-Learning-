clear
load MNIST_digit_data

w=zeros(10,784);
b=zeros(10,1);
pred_table=zeros(1,10);
conf_matrix=zeros(10,10);

for j=1:10
    [w(j,:),b(j,1)]=SVM_train(images_train,labels_train,j-1);
end

for j=1:10000
        
    
        for i=1:10
            pred_table(1,i)=dot(images_test(j,:),w(i,:))+b(i,1);
        end
        
        [max_value,index] = max(pred_table);
        conf_matrix(labels_test(j,1)+1,index)=conf_matrix(labels_test(j,1)+1,index)+1;
        
end


for i=1:10
    Normalizedconf_matrix(:,i)=conf_matrix(i,:)/sum(conf_matrix(i,:));
end

avg=(trace(Normalizedconf_matrix))/10;
accracy=avg*100;


