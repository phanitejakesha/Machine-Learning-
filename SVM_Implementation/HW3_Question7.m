clear
load MNIST_digit_data

w=zeros(10,784);
b=zeros(10,1);
pred_table=zeros(1,10);
conf_matrix=zeros(10,10);

for j=1:10
    [w(j,:),b(j,1)]=SVM_train(images_train,labels_train,j-1);
   % disp(norm(w(j,:)));
end

top_wrong=[];
top_act=zeros(1000,3);
%b=b/norm(b);
x=1;

for j=1:10000        
        for i=1:10
            pred_table(1,i)=dot(images_test(j,:),w(i,:))+b(i,1);
        end
        
        [max_value,index] = max(pred_table);
        if ne(labels_test(j,1)+1,index)
            top_wrong(x,:)=images_test(j,:);
            top_act(x,1)=max_value;
            top_act(x,2)=labels_test(j,:);
            top_act(x,3)=index-1;
            top_act(x,4)=j;
            x=x+1;
        end
        conf_matrix(labels_test(j,1)+1,index)=conf_matrix(labels_test(j,1)+1,index)+1;
        
end
B=sortrows(top_act,1);
display(trace(conf_matrix));
for i=1 :20
in1 = reshape(images_test(B(size(top_act,1)+1-i,4), :), [28 28]);
subplot(4,5,i);
imshow(in1);
title(strcat("Predicted=",int2str(B(size(top_act,1)+1-i,3))," Original=",int2str(B(size(top_act,1)+1-i,2))));
end
