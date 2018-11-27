clear
load MNIST_digit_data


j1=1;
j2=1;

for i=1:15000
  if labels_train(i,1)==1
      train_data_1(j1,:)=images_train(i,:);
       labels_traindata_1(j1,1)=1;
        j1=j1+1;
  end  
    if labels_train(i,1)==6
      train_data_6(j2,:)=images_train(i,:);
      labels_traindata_6(j2,1)=-1;
        j2=j2+1;
    end 
end

images_train1(1:500,:)=train_data_1(1:500,:);
images_train1(501:1000,:)=train_data_6(1:500,:);
labels_train1(1:500,:)=labels_traindata_1(1:500,:);
labels_train1(501:1000,:)=labels_traindata_6(1:500,:);



images_test1(1:500,:)=train_data_1(501:1000,:);
images_test1(501:1000,:)=train_data_6(501:1000,:);
labels_test1(1:500,:)=labels_traindata_1(501:1000,:);
labels_test1(501:1000,:)=labels_traindata_6(501:1000,:);

rand('seed', 3);
inds = randperm(size(images_train1, 1));
images_train1 = images_train1(inds, :);
labels_train1 = labels_train1(inds, :);

inds = randperm(size(images_test1, 1));
images_test1 = images_test1(inds, :);
labels_test1 = labels_test1(inds, :);



weights=zeros(1,784);
b=0;
err_arr=[];

for ep=1:20 %% from 5th epoch the error is not changed and model is constant
    
for i=1:1000    
    pred=dot(images_train1(i,:),weights)+b;
    if pred*labels_train1(i,1)<=0
        weights=weights+images_train1(i,:)*labels_train1(i,1); 
        b=b+labels_train1(i,1);
    end
    
end

err=0;
pred_table_1=zeros(500,3);
pred_table_2=zeros(500,3);
m=1;
n=1;


for i=1:1000
    pred=dot(images_test1(i,:),weights)+b;
   
    if sign(labels_test1(i,1))==1
    pred_table_1(m,1)=pred;
    pred_table_1(m,2)=i;
    pred_table_1(m,3)=sign(labels_test1(i,1));
    m=m+1;
    end
    
    if sign(labels_test1(i,1))==-1
    pred_table_2(n,1)=pred;
    pred_table_2(n,2)=i;
    pred_table_2(n,3)=sign(labels_test1(i,1));
    n=n+1;
    end
    
    if sign(pred)*sign(labels_test1(i,1))<=0
        err=err+1;
    end
end
err_arr(ep)=(1000-err)/10;
end

[~,idx1] = sort(pred_table_1(:,1)); % sort just the first column
sortedmat1 = pred_table_1(idx1,:);   % sort the whole matrix using the sort indices



[~,idx2] = sort(pred_table_2(:,1)); % sort just the first column
sortedmat2 = pred_table_2(idx2,:);   % sort the whole matrix using the sort indices

    
in1 = reshape(images_test1(sortedmat1(1,2),:), [28 28]);  
in2 = reshape(images_test1(sortedmat1(2,2),:), [28 28]);  
in3 = reshape(images_test1(sortedmat1(3,2),:), [28 28]);  
in4 = reshape(images_test1(sortedmat1(4,2),:), [28 28]);  
in5 = reshape(images_test1(sortedmat1(5,2),:), [28 28]);  
in6= reshape(images_test1(sortedmat1(6,2),:), [28 28]);  
in7 = reshape(images_test1(sortedmat1(7,2),:), [28 28]);  
in8 = reshape(images_test1(sortedmat1(8,2),:), [28 28]);  
in9 = reshape(images_test1(sortedmat1(9,2),:), [28 28]);  
in10 = reshape(images_test1(sortedmat1(10,2),:), [28 28]);  
in11 = reshape(images_test1(sortedmat1(11,2),:), [28 28]);  
in12 = reshape(images_test1(sortedmat1(12,2),:), [28 28]);  
in13= reshape(images_test1(sortedmat1(13,2),:), [28 28]);  
in14 = reshape(images_test1(sortedmat1(14,2),:), [28 28]);  
in15 = reshape(images_test1(sortedmat1(15,2),:), [28 28]);  
in16 = reshape(images_test1(sortedmat1(16,2),:), [28 28]);  
in17 = reshape(images_test1(sortedmat1(17,2),:), [28 28]);  
in18 = reshape(images_test1(sortedmat1(18,2),:), [28 28]);  
in19 = reshape(images_test1(sortedmat1(19,2),:), [28 28]);  
in20 = reshape(images_test1(sortedmat1(20,2),:), [28 28]);  

x=horzcat(in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11,in12,in13,in14,in15,in16,in17,in18,in19,in20);


in1 = reshape(images_test1(sortedmat2(1,2),:), [28 28]);  
in2 = reshape(images_test1(sortedmat2(2,2),:), [28 28]);  
in3 = reshape(images_test1(sortedmat2(3,2),:), [28 28]);  
in4 = reshape(images_test1(sortedmat2(4,2),:), [28 28]);  
in5 = reshape(images_test1(sortedmat2(5,2),:), [28 28]);  
in6= reshape(images_test1(sortedmat2(6,2),:), [28 28]);  
in7 = reshape(images_test1(sortedmat2(7,2),:), [28 28]);  
in8 = reshape(images_test1(sortedmat2(8,2),:), [28 28]);  
in9 = reshape(images_test1(sortedmat2(9,2),:), [28 28]);  
in10 = reshape(images_test1(sortedmat2(10,2),:), [28 28]);  
in11 = reshape(images_test1(sortedmat2(11,2),:), [28 28]);  
in12 = reshape(images_test1(sortedmat2(12,2),:), [28 28]);  
in13= reshape(images_test1(sortedmat2(13,2),:), [28 28]);  
in14 = reshape(images_test1(sortedmat2(14,2),:), [28 28]);  
in15 = reshape(images_test1(sortedmat2(15,2),:), [28 28]);  
in16 = reshape(images_test1(sortedmat2(16,2),:), [28 28]);  
in17 = reshape(images_test1(sortedmat2(17,2),:), [28 28]);  
in18 = reshape(images_test1(sortedmat2(18,2),:), [28 28]);  
in19 = reshape(images_test1(sortedmat2(19,2),:), [28 28]);  
in20 = reshape(images_test1(sortedmat2(20,2),:), [28 28]);  

y=horzcat(in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11,in12,in13,in14,in15,in16,in17,in18,in19,in20);



in1 = reshape(images_test1(sortedmat2(500,2),:), [28 28]);  
in2 = reshape(images_test1(sortedmat2(499,2),:), [28 28]);  
in3 = reshape(images_test1(sortedmat2(498,2),:), [28 28]);  
in4 = reshape(images_test1(sortedmat2(497,2),:), [28 28]);  
in5 = reshape(images_test1(sortedmat2(496,2),:), [28 28]);  
in6= reshape(images_test1(sortedmat2(495,2),:), [28 28]);  
in7 = reshape(images_test1(sortedmat2(494,2),:), [28 28]);  
in8 = reshape(images_test1(sortedmat2(493,2),:), [28 28]);  
in9 = reshape(images_test1(sortedmat2(492,2),:), [28 28]);  
in10 = reshape(images_test1(sortedmat2(491,2),:), [28 28]);  
in11 = reshape(images_test1(sortedmat2(490,2),:), [28 28]);  
in12 = reshape(images_test1(sortedmat2(489,2),:), [28 28]);  
in13= reshape(images_test1(sortedmat2(488,2),:), [28 28]);  
in14 = reshape(images_test1(sortedmat2(487,2),:), [28 28]);  
in15 = reshape(images_test1(sortedmat2(486,2),:), [28 28]);  
in16 = reshape(images_test1(sortedmat2(485,2),:), [28 28]);  
in17 = reshape(images_test1(sortedmat2(484,2),:), [28 28]);  
in18 = reshape(images_test1(sortedmat2(483,2),:), [28 28]);  
in19 = reshape(images_test1(sortedmat2(482,2),:), [28 28]);  
in20 = reshape(images_test1(sortedmat2(481,2),:), [28 28]);  


x1=horzcat(in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11,in12,in13,in14,in15,in16,in17,in18,in19,in20);

in1 = reshape(images_test1(sortedmat1(500,2),:), [28 28]);  
in2 = reshape(images_test1(sortedmat1(499,2),:), [28 28]);  
in3 = reshape(images_test1(sortedmat1(498,2),:), [28 28]);  
in4 = reshape(images_test1(sortedmat1(497,2),:), [28 28]);  
in5 = reshape(images_test1(sortedmat1(496,2),:), [28 28]);  
in6= reshape(images_test1(sortedmat1(495,2),:), [28 28]);  
in7 = reshape(images_test1(sortedmat1(494,2),:), [28 28]);  
in8 = reshape(images_test1(sortedmat1(493,2),:), [28 28]);  
in9 = reshape(images_test1(sortedmat1(492,2),:), [28 28]);  
in10 = reshape(images_test1(sortedmat1(491,2),:), [28 28]);  
in11 = reshape(images_test1(sortedmat1(490,2),:), [28 28]);  
in12 = reshape(images_test1(sortedmat1(489,2),:), [28 28]);  
in13= reshape(images_test1(sortedmat1(488,2),:), [28 28]);  
in14 = reshape(images_test1(sortedmat1(487,2),:), [28 28]);  
in15 = reshape(images_test1(sortedmat1(486,2),:), [28 28]);  
in16 = reshape(images_test1(sortedmat1(485,2),:), [28 28]);  
in17 = reshape(images_test1(sortedmat1(484,2),:), [28 28]);  
in18 = reshape(images_test1(sortedmat1(483,2),:), [28 28]);  
in19 = reshape(images_test1(sortedmat1(482,2),:), [28 28]);  
in20 = reshape(images_test1(sortedmat1(481,2),:), [28 28]);  

y1=horzcat(in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11,in12,in13,in14,in15,in16,in17,in18,in19,in20);

z=vertcat(y1,x,y,x1);
imshow(z);









