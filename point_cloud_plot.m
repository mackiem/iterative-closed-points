graphics_toolkit('gnuplot')
P = dlmread('P.txt');
Q = dlmread('Q.txt');
clf();
figure(1);
plot3(P(1,:),P(2,:),P(3,:), '.r', Q(1,:),Q(2,:),Q(3,:), '.b');
view(3);
iters = [1, 2, 5, 10, 15, 20];
no = 2;
for (iter = iters)
tQ = dlmread(strcat('transformed_Q_',int2str(iter),'.txt'));
figure(no++);
plot3(P(1,:),P(2,:),P(3,:), '.r', tQ(1,:),tQ(2,:), tQ(3,:), '.g');
view(3);
end