clear
clc

p = 0.7;
n = 1e4;
s1 = eye(2) + 3;
s2 = eye(2) + 1;
mu1 = [1,1];
mu2 = [-1,-2];

% mu2 = mu1;
% s2 = s1;
% p = 0.5;
% p=1;

x1 = mvnrnd(mu1, s1, floor(n*p));
x2 = mvnrnd(mu2, s2, floor(n*(1-p)));
x = [x1;x2];

f = figure('units','normalized','outerposition',[0 0 1 1]);
plot(x1(:,1), x1(:,2), '.r');
hold on; grid on;
plot(x2(:,1), x2(:,2), '.b');
plot_ellipse(p, mu1, s1, 'g');
plot_ellipse(1 - p, mu2, s2, 'g');

p_old = rand;
s1_old = eye(2,2);
s2_old = eye(2,2);
mu1_old = rand(2,1);
mu2_old = rand(2,1);

for i = 1:5000
    
    if mod(i, 10) == 0 || i==1
        if ishandle(f)
            ll = mean(log(mvnpdf(x, mu1_old', s1_old) * p_old + mvnpdf(x, mu2_old', s2_old) * (1 - p_old)));
            title(sprintf('iteration: %d, L: %f, p: %f , s: %f', i, ll, p_old, s1_old(1, 1)));
            e1 = plot_ellipse(p_old, mu1_old, s1_old, 'r');
            e2 = plot_ellipse(1 - p_old, mu2_old, s2_old, 'b');
        else
            break
        end
        pause(0.1);
    end
    
    r = [mvnpdf(x, mu1_old', s1_old), mvnpdf(x, mu2_old', s2_old)];
    r = bsxfun(@times, r, [p_old, 1-p_old]);
    r = bsxfun(@rdivide, r, sum(r,2));
    p_old = sum(r(:,1)) / sum(r(:));
    mu1_old = sum(bsxfun(@times, x, r(:,1)),1)' / sum(r(:,1));
    mu2_old = sum(bsxfun(@times, x, r(:,2)),1)' / sum(r(:,2));
    s1_new = (bsxfun(@times, x, r(:,1))' * x) / sum(r(:,1)) - mu1_old * mu1_old' ;
    s1_new = (s1_new + s1_new') / 2;
    s2_new = (bsxfun(@times, x, r(:,2))' * x) / sum(r(:,2)) - mu2_old * mu2_old';
    s2_new = (s2_new + s2_new') / 2;
    
    dif = s1_old - s1_new;
%     if sum(abs(dif(:))) < 2e-4
%         break
%     end
    
    s1_old = s1_new;
    s2_old = s2_new;
    
    %     F(i) = getframe(gcf);
    
    delete(e1); delete(e2);
    
end

if ishandle(f)
    e1 = plot_ellipse(p_old, mu1_old, s1_old, 'r');
    e2 = plot_ellipse(1 - p_old, mu2_old, s2_old, 'b');
end

% writerObj = VideoWriter('myVideo2.avi');
% writerObj.FrameRate = 40;
% open(writerObj);
% for i=1:length(F)
%     frame = F(i) ;
%     writeVideo(writerObj, frame);
% end
% close(writerObj);