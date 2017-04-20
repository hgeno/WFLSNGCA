function C = setKcenter(X,Nc)

Xtemp = X;
C = [];
idx = randperm(Nc);
C = Xtemp(:,idx(1:Nc));
% C = [C, Xtemp(:,idx(1:floor(Nc/2)))]; Xtemp(:,idx(1:floor(Nc/2))) = [];
% % 
% for i = floor(Nc/2)+1:Nc
%     [~,select_id] = max(min(sum(bsxfun(@minus,permute(C,[1,3,2]),Xtemp).^2,1),[],3),[],2);
%     C = [C, Xtemp(:,select_id)]; Xtemp(:,select_id) = [];
% end

end