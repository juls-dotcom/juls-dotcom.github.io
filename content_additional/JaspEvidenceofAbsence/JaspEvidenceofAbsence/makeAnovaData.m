%% this workbook creates csv files for the JASP workshop

n=20
k=4
noise=randn(n,k);
subj=rand(n,1);
cond=[1,1,1.3,1.8];
data=repmat(subj,1,k)+repmat(cond,n,1)+noise;
csvwrite('ttest.csv',cat(1,cond,data))

%%

%% this workbook creates csv files for the JASP workshop
% this will generate a situation in which we get four columns the first two
% simulate a saline, the second two a muscimol condition. In each case we
% have bl followed by shock, and we look say at freezing. 
n=20
k=4
noise=randn(n,k);
subj=rand(n,1);
cond=[1,1.8,1,1.8];
noise(:,2)=noise(:,2).*3
data=repmat(subj,1,k)+repmat(cond,n,1)+noise;
csvwrite('anova.csv',cat(1,cond,data))