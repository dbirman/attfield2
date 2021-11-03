%% Load file

[header, data] = csvreadh('~/proj/attfield2/stats/rawscores.csv');
% remove last row
data = data(1:end-1,:);
%%
%   Columns 1 through 6
% 
%     ''    'Gauss_1.1'    'Gauss_2.0'    'Gauss_4.0'    'Gauss_11.0'    'Dist.'
% 
%   Columns 7 through 11
% 
%     'Flat_1.1'    'Flat_2.0'    'Flat_4.0'    'Flat_11.0'    'Shift_1.1'
% 
%   Columns 12 through 16
% 
%     'Shift_2.0'    'Shift_4.0'    'Shift_11.0'    'al_1.1'    'al_2.0'
% 
%   Columns 17 through 22
% 
%     'al_4.0'    'al_11.0'    'l1_1.1'    'l1_2.0'    'l1_4.0'    'l1_11.0'
% 
%   Columns 23 through 28
% 
%     'l2_1.1'    'l2_2.0'    'l2_4.0'    'l2_11.0'    'l3_1.1'    'l3_2.0'
% 
%   Columns 29 through 34
% 
%     'l3_4.0'    'l3_11.0'    'l4_1.1'    'l4_2.0'    'l4_4.0'    'l4_11.0'
% 
%   Columns 35 through 36
% 
%     'Reconstruct_fake'    'Reconstruct_undo'
% 

%%
medDist = nanmedian(data(:,6));

for i = 2:size(data,2)
    dat = data(:,i);
    med = nanmedian(dat);
    ci = bootci(10000,@nanmedian,dat);
    if (ci(1) > medDist) 
        star = '*> ';
    elseif (ci(2) < medDist)
        star = '*< ';
    else
        star = '';
    end
    disp(sprintf('%s%s: %0.2f, [%0.2f, %0.2f]',star,header{i},med,ci(1),ci(2)));
end

%% 
wdata = data - repmat(data(:,6),1,size(data,2));


for i = 2:size(wdata,2)
    dat = wdata(:,i);
    med = nanmedian(dat);
    ci = bootci(10000,@nanmedian,dat);
    disp(sprintf('delta within %s: %0.2f, [%0.2f, %0.2f]',header{i},med,ci(1),ci(2)));
end