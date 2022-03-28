%% Analyze behavioral data from pilot
% Josh and Kai did a pilot test on 2/28. This script loads their data and
% checks the psychometric function as a function of duration.
addpath(genpath(fullfile('~/proj/attfield2')));
addpath(genpath(fullfile('~/proj/mgl')));
addpath(genpath(fullfile('~/proj/gru')));
addpath(genpath(fullfile('~/proj/mrTools')));

subjs = [600 611 620 700 99 731 733];
yob = [1997 2001 1996 2002 1999 2001 2000];
gender = [2 2 2 1 2 2 2];

SIDs = {};
for si = 1:length(subjs)
    SIDs{si} = sprintf('s%03.0f',subjs(si));
end

%% Load data files
dataFolder = fullfile('~/data/spatobj/');

% get the category labels
ex = load(fullfile(dataFolder,'exemplars.mat'));
categories = ex.cat_map;

adatas = {};
frameRate = [];
for si = 1:length(SIDs)
    sid = SIDs{si};
    subjFolder = fullfile(dataFolder,sid);
    files = dir(fullfile(dataFolder,sid,'*.mat'));
    
    adata = [];
    blockn = 1;
    % COLUMN
    %    1      2       3       4      5        6               7           8        9     10   11   12   13   14   15 
    % block# trial# targetCat focal image# targetPresent responsePresent duration correct dead cat0 cat1 cat2 cat3  rt 
    
    dataFiles = {};
    for fi = 1:length(files)
        load(fullfile(subjFolder,files(fi).name));
        frameRate(si) = myscreen.frametime;
        e = getTaskParameters(myscreen,task);
        e = e{1};
        
        % convert to long format 
        for ei = 1:length(e)
            cBlock = e(ei);
            if cBlock.nTrials>1
                if ~any(cBlock.parameter.targetCategory==1)
                    % this is a real block, not the example phase, go through
                    % the trials and add them
                    cdata = [blockn*ones(cBlock.nTrials,1) (1:cBlock.nTrials)' cBlock.parameter.targetCategory' ...
                        cBlock.parameter.focal' cBlock.parameter.imageNumber' cBlock.parameter.targetPresent' ...
                        cBlock.randVars.responsePresent' cBlock.randVars.duration' cBlock.randVars.correct' ...
                        cBlock.randVars.dead' cBlock.randVars.imgCat0' cBlock.randVars.imgCat1' ...
                        cBlock.randVars.imgCat2' cBlock.randVars.imgCat3' cBlock.reactionTime'];
                    adata = [adata ; cdata];
                end
            end
        end
    end
    
    % remove dead trials
    if any(adata(:,10))
        disp(sprintf('Removing %i dead trials out of %i',sum(adata(:,10)),size(adata,1)));
        adata = adata(adata(:,10)==0,:);
    end
    
    adatas{si} = adata;
end

%% Make a new dataset to save to a CSV
% Header:
%    1      2     3      4         5               6    7    8
% trial #  cat  focal   present  responsePresent  dur   tp   fa 

header = {'trial','category','focal','targetPresent','responsePresent','duration','tp','fa'};

tp = (adata(:,6)==1) .* (adata(:,7)==1); 
fa = (adata(:,6)==0) .* (adata(:,7)==1);
outdata = [(1:size(adata,1))' adata(:,2) adata(:,4) adata(:,6) adata(:,7) adata(:,8) tp fa];

outdata = outdata(~isnan(adata(:,7)),:);

csvwriteh(fullfile('~/proj/attfield2/behav/out.csv'),outdata,header);

%% Stack all data at end
adatas{end+1} = [];
for ai = 1:(length(adatas)-1)
    adatas{end} = [adatas{end} ; adatas{ai}];
end
SIDs{end+1} = 'all';

%% Check performance across categories
catperf = [];

ucats = unique(adata(:,3));

for i = 1:length(ucats)
    catperf(i) = nanmean(adata(adata(:,3)==ucats(i),9));
end

%% Save out data

dps = [];
cs = [];
rts = [];
vs = [];
as = [];
Ters = [];

for si = 1:length(adatas)
    
    adata = adatas{si};
    
    %    1      2       3       4      5        6               7           8        9     10   11   12   13   14 
    % block# trial# targetCat focal image# targetPresent responsePresent duration correct dead cat0 cat1 cat2 cat3 
    
    udur = unique(adata(:,8));
    
    % duration x target category x focal condition
    %                               (dist=1, focal=2)

    dp = nan(length(udur),2);
    dp_ci = nan(length(udur),2);
    c = nan(length(udur),2);
    rt = nan(length(udur),2);
    v = nan(length(udur),2);
    a = nan(length(udur),2);
    Ter = nan(length(udur),2);
    for ui = 1:length(udur)
        cdur = udur(ui);
        sdata = sel(adata,8,cdur);
        
%         for tar = setdiff(0:20,1)
%             tdata = sel(sdata,3,tar);
            
            for fd = [0 1]
                fdata = sel(sdata,4,fd);
                
                if ~isempty(fdata)
                    % COMPUTE STATISTICS FOR EZDIFFUSION
                    % we need the % corr, mean rt, and var rt
                    Pc = mean(fdata(:,9));
                    mrt = mean(fdata(:,15));
                    vrt = var(fdata(:,15));
                    % convert
                    [v(ui,fd+1),a(ui,fd+1),Ter(ui,fd+1)] = ezdiffusion(Pc,vrt,mrt);
                    
                    % COMPUTE STATISTICS FOR DPRIME
                    % compute d' 
                    hitsfunc = @(x) sum(x(:,6)==1 .* x(:,7)==1)/size(x,1);
                    fafunc = @(x) sum(x(:,6)==0 .* x(:,7)==1)/size(x,1); 
                    dpfunc = @(x) norminv(hitsfunc(x)) - norminv(fafunc(x));
                    
                    dp(ui,fd+1) = dpfunc(fdata);
                    
                    reps = 1000;
                    dp_ = nan(1,reps);
                    % repeat 1000 times selecting a random subset of data
                    % and computing the dp parameter
                    parfor ri=1:reps
                        rdata = fdata(randsample(1:size(fdata,1),size(fdata,1),true),:);
                        dp_(ri) = dpfunc(rdata);
                    end
                    dp_ci(ui,fd+1) = 1.96*nanstd(dp_(~(dp_==inf)));
                    
                    cfunc = @(x) 0.5*(norminv(hitsfunc(x)) + norminv(fafunc(x)));

                    c(ui,fd+1) = cfunc(fdata);
                    
                    % compute rt
                    rfunc = @(x) nanmean(x(:,15));
                    rt(ui,fd+1) = rfunc(fdata);
                end
            end
%         end
    end
    
    vs(si,:,:) = v;
    as(si,:,:) = a;
    Ters(si,:,:) = Ter;
    cs(si,:,:) = c;
    dps(si,:,:) = dp;
    rts(si,:,:) = rt;
    
end

%% Setup some plotting stuff
msdur = frameRate(1) * udur;

%% Plot drift diffusion parameters
h = figure; hold on
% vs(isnan(vs)) = 0;

for si = 8
%     subplot(length(adatas),1,si); hold on
    title(SIDs{si});
    
    % distributed
    plot(msdur,squeeze(vs(si,:,1)),'ok');
    vfit1 = fitLog(msdur,squeeze(vs(si,:,1))');
    plot(vfit1.x,vfit1.y,'-k');
    % focal
    plot(msdur,squeeze(vs(si,:,2)),'ob');
    vfit1 = fitLog(msdur,squeeze(vs(si,:,2))');
    plot(vfit1.x,vfit1.y,'-b');
end
savepdf(h,fullfile('~/proj/attfield2/behav/figures/drift_is_dprime.pdf'));

%% FOR PAPER: Correlation of dp and v
dp_subj = dps(1:7,:,:);
v_subj = vs(1:7,:,:);

% for each subject, compute correlation across durations for focal/dist
for si = 1:7
    for fi = 1:2
        dat = [dp_subj(si,:,fi)' v_subj(si,:,fi)'];
%         if any(isnan(dat,2))
        temp = corrcoef(dat);
        c(si,fi) = temp(1,2);
    end
end

%% Plot RT
h = figure;

for si = 1:length(adatas)
    subplot(length(adatas),1,si); hold on
    title(SIDs{si});
    
    % distributed
    plot(msdur,squeeze(rts(si,:,1)),'-k');
    plot(msdur,squeeze(rts(si,:,2)),'-b');
end

%% Save data to csv file
csvdata = [];

for si = 1:7
    for di = 1:length(msdur)
        for fi = 1:2
            dat = [si msdur(di) fi dps(si,di,fi)];
            csvdata = [csvdata; dat];
        end
    end
end

csvwriteh(fullfile('~/proj/attfield2/behav/dps.csv'),csvdata,{'Subject','Duration','Attend','dprime'});

%% Plot sensitivity
SIDs{end+1} = 'all';
yvalsdp = [0 1 2 3];
yvalsauc = normcdf(yvalsdp./sqrt(2));
yvalsauc = round(yvalsauc*100)/100;
ylabels = arrayfun(@(x,y) sprintf('%1.0f/%0.2f',x,y),yvalsdp,yvalsauc,'UniformOutput',false);

for si = length(adatas)
    h = figure; hold on
    %subplot(length(adatas),1,si); hold on
    
    dp = squeeze(dps(si,:,:));
    
    title(SIDs{si});
    % distributed
%     errbar(msdur,dp(:,1),dp_ci(:,1),'-k');
    p(1) = plot(msdur,dp(:,1),'o','MarkerFaceColor','k','MarkerEdgeColor','w','MarkerSize',5);
%     fit = fitSatExponential(msdur,dp(:,1));
%     plot(fit.x,fit.y,'--k');
    fitfocal = fitLog(msdur,dp(:,1));
    plot(fitfocal.x,fitfocal.y,'-k');
    disp(sprintf('Focal d''(x) = %1.4f * log(%1.4f * x + 1)',fitfocal.params(1),fitfocal.params(2)));
    % focal
%     errbar(msdur+.002,dp(:,2),dp_ci(:,2),'-b');
    p(2) = plot(msdur+.002,dp(:,2),'o','MarkerFaceColor','b','MarkerEdgeColor','w','MarkerSize',5);
%     fit = fitSatExponential(msdur,dp(:,2));
%     plot(fit.x,fit.y,'--b');
    fitdist = fitLog(msdur,dp(:,2));
    plot(fitdist.x,fitdist.y,'-b');
    disp(sprintf('Distributed d''(x) = %1.4f * log(%1.4f * x + 1)',fitdist.params(1),fitdist.params(2)));

    set(gca,'YTick',yvalsdp,'YTickLabel',ylabels);
    
    xlabel('Duration (s)');
    ylabel('Sensitivity (d''/AUC)');
    
    axis([0 0.3 0 3]);
    legend(p,{'Distributed','Focal'},'Location','SouthEast');
    
%     drawPublishAxis('figSize=[8.9,12]');
end

%%

savepdf(h,fullfile('~/proj/attfield2/behav/figures/pilot_2020_12_08.pdf'));

%%

h = figure; hold on

dp = squeeze(dps(end,:,:));
dp_ci = squeeze(bootci(1000, @mean, squeeze(dps(1:(end-1),:,:))));
dp_err = squeeze(dp_ci(2,:,:)) - dp;

% distributed

fitboth = fitLogModel(msdur,dp(:,2),dp(:,1));
%     errbar(msdur,dp(:,1),dp_ci(:,1),'-k');
plot(fitboth.x,fitboth.y_d,'-k','LineWidth',3);
errbar(msdur,dp(:,1),dp_err(:,1),'-k','LineWidth',3);
p(1) = plot(msdur,dp(:,1),'o','MarkerFaceColor','k','MarkerEdgeColor','w','MarkerSize',15);
disp(sprintf('Focal d''(x) = %1.3f * log(%1.3f * x + 1)',fitboth.params(3)*fitboth.params(1),fitboth.params(2)));
% focal
%     errbar(msdur+.002,dp(:,2),dp_ci(:,2),'-b');
plot(fitboth.x,fitboth.y_f,'-r','LineWidth',3);
errbar(msdur,dp(:,2),dp_err(:,2),'-r','LineWidth',3);
p(2) = plot(msdur,dp(:,2),'o','MarkerFaceColor','r','MarkerEdgeColor','w','MarkerSize',15);
disp(sprintf('Distributed d''(x) = %1.3f * log(%1.3f * x + 1)',fitboth.params(1),fitboth.params(2)));

yvalsdp = [0 1 2];
yvalsauc = normcdf(yvalsdp./sqrt(2));
yvalsauc = round(yvalsauc*100)/100;
ylabels = arrayfun(@(x,y) sprintf('%1.0f/%0.2f',x,y),yvalsdp,yvalsauc,'UniformOutput',false);

axis([0 0.3 0 2.5]);
set(gca,'YTick',yvalsdp,'YTickLabel',ylabels);

xlabel('Duration (s)');
ylabel('Sensitivity (d''/AUC)');

% legend(p,{'Distributed','Focal'},'Location','SouthEast');

drawPublishAxis('figSize=[20,14]','yLabelOffset=-11/64','poster=1');

savepdf(h,fullfile('~/proj/attfield2/behav/figures/pilot_poster.pdf'));

%% PAPER: Find ms duration when d' > 1 for dist/focal

for si = 1:7
    fitboth = fitLogModel(msdur,squeeze(dps(si,:,2))',squeeze(dps(si,:,1))');

    % save the x value at which d'>1
    dp1d(si) = fitboth.x(find(fitboth.y_d>=1,1));
    dp1f(si) = fitboth.x(find(fitboth.y_f>=1,1));
    
    % save the multiplier parameter for focal
    dpmult(si) = fitboth.params(3);
end

dp1d_ci = bootci(10000,@nanmean,dp1d*1000);
disp(sprintf('ms average to dp 1 distributed %1.1f [%1.1f, %1.1f]',mean(dp1d)*1000,dp1d_ci(1),dp1d_ci(2)));
dp1f_ci = bootci(10000,@nanmean,dp1f*1000);
disp(sprintf('ms average to dp 1 focal %1.1f [%1.1f, %1.1f]',mean(dp1f)*1000,dp1f_ci(1),dp1f_ci(2)));

dpmult_ci = bootci(10000,@nanmean,dpmult);
disp(sprintf('Distributed to focal d'' multiplier %1.2f [%1.2f, %1.2f]',mean(dpmult),dpmult_ci(1),dpmult_ci(2)));

%% REPEAT analysis for drift rate
vs_fix = vs;
for di = 1:6
    for fi = 1:2
        temp = squeeze(vs(:,di,fi));
        temp(isnan(temp)) = nanmean(temp);
        vs_fix(:,di,fi) = temp;
    end
end

clear dp1d dp1f dpmult
for si = 1:7
    fitboth = fitLogModel(msdur,squeeze(vs_fix(si,:,2))',squeeze(vs_fix(si,:,1))');

    % save the x value at which d'>1
    dp1d(si) = fitboth.x(find(fitboth.y_d>=0.05,1));
    dp1f(si) = fitboth.x(find(fitboth.y_f>=0.05,1));
    
    % save the multiplier parameter for focal
    dpmult(si) = fitboth.params(3);
end

dp1d_ci = bootci(10000,@nanmean,dp1d*1000);
disp(sprintf('ms average to dp 1 distributed %1.1f [%1.1f, %1.1f]',mean(dp1d)*1000,dp1d_ci(1),dp1d_ci(2)));
dp1f_ci = bootci(10000,@nanmean,dp1f*1000);
disp(sprintf('ms average to dp 1 focal %1.1f [%1.1f, %1.1f]',mean(dp1f)*1000,dp1f_ci(1),dp1f_ci(2)));

dpmult_ci = bootci(10000,@nanmean,dpmult);
disp(sprintf('Distributed to focal d'' multiplier %1.2f [%1.2f, %1.2f]',mean(dpmult),dpmult_ci(1),dpmult_ci(2)));

%% CHECK IF OTHER DRIFT DIFFUSION PARAMETERS CHANGE
h = figure; hold on
vs_ = squeeze(nanmean(vs));

for si = 1:7
    vsdist = squeeze(vs(si,:,1))';
    vsfocal = squeeze(vs(si,:,2))';
    vsdist(isnan(vsdist)) = nanmean(vsdist);
    vsfocal(isnan(vsfocal)) = nanmean(vsfocal);
    fitboth = fitLogModel(msdur,vsfocal,vsdist);
    va(si) = fitboth.params(3);
end
plot(msdur,vs_(:,1),'ok');
plot(msdur,vs_(:,2),'or');


h = figure; hold on
Ters(Ters<0) = 0;
ter_ = squeeze(nanmean(Ters));

for si = 1:7
    terdist = squeeze(Ters(si,:,1))';
    terfocal = squeeze(Ters(si,:,2))';
    terdist(isnan(terdist)) = nanmean(terdist);
    terfocal(isnan(terfocal)) = nanmean(terfocal);
    fitboth = fitLogModel(msdur,terfocal,terdist);
    tera(si) = fitboth.params(3);
end
plot(msdur,ter_(:,1),'ok');
plot(msdur,ter_(:,2),'or');

Tersno0 = Ters;
Tersno0(Tersno0==0) = nan;

%% STATs
vdiff = vs(:,:,2)-vs(:,:,1);
% average 