%% Analyze behavioral data from pilot
% Josh and Kai did a pilot test on 2/28. This script loads their data and
% checks the psychometric function as a function of duration.

subjs = [600 611 620 700];

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

% header = {'trial','category','focal','targetPresent','responsePresent','duration','tp','fa'};
% 
% tp = (adata(:,6)==1) .* (adata(:,7)==1); 
% fa = (adata(:,6)==0) .* (adata(:,7)==1);
% outdata = [(1:size(adata,1))' adata(:,2) adata(:,4) adata(:,6) adata(:,7) adata(:,8) tp fa];
% 
% outdata = outdata(~isnan(adata(:,7)),:);
% 
% csvwriteh(fullfile('~/proj/attfield2/behav/out.csv'),outdata,header);

%% Stack all data at end
adatas{end+1} = [];
for ai = 1:(length(adatas)-1)
    adatas{end} = [adatas{end} ; adatas{ai}];
end
SIDs{end+1} = 'all';

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
h = figure;
vs(isnan(vs)) = 0;

for si = 1:length(adatas)
    subplot(length(adatas),1,si); hold on
    title(SIDs{si});
    
    % distributed
    plot(msdur,squeeze(vs(si,:,1)),'ok');
    vfit1 = fitLog(msdur,squeeze(vs(si,:,1))');
    plot(vfit1.x,vfit1.y,'-k');
    plot(msdur,squeeze(vs(si,:,2)),'ob');
    vfit1 = fitLog(msdur,squeeze(vs(si,:,2))');
    plot(vfit1.x,vfit1.y,'-b');
end
savepdf(h,fullfile('~/proj/attfield2/behav/figures/drift_is_dprime.pdf'));


%% Plot RT
h = figure;

for si = 1:length(adatas)
    subplot(length(adatas),1,si); hold on
    title(SIDs{si});
    
    % distributed
    plot(msdur,squeeze(rts(si,:,1)),'-k');
    plot(msdur,squeeze(rts(si,:,2)),'-b');
end

%% Plot sensitivity

h = figure;

yvalsdp = [0 1 2 3];
yvalsauc = normcdf(yvalsdp./sqrt(2));
yvalsauc = round(yvalsauc*100)/100;
ylabels = arrayfun(@(x,y) sprintf('%1.0f/%0.2f',x,y),yvalsdp,yvalsauc,'UniformOutput',false);

for si = 1:length(adatas)
    subplot(length(adatas),1,si); hold on
    
    dp = squeeze(dps(si,:,:));
    
    title(SIDs{si});
    % distributed
%     errbar(msdur,dp(:,1),dp_ci(:,1),'-k');
    p(1) = plot(msdur,dp(:,1),'o','MarkerFaceColor','k','MarkerEdgeColor','w','MarkerSize',5);
%     fit = fitSatExponential(msdur,dp(:,1));
%     plot(fit.x,fit.y,'--k');
    fitfocal = fitLog(msdur,dp(:,1));
    plot(fitfocal.x,fitfocal.y,'-k');
    disp(sprintf('Focal d''(x) = %1.3f * log(%1.3f * x + 1)',fitfocal.params(1),fitfocal.params(2)));
    % focal
%     errbar(msdur+.002,dp(:,2),dp_ci(:,2),'-b');
    p(2) = plot(msdur+.002,dp(:,2),'o','MarkerFaceColor','b','MarkerEdgeColor','w','MarkerSize',5);
%     fit = fitSatExponential(msdur,dp(:,2));
%     plot(fit.x,fit.y,'--b');
    fitdist = fitLog(msdur,dp(:,2));
    plot(fitdist.x,fitdist.y,'-b');
    disp(sprintf('Distributed d''(x) = %1.3f * log(%1.3f * x + 1)',fitdist.params(1),fitdist.params(2)));

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

dp = squeeze(dps(4,:,:));
dp_ci = squeeze(bootci(1000, @mean, squeeze(dps(1:3,:,:))));
dp_err = squeeze(dp_ci(2,:,:)) - dp;

% distributed
%     errbar(msdur,dp(:,1),dp_ci(:,1),'-k');
fitfocal = fitLog(msdur,dp(:,1));
plot(fitfocal.x,fitfocal.y,'-k','LineWidth',3);
errbar(msdur,dp(:,1),dp_err(:,1),'-k','LineWidth',3);
p(1) = plot(msdur,dp(:,1),'o','MarkerFaceColor','k','MarkerEdgeColor','w','MarkerSize',15);
%     fit = fitSatExponential(msdur,dp(:,1));
%     plot(fit.x,fit.y,'--k');
disp(sprintf('Focal d''(x) = %1.3f * log(%1.3f * x + 1)',fitfocal.params(1),fitfocal.params(2)));
% focal
%     errbar(msdur+.002,dp(:,2),dp_ci(:,2),'-b');
fitdist = fitLog(msdur,dp(:,2));
plot(fitdist.x,fitdist.y,'-r','LineWidth',3);
errbar(msdur,dp(:,2),dp_err(:,2),'-r','LineWidth',3);
p(2) = plot(msdur,dp(:,2),'o','MarkerFaceColor','r','MarkerEdgeColor','w','MarkerSize',15);
%     fit = fitSatExponential(msdur,dp(:,2));
%     plot(fit.x,fit.y,'--b');
disp(sprintf('Distributed d''(x) = %1.3f * log(%1.3f * x + 1)',fitdist.params(1),fitdist.params(2)));

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