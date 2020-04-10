%% Analyze behavioral data from pilot
% Josh and Kai did a pilot test on 2/28. This script loads their data and
% checks the psychometric function as a function of duration.

subjs = [605];

SIDs = {};
for si = 1:length(subjs)
    SIDs{si} = sprintf('s%03.0f',subjs(si));
end

%% Load data files
dataFolder = fullfile('~/data/spatobj/');

adatas = {};
frameRate = [];
for si = 1:length(SIDs)
    sid = SIDs{si};
    subjFolder = fullfile(dataFolder,sid);
    files = dir(fullfile(dataFolder,sid,'*.mat'));
    
    adata = [];
    blockn = 1;
    % COLUMN
    %    1      2       3       4      5        6               7           8        9     10   11   12   13   14 
    % block# trial# targetCat focal image# targetPresent responsePresent duration correct dead cat0 cat1 cat2 cat3 
    
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
                % this is a real block, not the example phase, go through
                % the trials and add them
                cdata = [blockn*ones(cBlock.nTrials,1) (1:cBlock.nTrials)' cBlock.parameter.targetCategory' ...
                    cBlock.parameter.focal' cBlock.parameter.imageNumber' cBlock.parameter.targetPresent' ...
                    cBlock.randVars.responsePresent' cBlock.randVars.duration' cBlock.randVars.correct' ...
                    cBlock.randVars.dead' cBlock.randVars.imgCat0' cBlock.randVars.imgCat1' ...
                    cBlock.randVars.imgCat2' cBlock.randVars.imgCat3'];
                adata = [adata ; cdata];
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

%% Make a plot
h = figure;

dps = [];

for si = 1:length(adatas)
    subplot(length(adatas)+1,1,si); hold on
    
    title(SIDs{si});
    
    adata = adatas{si};
    
    udur = unique(adata(:,8));
    
    % duration x target category x focal condition
    %                               (dist=1, focal=2)
%     pcorr = nan(length(udur),20,2);
    dp = nan(length(udur),2);
    c = nan(length(udur),2);
%     ecorr = nan(length(udur),2);
    for ui = 1:length(udur)
        cdur = udur(ui);
        sdata = sel(adata,8,cdur);
        
%         for tar = 0:19
%             tdata = sel(sdata,3,tar);
            
            for fd = [0 1]
                fdata = sel(sdata,4,fd);
                
                if ~isempty(fdata)
                    % compute d' 
                    hits = sum(fdata(:,6)==1 .* fdata(:,7)==1)/size(fdata,1);
                    fa = sum(fdata(:,6)==0 .* fdata(:,7)==1)/size(fdata,1);
                    dp(ui,fd+1) = norminv(hits)-norminv(fa);
                    c(ui,fd+1) = 0.5*(norminv(hits)+norminv(fa));
%                     if size(fdata,1)>1
%                         ci = bootci(100,@nanmean,fdata(:,9));
%                         ecorr(ui,fd+1) = ci(2);
%                     end
                end
            end
%         end
    end
    
    cs(si,:,:) = c;
    dps(si,:,:) = dp;
    
    msdur = frameRate(si) * udur;
    % distributed
%     errbar(msdur,dp(:,1),ecorr(:,1)-dp(:,1),'-k');
    p(1) = plot(msdur,dp(:,1),'o','MarkerFaceColor','k','MarkerEdgeColor','w','MarkerSize',5);
%     fit = fitSatExponential(msdur,dp(:,1));
%     plot(fit.x,fit.y,'--k');
    fit = fitLog(msdur,dp(:,1));
    plot(fit.x,fit.y,'-k');
    % focal
%     errbar(msdur,dp(:,2),ecorr(:,2)-dp(:,2),'-b');
    p(2) = plot(msdur,dp(:,2),'o','MarkerFaceColor','b','MarkerEdgeColor','w','MarkerSize',5);
%     fit = fitSatExponential(msdur,dp(:,2));
%     plot(fit.x,fit.y,'--b');
    fit = fitLog(msdur,dp(:,2));
    plot(fit.x,fit.y,'-b');
    
    xlabel('Duration (s)');
    ylabel('Sensitivity (d'')');
    
    axis([0 0.3 0 2]);
    legend(fliplr(p),{'Focal','Distributed'},'Location','SouthEast');
    
    drawPublishAxis('figSize=[8.9,12]');
end


%% Add a final subplot with the average across the full dataset
% subplot(length(adatas)+1,1,length(adatas)+1); hold on
% title('Average');
% 
% pcorr_ = squeeze(mean(pcorrs,1));
% % todo with more data 
% % ci = bootci(1000,@nanmean,pcorrs);
% 
% plot(msdur,pcorr_(:,1),'o','MarkerFaceColor','k','MarkerEdgeColor','w');
% plot(msdur,pcorr_(:,2),'o','MarkerFaceColor','b','MarkerEdgeColor','w');
% hline(0.5,'--r');
% 
% axis([0 0.3 0 1]);
% legend(fliplr(p),{'Focal','Distributed'},'Location','SouthEast');
% drawPublishAxis('figSize=[8.9,12]');
% xlabel('Duration (s)');
% ylabel('Proportion correct (%)');

%%

savepdf(h,fullfile('~/proj/attfield2/behav/figures/pilot_2020_03_06.pdf'));