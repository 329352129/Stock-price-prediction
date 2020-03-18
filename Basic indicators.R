#Load files by one batch
rm(list=ls());
homedir <- "/home/rhincodon/Documents/undergraduated/Abroad/Research training/data/DIS/2014" ;#Direction contains Original data files
setwd(homedir);
filelist = list.files(pattern="*.RData");
#Loading finishes

#Functino final is to get dataframe of Appendix A.
final<-function(addr){
  load(addr);
  df1<-sorted.canonical.data;
  time_s<-data.frame(df1$TIME);
  time_s<-t(time_s);
  time_s<-as.vector(time_s);
  i<-1;
  lst<-list();
  lst2<-list();
  interval<-5*60
  
  
  for (temp in seq(time_s[1],time_s[length(time_s)],interval)) {
    selectresult=subset(df1,(TIME>=temp)&(TIME<=temp+interval-1)); 
    if(nrow(selectresult)==0){
      next();
    }
    lst[[i]] <-selectresult;  
    i<-i+1;  
  } 
  
  
  Open<-c();
  Close<-c();
  Low<-c();
  High<-c();
  Pvwap<-c();
  MeanPriceD<-c();
  MaxPriceD<-c();
  StdPriceD<-c();
  Trades_num<-c();
  
  Max_Trade_Size<-c();
  Mean_Trade_Size<-c();
  Volume<-c();
  P_Volume<-c();
  N_Volume<-c();
  Neutral_Volume<-c();
  Signal<-c();
  rm(temp);
  a<-1
  
  for(temp in lst){
    

    #P_Open
    t<-temp[1,2];
    Open<-c(Open,t);
    
    #P_Close
    t<-temp[dim(temp)[1],2];
    Close<-c(Close,t);
    
    #P_Low and P_High
    Low<-c(Low,range(temp[,2])[1]);
    High<-c(High,range(temp[,2])[2]);
    
    #Pvwap
    r<-temp[,2];
    l<-temp[,3];
    t<-sum(r*l)/sum(l);
    Pvwap<-c(Pvwap,t);
    
    #MeanPriceD
    t<-temp[1,2]-temp[dim(temp)[1],2]/sum(temp[,4])-1;
    MeanPriceD<-c(MeanPriceD,t);
    
    #MaxPriceD
    t<-diff(range(temp[,2]));
    MaxPriceD<-c(MaxPriceD,t);
    
    #Trades_num
    Trades_num<-c(Trades_num,sum(temp[,4]));
    
    #Max_Trade_size
    Max_Trade_Size<-c(Max_Trade_Size,range(temp[,3])[2]);
    
    #Mean_Trade_Size
    Mean_Trade_Size<-c(Mean_Trade_Size,sum(temp[,3])/sum(temp[,4]));
    
    #Volume
    Volume<-c(Volume,sum(temp[,3]));
    
    #StdPriceD
    n<-sum(temp[,4]);
    t<-temp[1,2]-temp[dim(temp)[1],2]/sum(temp[,4])-1;
    every<-temp[,2][1:length(temp[,2])-1]-temp[,2][2:length(temp[,2])];
    sdpd<-sqrt(sum((every-t)*(every-t))/(n-1));
    StdPriceD<-c(StdPriceD,sdpd);
    
    temp_p<-c();
    temp_n<-c();
    temp_neutral<-c();
    ind<-1
    for (am in every) {
      if(am>0){
        temp_p<-c(temp_p,temp[ind,3]);
      }
      else if(am<0){
        temp_n<-c(temp_n,temp[ind,3]);
      }
      else{
        temp_neutral<-c(temp_neutral,temp[ind,3]);
      }
      ind<-ind+1
    }
    P_Volume<-c(P_Volume,sum(temp_p));
    N_Volume<-c(N_Volume,sum(temp_n));
    Neutral_Volume<-c(Neutral_Volume,sum(temp_neutral));
    
    temp<-Close[[a]]-Open[[a]];
    if(temp>0){
      Signal<-c(Signal,1);
    }
    else{
      Signal<-c(Signal,0);
    }
    a<-a+1;
  }
  
  #Return dataframe
  Final_data.data<-data.frame(
    P_Open=Open,
    P_Close=Close,
    P_Low=Low,
    P_High=High,
    Pvwap=Pvwap,
    MeanPriceD=MeanPriceD,
    MaxPriceD=MaxPriceD,
    StdPriceD=StdPriceD,
    Trades_num=Trades_num,
    Max_Trade_Size=Max_Trade_Size,
    Mean_Trade_Size=Mean_Trade_Size,
    Volume=Volume,
    P_Volume=P_Volume,
    N_Volume=N_Volume,
    Neutral_Volume=Neutral_Volume,
    Signal=Signal,
    stringsAsFactors = FALSE
  )
}
#Function ends


#Main part
err<-1;

for (i in filelist ) {
  
  Appendix_A.df<-final(i);
  Appendix_A.df[is.na(Appendix_A.df)]<-0
  
  save(Appendix_A.df,file = paste('/home/rhincodon/Documents/undergraduated/Abroad/Research training/data/DIS/feature_1/',i)); #Direction to save results files 
  err<-err+1;
  
}

