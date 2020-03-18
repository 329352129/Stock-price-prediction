n#######################################################################################################################
##Data Load##
#######################################################################################################################

rm(list=ls());
setwd("D:/undergraduated/Abroad/Research training/data/AA/feature_1");#direction storing basic features.
filelist = list.files(pattern="*.RData");
df1<-list();
n<-1;

for(i in filelist){
  load(i);
  temp<-Appendix_A.df;
  df1[[n]]<-temp;
  n<-n+1;
}

df<-do.call("rbind",df1);# bind all data files to make one dataframe containing data 


#######################################################################################################################
##Indicator functions##
#######################################################################################################################


#Typical Price
P<-function(P_High,P_Low,P_Close){
  p<-(P_High+P_Low+P_Close)/3;
  p;
}
#Typical Price Ends

# SMA:SMA_Total's parameter is vector while SMA's is element.
SMA_Total<-function(vector,n){
  sma<-c(vector[1:n]);
  
  for(i in (n+1):length(vector)){
    a<-i-n;
    b<-i-1;
    sma<-c(sma,sum(vector[a:b])/n);
  }
  sma;
}

SMA<-function(vector,t,n){
  sma<-c();
  if(t<=n){
    sma<-c(sma,vector[t]);
  }
  else{
    a<-t-n;
    b<-t-1;
    sma<-sum(vector[a:b])/n;
  }
  
  sma;
}
# SMA Ends

#EWMA:EWMA_Total's parameter is vector while EWMA's is element.
EWMA_Total<-function(vector,k){
  ewma<-c();
  for(i in 1:length(vector)){
    if(i==1){
      ewma<-c(ewma,EWMA(vector,i,k));
    }
    else{
      ewma<-c(ewma,k*vector[i]+(1-k)*ewma[(i-1)]);
    }
  }
  ewma;
}

EWMA<-function(vector,t,k){
  if(t==0){
    return(0);
  }
  else{
    return(k*vector[t]+(1-k)*EWMA(vector,t-1,k));
  }
}
#EWMA End

#ADI
ADI<-function(P_Close,P_Low,P_Hight,Volume)
{
  clv<-c();
  adi<-c();
  for(i in 1:length(P_Close)){
    clv<-c(clv,(P_Close[i]-P_Low[i]-P_Hight[i]+P_Close[i])/(P_Hight[i]-P_Low[i]));
  }
  for(i in 1:length(P_Close)){
    adi<-c(adi,sum(clv[i]*Volume[1:i]));
  }
  adi;
} 
#ADI Ends

#BB
BB<-function(P_typical,P_Close,d,n){
  MiddleBB<-SMA_Total(P_typical,n);
  theta<-c();
  temp<-c();
  for (i in 1:n) {
    temp<-c(temp,sum((P_typical[1:i]-MiddleBB[i])*(P_typical[1:i]-MiddleBB[i]))/i)
  }
  for (i in 1:n) {
    theta<-c(theta,sqrt(temp[i]));
  }
  for(i in (n+1):length(P_typical)){
    b<-i-1;
    a<-i-n;
    temp<-c(temp,sum((P_typical[a:b]-MiddleBB[i])*(P_typical[a:b]-MiddleBB[i]))/n);
  }
  for (i in (n+1):length(P_typical)) {
    b<-i-1;
    a<-i-n;
    theta<-c(theta,sqrt(temp[i]));
  }
  UpperBB<-MiddleBB+d*theta;
  LowerBB<-MiddleBB-d*theta;
  b_percent<-(P_Close-LowerBB)/(UpperBB-LowerBB);
  Bandwith<-(UpperBB-LowerBB)/MiddleBB;
  b_percent[1]<-0;
  l<-list(MiddleBB,UpperBB,LowerBB,b_percent,Bandwith);
  l;
}
#BB Ends

#Stochastic Oscillator
SO<-function(P_Close,P_High,P_Low,n,m){
  K<-c();
  for (i in 1:length(P_Close)) {
    if(i<=n){
      K<-c(K,(P_Close[i]-range(P_Low[1:i])[1])/(range(P_High[1:i])[1]-range(P_Low[1:i])[1]));
    }
    else{
      a<-i-n;
      b<-i;
      K<-c(K,(P_Close[i]-range(P_Low[a:b])[1])/(range(P_High[a:b])[1]-range(P_Low[a:b])[1]));
    }
  }
  StochOsci<-100*SMA_Total(K,m);
  StochOsci;
 
}
#Stochastic Oscillator Ends


#Relative Strength Index
RSI<-function(P_Close,n){
  Ut<-c(P_Close[1]);
  Dt<-c(P_Close[1]);
  for (i in 2:length(P_Close)) {
    a<-P_Close[i];
    b<-P_Close[i-1];
    if(a>b){
      Ut<-c(Ut,a-b);
      Dt<-c(Dt,0);
    }
    else{
      Ut<-c(Ut,0);
      Dt<-c(Dt,b-a);
    }
  }
  rsi<-c();
  e_Ut<-c();
  e_Dt<-c();
  t<-c();
  e_Ut<-EWMA_Total(Ut,1/n);
  e_Dt<-EWMA_Total(Dt,1/n);
 for(i in 1:length(P_Close)){
   t<-c(t,1+(e_Ut[i]/e_Dt[i]));
   rsi<-c(rsi,100-100/t[i]);
 }
  rsi;
}
#RSI Ends

#Commodity Channel Index
MAD<-function(vector,n){
  mad<-c();
  b<-c();
  for(i in 1:length(vector)){
   b<-c(b,SMA(vector,i,n));
  }
  for(i in 1:length(vector)){
    if(i<=n){
      mad<-c(mad,sum(abs(vector[1:i]-b[i]))/i);
    }
    else{
      am<-i-n;
      bm<-i-1;
      mad<-c(mad,sum(abs(vector[am:bm]-b[i]))/n);
    }
  }
  
  mad;
}
CCI<-function(P_Typical,n){
  cci<-c();
  mad<-MAD(P_Typical,n);
  sma<-SMA_Total(P_Typical,n);
  cci<-P_Typical-sma;
  cci<-cci/mad;
  cci<-cci/0.015;
  cci[1]<-0;
  cci;
}
#CCI Ends

#Average Directional Moving Index
ADX<-function(P_High,P_Low,P_Close,n){
  UpMove<-c(P_High[1]);
  DownMove<-c(P_Low[1]);
  DM_P<-c();
  DM_N<-c();
  DI_P<-c();
  DI_N<-c();
  TR<-c(max(c(P_High[1]-P_Low[1],abs(P_High[1]-0),abs(P_Low[1]-0))));
  adx<-c();
  for (i in 2:length(P_High)){
    b<-i-1;
    UpMove<-c(UpMove,P_High[i]-P_High[b]);
    DownMove<-c(DownMove,P_Low[b]-P_Low[i]);
  }
  
  for(i in 1:length(P_High)){
    if(UpMove[i]>DownMove[i]&&UpMove[i]>0){
      DM_P<-c(DM_P,UpMove[i]);
      DM_N<-c(DM_N,0);
    }
    else if(DownMove[i]>UpMove[i]&&DownMove[i]>0){
      DM_N<-c(DM_N,DownMove[i]);
      DM_P<-c(DM_P,0);
    }
    else{
      DM_P<-c(DM_P,0);
      DM_N<-c(DM_N,0);
    }
  }
  
  for (i in 2:length(P_Close)) {
    TR<-c(TR,max(P_High[i]-P_Low[i],abs(P_High[i]-P_Close[i-1]),abs(P_Low[i]-P_Close[i-1])));
  }
  DI_P<-100*EWMA_Total(DM_P,1/n)/SMA_Total(TR,n);
  DI_N<-100*EWMA_Total(DM_N,1/n)/SMA_Total(TR,n);
  for (i in 1:length(TR)) {

  }
  
 
    a<-abs(DI_P-DI_N);
    b<-abs(DI_P+DI_N);
 
  adx<-100*EWMA_Total(a/b,1/n);
  adx;
}
#ADX Ends

#Double & Triple Exponentially Smoothed Returns
D_TR_IX<-function(P_Close,P_Open,n){
  Close<-EWMA_Total(P_Close,1/n);
  Close<-EWMA_Total(Close,1/n);
  Open<-EWMA_Total(P_Open,1/n);
  Open<-EWMA_Total(Open,1/n);
  ix<-(Close-Open)/Open;
  ix;
}
#DIX&TRIX Ends

#Moving Average Convergence-Divergence
MACD<-function(P_Typical,n1,n2,n3){
  divergence<-EWMA_Total(P_Typical,1/n1)-EWMA_Total(P_Typical,1/n2);
  macd<-divergence-EWMA_Total(divergence,1/n3);
  macd;
}
#MACD Ends

#Moving Flow Index
MFI<-function(P_Typical,P_Volume,N_Volume,n){
  MF_P<-c();
  MF_N<-c();
  mfi<-c();
  for(i in 1:length(P_Typical)){
    if(i<=n){
      MF_N<-c(MF_N,sum(P_Typical[1:i]*N_Volume[1:i]));
      MF_P<-c(MF_P,sum(P_Typical[1:i]*P_Volume[1:i]));
    }
    else{
      a<-i-n;
      b<-i-1;
      MF_P<-c(MF_P,sum(P_Typical[a:b]*P_Volume[a:b]));
      MF_N<-c(MF_N,sum(P_Typical[a:b]*N_Volume[a:b]));
    }
   
  }
  mfi<-c(mfi,100*MF_P/(MF_P+MF_N));
  mfi;
}
#MFI Ends


#Price Disagreement and Polarity
PDP<-function(V_P,V_N,V_Z,n){
  VP<-c();
  VN<-c();
  VZ<-c();
  for(i in 1:length(V_P)){
    if(i<=n){
      VP<-c(VP,sum(V_P[1:i]));
      VN<-c(VN,sum(V_N[1:i]));
      VZ<-c(VZ,sum(V_Z[1:i]));
    }
    else{
      a<-i-n;
      b<-i-1;
      VP<-c(VP,sum(V_P[a:b]));
      VN<-c(VN,sum(V_N[a:b]));
      VZ<-c(VZ,sum(V_Z[a:b]));
    }
  }
  Disagreement<-abs(1-abs((VP-VN)/(VP+VN)));
  Polarity<-(VP-VN)/VZ;
  DP<-list(Disagreement,Polarity);
  DP;
}
#Price Disagreement and Polarity Ends;

Signal<-function(P_Close,P_Open){
  s<-c();
  temp<-P_Close-P_Open;
  for(i in 1:length(temp)){
    if(temp[i]>0){
      s<-c(s,1);
    }
    else{
      s<-c(s,0);
    }
  }
  s;
}
#######################################################################################################################
##Indicator functions Ends##
##Main function##
#######################################################################################################################

#Parameters illustration: "n" is window size, "m" is m in Stochastic Oscillator, "n1" "n2" "n3" is n1 n2 n3 in Moving Average Convergence-Divergence
Main<-function(n,m,n1,n2,n3){
  
  Typical_P<-P(data.frame(df$P_High)[,1],data.frame(df$P_Low)[,1],data.frame(df$P_Close)[,1]);
  Adi<-ADI(data.frame(df$P_Close)[,1],data.frame(df$P_Low)[,1],data.frame(df$P_High)[,1],data.frame(df$Volume)[,1]);
  StochOsci1<-SO(data.frame(df$P_Close)[,1],data.frame(df$P_High)[,1],data.frame(df$P_Low)[,1],6,2);
  StochOsci2<-SO(data.frame(df$P_Close)[,1],data.frame(df$P_High)[,1],data.frame(df$P_Low)[,1],12,2);
  StochOsci3<-SO(data.frame(df$P_Close)[,1],data.frame(df$P_High)[,1],data.frame(df$P_Low)[,1],36,2);
  StochOsci4<-SO(data.frame(df$P_Close)[,1],data.frame(df$P_High)[,1],data.frame(df$P_Low)[,1],78,2);
  StochOsci5<-SO(data.frame(df$P_Close)[,1],data.frame(df$P_High)[,1],data.frame(df$P_Low)[,1],234,2);
  StochOsci6<-SO(data.frame(df$P_Close)[,1],data.frame(df$P_High)[,1],data.frame(df$P_Low)[,1],6,3);
  StochOsci7<-SO(data.frame(df$P_Close)[,1],data.frame(df$P_High)[,1],data.frame(df$P_Low)[,1],12,3);
  StochOsci8<-SO(data.frame(df$P_Close)[,1],data.frame(df$P_High)[,1],data.frame(df$P_Low)[,1],36,3);
  StochOsci9<-SO(data.frame(df$P_Close)[,1],data.frame(df$P_High)[,1],data.frame(df$P_Low)[,1],78,3);
  StochOsci10<-SO(data.frame(df$P_Close)[,1],data.frame(df$P_High)[,1],data.frame(df$P_Low)[,1],234,3);
  rsi1<-RSI(data.frame(df$P_Close)[,1],6);
  rsi2<-RSI(data.frame(df$P_Close)[,1],12);
  rsi3<-RSI(data.frame(df$P_Close)[,1],36);
  rsi4<-RSI(data.frame(df$P_Close)[,1],78);
  rsi5<-RSI(data.frame(df$P_Close)[,1],234);
  cci1<-CCI(Typical_P,6);
  cci2<-CCI(Typical_P,12);
  cci3<-CCI(Typical_P,36);
  cci4<-CCI(Typical_P,78);
  cci5<-CCI(Typical_P,234);
  adx1<-ADX(data.frame(df$P_High)[,1],data.frame(df$P_Low)[,1],data.frame(df$P_Close)[,1],6);
  adx2<-ADX(data.frame(df$P_High)[,1],data.frame(df$P_Low)[,1],data.frame(df$P_Close)[,1],12);
  adx3<-ADX(data.frame(df$P_High)[,1],data.frame(df$P_Low)[,1],data.frame(df$P_Close)[,1],36);
  adx4<-ADX(data.frame(df$P_High)[,1],data.frame(df$P_Low)[,1],data.frame(df$P_Close)[,1],78);
  adx5<-ADX(data.frame(df$P_High)[,1],data.frame(df$P_Low)[,1],data.frame(df$P_Close)[,1],234);
  dix1<-D_TR_IX(data.frame(df$P_Close)[,1],data.frame(df$P_Open)[,1],6);
  dix2<-D_TR_IX(data.frame(df$P_Close)[,1],data.frame(df$P_Open)[,1],12);
  dix3<-D_TR_IX(data.frame(df$P_Close)[,1],data.frame(df$P_Open)[,1],36);
  dix4<-D_TR_IX(data.frame(df$P_Close)[,1],data.frame(df$P_Open)[,1],78);
  dix5<-D_TR_IX(data.frame(df$P_Close)[,1],data.frame(df$P_Open)[,1],234);
  trix1<-D_TR_IX(data.frame(df$P_Close)[,1],data.frame(df$P_Open)[,1],6);
  trix2<-D_TR_IX(data.frame(df$P_Close)[,1],data.frame(df$P_Open)[,1],12);
  trix3<-D_TR_IX(data.frame(df$P_Close)[,1],data.frame(df$P_Open)[,1],36);
  trix4<-D_TR_IX(data.frame(df$P_Close)[,1],data.frame(df$P_Open)[,1],78);
  trix5<-dix5;

  macd1<-MACD(Typical_P,3,6,2);
  macd2<-MACD(Typical_P,6,12,4);
  macd3<-MACD(Typical_P,18,36,12);
  macd4<-MACD(Typical_P,36,78,26);
  macd5<-MACD(Typical_P,117,234,78);
  mfi1<-MFI(Typical_P,data.frame(df$P_Volume)[,1],data.frame(df$N_Volume)[,1],6);
  mfi2<-MFI(Typical_P,data.frame(df$P_Volume)[,1],data.frame(df$N_Volume)[,1],12);
  mfi3<-MFI(Typical_P,data.frame(df$P_Volume)[,1],data.frame(df$N_Volume)[,1],36);
  mfi4<-MFI(Typical_P,data.frame(df$P_Volume)[,1],data.frame(df$N_Volume)[,1],78);
  mfi5<-MFI(Typical_P,data.frame(df$P_Volume)[,1],data.frame(df$N_Volume)[,1],234);
  pdp1<-PDP(data.frame(df$P_Volume)[,1],data.frame(df$N_Volume)[,1],data.frame(df$Neutral_Volume)[,1],6);
  pdp2<-PDP(data.frame(df$P_Volume)[,1],data.frame(df$N_Volume)[,1],data.frame(df$Neutral_Volume)[,1],12);
  pdp3<-PDP(data.frame(df$P_Volume)[,1],data.frame(df$N_Volume)[,1],data.frame(df$Neutral_Volume)[,1],36);
  pdp4<-PDP(data.frame(df$P_Volume)[,1],data.frame(df$N_Volume)[,1],data.frame(df$Neutral_Volume)[,1],78);
  pdp5<-PDP(data.frame(df$P_Volume)[,1],data.frame(df$N_Volume)[,1],data.frame(df$Neutral_Volume)[,1],234);
  bb1<-BB(Typical_P,data.frame(df$P_Close)[,1],3,6);
  bb2<-BB(Typical_P,data.frame(df$P_Close)[,1],3,12);
  bb3<-BB(Typical_P,data.frame(df$P_Close)[,1],3,36);
  bb4<-BB(Typical_P,data.frame(df$P_Close)[,1],3,78);
  bb5<-BB(Typical_P,data.frame(df$P_Close)[,1],3,234);
  signal<-Signal(data.frame(df$P_Close)[,1],data.frame(df$P_Open)[,1])
  middleBB1=bb1[1];
  lowerBB1=bb1[3];
  upperBB1=bb1[2];
  bandwith1=bb1[5];
  middleBB2=bb2[1];
  lowerBB2=bb2[3];
  upperBB2=bb2[2];
  bandwith2=bb2[5];
  middleBB3=bb3[1];
  lowerBB3=bb3[3];
  upperBB3=bb3[2];
  bandwith3=bb4[5];
  middleBB4=bb4[1];
  lowerBB4=bb4[3];
  upperBB4=bb4[2];
  bandwith4=bb4[5];
  middleBB5=bb5[1];
  lowerBB5=bb5[3];
  upperBB5=bb5[2];
  bandwith5=bb5[5];
  
  AppendixB.data<-data.frame(
   Price_Typical=Typical_P,
   ADI=Adi,
   StochOsci1=StochOsci1,
   StochOsci2=StochOsci2,
   StochOsci3=StochOsci3,
   StochOsci4=StochOsci4,
   StochOsci5=StochOsci5,
   StochOsci6=StochOsci6,
   StochOsci7=StochOsci7,
   StochOsci8=StochOsci8,
   StochOsci9=StochOsci9,
   StochOsci10=StochOsci10,
   RSI1=rsi1,
   RSI2=rsi2,
   RSI3=rsi3,
   RSI4=rsi4,
   RSI5=rsi5,
   CCI1=cci1,
   CCI2=cci2,
   CCI3=cci3,
   CCI4=cci4,
   CCI5=cci5,
   ADX1=adx1,
   ADX2=adx2,
   ADX3=adx3,
   ADX4=adx4,
   ADX5=adx5,
   DIX1=dix1,
   DIX2=dix2,
   DIX3=dix3,
   DIX4=dix4,
   DIX5=dix5,
   TRIX1=trix1,
   TRIX2=trix2,
   TRIX3=trix3,
   TRIX4=trix4,
   TRIX5=trix5,
   MACD1=macd1,
   MACD2=macd2,
   MACD3=macd3,
   MACD4=macd4,
   MACD5=macd5,
   MFI1=mfi1,
   MFI2=mfi2,
   MFI3=mfi3,
   MFI4=mfi4,
   MFI5=mfi5,
   
   PDP1=pdp1,
   PDP2=pdp2,
   PDP3=pdp3,
   PDP4=pdp4,
   PDP5=pdp5,
  
   bb1<-BB(Typical_P,data.frame(df$P_Close)[,1],3,6),
   bb2<-BB(Typical_P,data.frame(df$P_Close)[,1],3,12),
   bb3<-BB(Typical_P,data.frame(df$P_Close)[,1],3,36),
   bb4<-BB(Typical_P,data.frame(df$P_Close)[,1],3,78),
   bb5<-BB(Typical_P,data.frame(df$P_Close)[,1],3,234),
   signal<-Signal(data.frame(df$P_Close)[,1],data.frame(df$P_Open)[,1]),
   middleBB1=bb1[1],
   lowerBB1=bb1[3],
   upperBB1=bb1[2],
   bandwith1=bb1[5],
   middleBB2=bb2[1],
   lowerBB2=bb2[3],
   upperBB2=bb2[2],
   bandwith2=bb2[5],
   middleBB3=bb3[1],
   lowerBB3=bb3[3],
   upperBB3=bb3[2],
   bandwith3=bb4[5],
   middleBB4=bb4[1],
   lowerBB4=bb4[3],
   upperBB4=bb4[2],
   bandwith4=bb4[5],
   middleBB5=bb5[1],
   lowerBB5=bb5[3],
   upperBB5=bb5[2],
   bandwith5=bb5[5],
   stringsAsFactors = FALSE
  )
}

#######################################################################################################################
##Main functions Ends##
#######################################################################################################################

appendixB.df<-Main(78,2,36,78,26);
appendixB2.df=as.data.frame((lapply(appendixB.df,as.numeric)))
write.csv(appendixB2.df,"D:/undergraduated/Abroad/Research training/data/AA/Appendix_B_AA_win.csv",row.names = TRUE);


