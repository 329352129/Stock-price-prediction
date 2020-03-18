rm(list=ls());
homedir <- "/home/rhincodon/Documents/undergraduated/Abroad/Research training/data/AA/feature_1" ;#Direction contains Original data files
setwd(homedir);
filelist = list.files(pattern="*.RData");

data<-data.frame()

for (i in filelist ) {
  t<-load(i)
 
  temp<-eval(parse(text = t))
  data<-rbind(data,temp);
}
write.csv(data,"/home/rhincodon/Documents/undergraduated/Abroad/Research training/data/AA/feature_1/Appendix_A_all.csv",row.names = TRUE);

