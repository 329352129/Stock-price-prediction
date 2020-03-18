import matplotlib.pyplot as plt
import pickle
import numpy as np


IBM_fpr=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_IBM.txt')
IBM_tpr=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_IBM.txt')




KO_fpr=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_KO.txt')
KO_tpr=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_KO.txt')



UTX_fpr=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_UTX.txt')
UTX_tpr=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_UTX.txt')


AA_fpr=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_AA.txt')
AA_tpr=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_AA.txt')


IBM_fpr1=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_IBM1.txt')
IBM_tpr1=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_IBM1.txt')




KO_fpr1=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_KO1.txt')
KO_tpr1=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_KO1.txt')




UTX_fpr1=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_UTX1.txt')
UTX_tpr1=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_UTX1.txt')

AA_fpr1=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_AA1.txt')
AA_tpr1=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_AA1.txt')


AA_fpr2=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_AA_mse.txt')
AA_tpr2=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_AA_mse.txt')

IBM_fpr2=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_IBM_mse.txt')
IBM_tpr2=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_IBM_mse.txt')




KO_fpr2=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_KO_mse.txt')
KO_tpr2=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_KO_mse.txt')




UTX_fpr2=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_UTX_mse.txt')
UTX_tpr2=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_UTX_mse.txt')


IBM_fpr3=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_IBM_best.txt')
IBM_tpr3=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_IBM_best.txt')




KO_fpr3=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_KO_best.txt')
KO_tpr3=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_KO_best.txt')




UTX_fpr3=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_UTX_best.txt')
UTX_tpr3=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_UTX_best.txt')


AA_fpr3=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_AA_best.txt')
AA_tpr3=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_AA_best.txt')

AA_fpr4=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_AA_lstm1.txt')
AA_tpr4=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_AA_lstm1.txt')

AA_fpr5=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_AA_sigmoidmi.txt')
AA_tpr5=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_AA_sigmoidmi.txt')

AA_fpr6=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_AA_mse1.txt')
AA_tpr6=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_AA_mse1.txt')

AA_fpr7=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/fpr_AA_msemi.txt')
AA_tpr7=np.loadtxt('C:/Users/Administrator/Desktop/research_trainning/tpr_AA_msemi.txt')

a='IBM'
#plt.title('ROC')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.plot(IBM_fpr, IBM_tpr,'r',label='IBM-DeepAR(sigmoid),0.611')
plt.plot(IBM_fpr1, IBM_tpr1,'b',label='IBM-LSTM,0.526')
plt.plot(IBM_fpr2, IBM_tpr2,'g',label='IBM-DeepAR(noind),0.580')
plt.plot(IBM_fpr3, IBM_tpr3,'y',label='IBM-DeepAR,0.689')
plt.plot([0,1],[0,1],'k',label='y=x')

plt.legend()
plt.show()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


plt.plot(AA_fpr, AA_tpr,'orange',label='mm-sig-idx,0.577')

plt.plot(AA_fpr2, AA_tpr2,'purple',label='mm-mse,0.611')
plt.plot(AA_fpr7, AA_tpr7,'yellowgreen',label='m1-mse-idx,0.621')
plt.plot(AA_fpr3, AA_tpr3,color='r',label='mm-mse-idx,0.693')
plt.plot(KO_fpr2, KO_tpr2,'b',label='garch,0.629')




plt.plot([0,1],[0,1],'k',linestyle=':',label='y=x')

plt.legend()

plt.show()

plt.plot(KO_fpr, KO_tpr,'r',label='KO-DeepAR(sigmoid),0.599')
plt.plot(KO_fpr1, KO_tpr1,'b',label='KO-LSTM,0.511')
plt.plot(KO_fpr2, KO_tpr2,'g',label='KO-DeepAR(noind),0.63')
plt.plot(KO_fpr3, KO_tpr3,'y',label='KO-DeepAR,0.691')
plt.plot([0,1],[0,1],'k',label='y=x')
plt.legend()

plt.show()


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(UTX_fpr, UTX_tpr,'r',label='UTX-DeepAR(sigmoid),0.609')
plt.plot(UTX_fpr1, UTX_tpr1,'b',label='UTX-LSTM,0.521')
plt.plot(UTX_fpr2, UTX_tpr2,'g',label='UTX-DeepAR(noind),0.601')
plt.plot(UTX_fpr3, UTX_tpr3,'y',label='UTX-DeepAR,0.701')
plt.plot([0,1],[0,1],'k',label='y=x')

plt.legend()
plt.show()
