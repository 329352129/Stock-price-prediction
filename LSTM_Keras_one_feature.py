import * from lib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


################################################################################################################
################################## AUC for a binary classifier#################################################################
################################################################################################################



def auc_v2(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put them all together
	in_out_put = concat(cols, axis=1)
	in_out_put.columns = names
	# drop rows with NaN values

	# Actually there is no NaN in datas but just for safety.
	if dropnan:
		in_out_put.dropna(inplace=True)
	return in_out_put

def create_baseline(input_dim_1,input_dim_2,outputs_dim):
	# create model
    sgd = optimizers.SGD(lr=0.1,decay=0, momentum=0.9, nesterov=True)
    model = Sequential()
    first_LSTM=LSTM(128, input_shape=(input_dim_1,input_dim_2),return_sequences=True)
    second_LSTM=LSTM(64, return_sequences=True)
    third_LSTM=LSTM(32, return_sequences=True)

    model.add(first_LSTM)
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(second_LSTM)
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(third_LSTM)
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(1),input_shape=(101,2)))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=[auc])#categorical_crossentropy
    return model

def cal_auc(y_true,y_pred):
    AUC = auc_v2(y_true,y_pred)
    print('AUC has been prepared')
    sess = tf.Session()
    init_global = tf.global_variables_initializer()
    init_local=tf.local_variables_initializer()
    sess.run(init_global)
    sess.run(init_local)
    AUC_Score=sess.run(AUC)
    print('Auc score has got')
    return AUC_Score

def auroc(y_true,y_pred):
    return tf.py_func(roc_auc_score,(y_true,y_pred),tf.double)
################################################################################################################
##################################DATA PREPARE##################################################################
################################################################################################################
def Create_set(address,feature_num,inputs_num,outputs_num):
    dataset = read_csv('C:/Users/Administrator/Desktop/research_trainning/Appendix_B_AA.csv', header=0, index_col=0)
    #dataset = read_csv('C:/Users/Administrator/Desktop/research_trainning/Appendix_B_all2.csv', header=0, index_col=0)
    #dataset.drop(dataset.columns[0:Label_num],axis=1,inplace=True)
    values = dataset.values
    values = values.astype('float64')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values)
    scaled=scaler.transform(values)
    reframed1 = series_to_supervised(scaled, inputs_num, outputs_num)
    true_label=reframed1.iloc[:,inputs_num*feature_num:]# It will be used in draw roc curve
    label=to_categorical(reframed1.iloc[:,inputs_num*feature_num:],2) #one-hot encoder to generate label set
    return [true_label,label]


dataset = read_csv('C:/Users/Administrator/Desktop/research_trainning/Appendix_B_AA.csv', header=0, index_col=0)
#dataset = read_csv('C:/Users/Administrator/Desktop/research_trainning/Appendix_B_all2.csv', header=0, index_col=0)
#dataset['StdPriceD'].fillna(0, inplace=True)#Some are NaNs in StdPriceD in my processed data. If there are no NaNs in datas, this line can be removed.
dataset2=read_csv('C:/Users/Administrator/Desktop/research_trainning/Appendix_B_AA.csv', header=0, index_col=0)
#dataset2=read_csv('C:/Users/Administrator/Desktop/research_trainning/Appendix_B_all2.csv', header=0, index_col=0)

dataset.drop(dataset.columns[0:15],axis=1,inplace=True)

dataset2.drop(dataset2.columns[10:16],axis=1,inplace=True)
dataset2.drop(dataset2.columns[0:2],axis=1,inplace=True)


values = dataset.values
values = values.astype('float64')
values2=dataset2.values
values2=values2.astype('float64')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(values)
scaled=scaler.transform(values)
scaler.fit(values2)
scaled2=scaler.transform(values2)#covarieties


# frame as supervised learning
inputs_num=100
outputs_num=101
reframed1 = series_to_supervised(scaled, inputs_num, outputs_num)
reframed2 = series_to_supervised(scaled2, inputs_num, outputs_num)#add covarieties
print(reframed2.head())

#set=Create_set('/home/rhincodon/Documents/undergraduated/Abroad/Research training/data/AA/Appendix_B/Appendix_B_2014_new.csv',15,100,78)
true_label=np.array(reframed1.iloc[:,100:])# It will be used in draw roc curve
label=np.array(reframed1.iloc[:,100:]) #one-hot encoder to generate label set
covariety=np.array(reframed2.iloc[:,8:])#

reframed1.drop(reframed1.columns[-100:], axis=1, inplace=True)# generate train set
reframed=reframed1

print(reframed.head())

################################################################################################################
##################################Seprate trainning set and testing set##################################################################
################################################################################################################


values = reframed.values
print(values.shape)
n_train = 200*78
print(len(values))
train_covariety=[]
test_covariety=[]
train = values[210*78:240*78, :]
test = values[n_train:225*78, :]


covariety_train=covariety[210*78:240*78,::]#
covariety_test=covariety[n_train:225*78,::]#1765*78
for i in range(0,len(train)):
    train_covariety.append([])
    for j in range(0,len(train[i])):
        train_covariety[i].append(covariety_train[i,j])

for i in range(0,len(test)):
    test_covariety.append([])
    for j in range(0,len(test[i])):

        test_covariety[i].append(covariety_test[i,j])

train_covariety=np.array(train_covariety)
test_covariety=np.array(test_covariety)
train_x_with_covariety=np.reshape(train_covariety,(len(train_covariety),101,1))
test_x_with_covariety=np.reshape(test_covariety,(len(test_covariety),101,1))

#split into input and outputs
train_X, train_y = train[:, :], label[210*78:240*78]
test_X, test_y = test[:, :], label[n_train:225*78]
train_y=np.reshape(train_y,(len(train_y),101,1))
test_y=np.reshape(test_y,(len(test_y),101,1))
true_label_test=true_label[n_train:225*78]
true_label_train=true_label[210*78:240*78]
# reshape input to be 3D [samples, timesteps, features]
train_X=np.reshape(train_X,(len(train_X), 101,1 ))
test_X=np.reshape(test_X,(len(test_X),101,1))
print(train_x_with_covariety.shape,train_y.shape,test_x_with_covariety.shape,test_y.shape)
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


################################################################################################################
##################################Establish Network Model##################################################################
################################################################################################################



model=create_baseline(test_X.shape[1], test_X.shape[2],78)

# fit network
#reduce_lr = ReduceLROnPlateau(monitor='loss', patience=2, mode='auto')
history = model.fit(train_X,train_y, epochs=30, batch_size=70, verbose=2)
json_file=open('model.json','r')
load_model_json=json_file.read()
json_file.close()
#model=model_from_json(load_model_json)


#weights=model.get_weights()[-1]
#model_json=model.to_json()
#with open('C:\\Users\\Administrator\\Desktop\\research\\model2.json','w') as f:
#    tf.write(model_json)
#print('fitting successed')

#####################################################################################################################33
##################################Rebuild results############################################################
####################################################################################################
predicted_1d=[]
predicted_2d=[]
true=[]
yhat = model.predict(test_X)

for i in range(0,len(yhat)):
    predicted_2d.append([])
    for j in range(0,len(yhat[i])):
        if yhat[i][j]>0.5:
            predicted_1d.append(1)
            predicted_2d[i].append(1)
        else:
            predicted_1d.append(0)
            predicted_2d[i].append(0)
        true.append(true_label_test[i][j])
predicted_with_likelihood=np.zeros(yhat.shape)+0.5
predicted_with_likelihood=yhat+predicted_with_likelihood

true=np.array(true)
predicted_1d=np.array(predicted_1d)
predicted_2d=np.array(predicted_2d)

yhat=yhat.reshape(true_label_test.shape)
#yhat=np.around(yhat,0)
predicted_2d=predicted_2d.reshape(true_label_test.shape)
print(yhat.shape)
print(predicted_2d.shape)
print(true_label.shape)
print('predicting successed')
print(yhat)
AUC = auc_v2(y_true=true_label_test,y_pred=yhat)

print('AUC has been prepared')
sess = tf.Session()
init_global = tf.global_variables_initializer()
init_local=tf.local_variables_initializer()
sess.run(init_global)
sess.run(init_local)
AUC_Score=sess.run(AUC)
print('Auc score has got')

print('AUC: ',AUC_Score)
print("*************************************************")
yhat2=yhat.reshape(true.shape)
fpr, tpr, thresholds = metrics.roc_curve(true, yhat2, pos_label=1)# true_label is label set consists of 0 or 1
np.savetxt('C:/Users/Administrator/Desktop/research_trainning/fpr_AA_garch.txt',fpr)
np.savetxt('C:/Users/Administrator/Desktop/research_trainning/tpr_AA_garch.txt',tpr)
plt.title('ROC with DeepAR (C)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.plot(fpr, tpr,label=AUC_Score)
plt.plot([0,1],[0,1])
plt.legend()
plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), history.history["auc"], label="train_auc")
plt.plot(np.arange(0, 20), history.history["val_auc"], label="val_auc")
plt.title("Training Loss and Accuracy on sar classifier")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.show()
