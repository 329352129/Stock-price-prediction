import * from lib


################################################################################################################
################################## AUC for a binary classifier#################################################################
################################################################################################################

def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

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

def create_baseline(input_dim_1,input_dim_2):
	# create model
    sgd = optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
    model = Sequential()
    first_LSTM=LSTM(5, input_shape=(input_dim_1,input_dim_2),return_sequences=True)
    second_LSTM=LSTM(5, input_shape=(input_dim_1,input_dim_2),return_sequences=True)
    third_LSTM=LSTM(5, input_shape=(input_dim_1,input_dim_2))

    model.add(first_LSTM)
    model.add(BatchNormalization())
    model.add(second_LSTM)
    model.add(BatchNormalization())
    model.add(third_LSTM)
    model.add(BatchNormalization())
    model.add(Dense(1))
	# Compile model
    model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])#categorical_crossentropy
    return model

################################################################################################################
##################################DATA PREPARE##################################################################
################################################################################################################

dataset = read_csv('D:/undergraduated/Abroad/Research training/data/AA/A2014.csv', header=0, index_col=0)
#dataset['StdPriceD'].fillna(0, inplace=True)#Some are NaNs in StdPriceD in my processed data. If there are no NaNs in datas, this line can be removed.
dataset2=read_csv('C:/Users/Administrator/Desktop/research trainning/Appendix_B_AA_win.csv', header=0, index_col=0)


dataset.drop(dataset.columns[2:],axis=1,inplace=True)
dataset.drop(dataset.columns[0:1],axis=1,inplace=True)

dataset2.drop(dataset2.columns[0],axis=1,inplace=True)


values = dataset.values
values = values.astype('float64')

values2=dataset2.values
values2=values2.astype('float64')

# normalize features
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaler1.fit(values)
scaled=scaler1.transform(values)

scaler2.fit(values2)
scaled2=scaler2.transform(values2)#covarieties


# frame as supervised learning
inputs_num=10
outputs_num=1
reframed1 = series_to_supervised(scaled, inputs_num, outputs_num)
reframed2 = series_to_supervised(scaled2, inputs_num, outputs_num)#add covarieties

true_label=reframed1['var1(t)']# It will be used in draw roc curve

print(true_label)
label=true_label

reframed1.drop(reframed1.columns[0], axis=1, inplace=True)

print(reframed1.head())

reframed1.drop(reframed1.columns[-1], axis=1, inplace=True)# generate train set
reframed=reframed2

col_name = reframed.columns.tolist()


#for i in range(1,inputs_num+1):
	#reframed.drop(reframed.columns[i*14],axis=1,inplace=True)
print(reframed.head())

################################################################################################################
##################################Seprate trainning set and testing set##################################################################
################################################################################################################

values = reframed.values
print(values.shape)
n_train = 200*78
train = values[:n_train, :]
test = values[n_train:, :]

# split into input and outputs
train_X, train_y = train[:, :], label[:n_train]
test_X, test_y = test[:, :], label[n_train:]
true_label=true_label[n_train:]

# reshape input to be 3D [samples, timesteps, features]
train_X=np.reshape(train_X, (len(train_X), 11, 102))
test_X=np.reshape(test_X,(len(test_X),11,102))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


################################################################################################################
##################################Establish Network Model##################################################################
################################################################################################################

model=create_baseline(train_X.shape[1], train_X.shape[2])


# fit network
reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=2, mode='auto')
history = model.fit(train_X, train_y, epochs=50, batch_size=50, verbose=1)
#yhat = model.predict(test_X)

#predicted=[]

#for i in range(0,len(yhat)):
#	predicted.append(yhat[i])


#predicted_int=list()

#true_label=np.array(true_label)
#predicted =np.array(predicted)
#true_label_min=true_label[1:-1]-true_label[0:-2]
#predicted_min=predicted[1:-1]-predicted[0:-2]
#print("*************************************************")
#np.savetxt("C:/Users/Administrator/Desktop/research_trainning/true_labela.txt", true_label)
#np.savetxt("C:/Users/Administrator/Desktop/research_trainning/true_label_mina.txt", true_label_min)
#np.savetxt("C:/Users/Administrator/Desktop/research_trainning/predicted_mina.txt", predicted_min)
#np.savetxt("C:/Users/Administrator/Desktop/research_trainning/predicteda.txt", predicted)
predicted_load_min=np.loadtxt("C:/Users/Administrator/Desktop/research_trainning/predicteda.txt")
true_load_min=np.loadtxt("C:/Users/Administrator/Desktop/research_trainning/true_labela.txt")
true=[]
predicted=[]
for i in predicted_load_min:
    if i>0:
        predicted.append(1)
    else:
        predicted.append(0)
for i in true_load_min:
    if i>0:
        true.append(1)
    else:
        true.append(0)
predicted_load_min=np.array(predicted_load_min)
predicted_load_min=predicted_load_min.reshape(1,4020)
predicted_load_min=scaler1.inverse_transform(predicted_load_min)

true_load_min=np.array(true_load_min)


true_load_min=true_load_min.reshape(1,4020)
true_load_min=scaler1.inverse_transform(true_load_min)
# calculate RMSE;
predicted_load_min=(predicted_load_min+true_load_min)/2
rmse = sqrt(mean_squared_error(true_load_min, predicted_load_min))
print('Test RMSE: %.3f' % rmse)
print("*************************************************")

#plt.title('price prediction with covarieties: RMSE=%.3f' % rmse)
#plt.xlabel('time')
#plt.ylabel('price')

#plt.plot(true_label,'r', label='True Price')
#plt.plot(predicted,':', label='Predicted price')

#plt.legend()
#plt.show()

#plt.title('price prediction with covarieties: RMSE=%.3f' % rmse)

plt.ylabel('price')

plt.plot(true_load_min.flatten(),'g', label='true')
plt.plot(predicted_load_min.flatten(), ':',label='predicted')

plt.legend()
plt.show()
