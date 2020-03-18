import * from lib
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

def create_baseline(input_dim_1,input_dim_2):
	# create model
    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model = Sequential()
    first_LSTM=LSTM(10, input_shape=(input_dim_1,input_dim_2),return_sequences=True)
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
dataset2=read_csv('D:/undergraduated/Abroad/Research training/data/AA/A2014.csv', header=0, index_col=0)

dataset.drop(dataset.columns[2:16],axis=1,inplace=True)
dataset.drop(dataset.columns[0:1],axis=1,inplace=True)
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
inputs_num=10
outputs_num=1
reframed1 = series_to_supervised(scaled, inputs_num, outputs_num)
reframed2 = series_to_supervised(scaled2, inputs_num, outputs_num)#add covarieties

true_label=reframed1['var1(t)']

print(true_label)
label=true_label
covariety1=reframed2['var1(t)']#covirieties
covariety2=reframed2['var2(t)']
covariety3=reframed2['var3(t)']
covariety4=reframed2['var4(t)']
covariety5=reframed2['var5(t)']
covariety6=reframed2['var6(t)']
reframed1.drop(reframed1.columns[0], axis=1, inplace=True)

print(reframed1.head())

reframed1.drop(reframed1.columns[-1], axis=1, inplace=True)# generate train set
reframed=reframed1

col_name = reframed.columns.tolist()

for i in range(1,534,7):
     col_name.insert(i,'covariety%d' % (i))
     col_name.insert(i,'covariety_two%d' % (i))
     col_name.insert(i,'covariety_three%d' % (i))
     col_name.insert(i,'covariety_four%d' % (i))
     col_name.insert(i,'covariety_five%d' % (i))
     col_name.insert(i,'covariety_six%d' % (i))



#reframed=reframed.reindex(columns=col_name)
for i in range(1,534,7):
     reframed['covariety%d' % (i)]=covariety1
     reframed['covariety_two%d' % (i)]=covariety2
     reframed['covariety_three%d' % (i)]=covariety3
     reframed['covariety_four%d' % (i)]=covariety4
     reframed['covariety_five%d' % (i)]=covariety5
     reframed['covariety_six%d' % (i)]=covariety6

#for i in range(1,inputs_num+1):
	#reframed.drop(reframed.columns[i*14],axis=1,inplace=True)
print(reframed.head())

################################################################################################################
##################################Seprate trainning set and testing set##################################################################
################################################################################################################

values = reframed.values
print(values.shape)
n_train = 110*78
train = values[:n_train, :]
test = values[n_train:, :]

# split into input and outputs
train_X, train_y = train[:, :], label[:n_train]
test_X, test_y = test[:, :], label[n_train:]
true_label=true_label[n_train:]

# reshape input to be 3D [samples, timesteps, features]
train_X=np.reshape(train_X, (len(train_X), 80, 6))
test_X=np.reshape(test_X,(len(test_X),80,6))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


################################################################################################################
##################################Establish Network Model##################################################################
################################################################################################################

model=create_baseline(train_X.shape[1], train_X.shape[2])

# fit network
reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=2, mode='auto')
history = model.fit(train_X, train_y, epochs=25, batch_size=50, verbose=1)
yhat = model.predict(test_X)

predicted=[]

for i in range(0,len(yhat)):
	predicted.append(yhat[i])


predicted_int=list()

true_label=np.array(true_label)-0.2
predicted = np.array(predicted)
true_label_min=true_label[1:-1]-true_label[0:-2]
predicted_min=predicted[1:-1]-predicted[0:-2]
print("*************************************************")
np.savetxt("C:/Users/Administrator/Desktop/research_trainning/true_label.txt", true_label)
np.savetxt("C:/Users/Administrator/Desktop/research_trainning/true_label_min.txt", true_label_min)
np.savetxt("C:/Users/Administrator/Desktop/research_trainning/predicted_min", predicted_min)
np.savetxt("C:/Users/Administrator/Desktop/research_trainning/predicted.txt", predicted)
predicted_load=np.loadtxt("C:/Users/Administrator/Desktop/research_trainning/predicted.txt")
true_load=np.loadtxt("C:/Users/Administrator/Desktop/research_trainning/true_label.txt")+0.2



print("*************************************************")

plt.title('price prediction(AutoReg)')
plt.xlabel('time')
plt.ylabel('price')

plt.plot(true_load,'r', label='True Price')
plt.plot(predicted_load,':', label='Predicted price')

plt.legend()
plt.show()
