import pandas as pd
import numpy as np

#importing train data

data = pd.read_csv("path/train.csv")

target = data.target

target = pd.DataFrame(target)

data.drop(["target","id"],axis=1,inplace=True)

#data preprocesssing

bin_3_to_binary = {"T":1,"F":0}

data["bin_3"] = data["bin_3"].replace(bin_3_to_binary)

bin_4_to_binary = {"Y":1,"N":0}

data["bin_4"] = data["bin_4"].replace(bin_4_to_binary)

dummies = pd.DataFrame(data.nom_0)
dummies["nom_1"] = data.nom_1
dummies["nom_2"] = data.nom_2
dummies["nom_3"] = data.nom_3
dummies["nom_4"] = data.nom_4
dummies["month"] = data.month.astype(str)
dummies["day"] = data.day.astype(str)

dummies = pd.get_dummies(dummies)

data.drop(["nom_0","nom_1","nom_2","nom_3","nom_4","nom_5","nom_6","nom_7","nom_8","nom_9","month","day"],axis=1,inplace=True)

data = pd.merge(data,dummies,right_index=True,left_index=True)

replacing_ord_1 = {"Novice":1,"Contributor":2,"Expert":3,"Master":4,"Grandmaster":5}

data["ord_1"] = data["ord_1"].replace(replacing_ord_1)

replacing_ord_2 = {"Freezing":1,"Cold":2,"Warm":3,"Hot":4,"Boiling Hot":5, "Lava Hot":6}

data["ord_2"] = data["ord_2"].replace(replacing_ord_2)



replacing_ord_3 = {"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8,"i":9,"j":10,"k":11,"l":12,"m":13,"n":14,"o":15,"p":16,"q":17,"r":18,"s":19,"t":20,"u":21,"v":22,"w":23,"x":24,"y":25,"z":26,"-":-1,"1":-1}

data["ord_3"] = data["ord_3"].replace(replacing_ord_3)

replacing_ord_4 = {"A":27,"B":28,"C":29,"D":30,"E":31,"F":32,"G":33,"H":34,"I":35,"J":36,"K":37,"L":38,"M":39,"N":40,"O":41,"P":42,"Q":43,"R":44,"S":45,"T":46,"U":47,"V":48,"W":49,"X":50,"Y":51,"Z":52,"-":-1,"1":-1}

data["ord_4"] = data["ord_4"].replace(replacing_ord_4)

data = data.fillna(-1)



ord_5 = data["ord_5"].astype(str)

ord5 = []

for i in ord_5:
	ord5.append(list(i))

ord5 = pd.DataFrame(ord5)


ord5[0] = ord5[0].replace(replacing_ord_4)
ord5[0] = ord5[0].replace(replacing_ord_3)

ord5[1] = ord5[1].replace(replacing_ord_4)
ord5[1] = ord5[1].replace(replacing_ord_3)


data.drop(["ord_5"],axis=1,inplace=True)

data = pd.merge(data,ord5,left_index=True,right_index=True)

#building model


input_data = np.array(data)

labels = np.array(target)



print(np.shape(input_data))

from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,BatchNormalization,Input, Embedding,Flatten
from keras.optimizers import nadam,Adam,RMSprop,SGD


model = Sequential([
	Dense(500, input_dim=(58)),
	BatchNormalization(),

	Dense(500,activation="sigmoid"),
	Dropout(0.3),
	BatchNormalization(),

	Dense(500,activation="sigmoid"),
	Dropout(0.3),
	BatchNormalization(),

	Dense(1,activation="sigmoid"),
	])

model.summary()

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=["accuracy"])

#prediction

model.fit(input_data,labels,batch_size=1000,validation_split=0.1,epochs=1,verbose=1)

model.save("cat_in_dot.h5")



model = load_model("cat_in_dot.h5")

#test data preprocessing

test_data = pd.read_csv("path/test.csv")

identification = test_data.id

test_data.drop(["id"],axis=1,inplace=True)


test_bin_3_to_binary = {"T":1,"F":0}

test_data["bin_3"] = test_data["bin_3"].replace(test_bin_3_to_binary)

test_bin_4_to_binary = {"Y":1,"N":0}

test_data["bin_4"] = test_data["bin_4"].replace(test_bin_4_to_binary)

test_dummies = pd.DataFrame(test_data.nom_0)
test_dummies["nom_1"] = test_data.nom_1
test_dummies["nom_2"] = test_data.nom_2
test_dummies["nom_3"] = test_data.nom_3
test_dummies["nom_4"] = test_data.nom_4
test_dummies["month"] = test_data.month.astype(str)
test_dummies["day"] = test_data.day.astype(str)

test_dummies = pd.get_dummies(test_dummies)

test_data.drop( ["nom_0","nom_1","nom_2","nom_3","nom_4","nom_5","nom_6","nom_7","nom_8","nom_9","month","day"],axis=1,inplace=True)

test_data = pd.merge(test_data,test_dummies,right_index=True,left_index=True)

replacing_test_ord_1 = {"Novice":1,"Contributor":2,"Expert":3,"Master":4,"Grandmaster":5}

test_data["ord_1"] = test_data["ord_1"].replace(replacing_test_ord_1)

replacing_test_ord_2 = {"Freezing":1,"Cold":2,"Warm":3,"Hot":4,"Boiling Hot":5, "Lava Hot":6}

test_data["ord_2"] = test_data["ord_2"].replace(replacing_test_ord_2)

replacing_test_ord_3 = {"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8,"i":9,"j":10,"k":11,"l":12,"m":13,"n":14,"o":15,"p":16,"q":17,"r":18,"s":19,"t":20,"u":21,"v":22,"w":23,"x":24,"y":25,"z":26,"-":-1,"1":-1}

test_data["ord_3"] = test_data["ord_3"].replace(replacing_test_ord_3)

replacing_test_ord_4 = {"A":27,"B":28,"C":29,"D":30,"E":31,"F":32,"G":33,"H":34,"I":35,"J":36,"K":37,"L":38,"M":39,"N":40,"O":41,"P":42,"Q":43,"R":44,"S":45,"T":46,"U":47,"V":48,"W":49,"X":50,"Y":51,"Z":52,"-":-1,"1":-1}

test_data["ord_4"] = test_data["ord_4"].replace(replacing_test_ord_4)

test_data = test_data.fillna(-1)



test_ord_5 = test_data ["ord_5"].astype(str)

test_ord_5_list = []

for i in test_ord_5:
	test_ord_5_list.append(list(i))

test_ord_5 = pd.DataFrame(ord5)


test_ord_5[0] = test_ord_5[0].replace(replacing_test_ord_4)
test_ord_5[0] = test_ord_5[0].replace(replacing_test_ord_4)

test_ord_5[1] = test_ord_5[1].replace(replacing_test_ord_4)
test_ord_5[1] = test_ord_5[1].replace(replacing_test_ord_4)


test_data.drop(["ord_5"],axis=1,inplace=True)

test_data = pd.merge(test_data,ord5,left_index=True,right_index=True)


test_data = np.array(test_data)


results = model.predict_classes(test_data)

for i in results:
	print(i)

submission = pd.DataFrame(identification)

submission["target"] = results



submission.to_csv("path/predictions.csv",index=False)
