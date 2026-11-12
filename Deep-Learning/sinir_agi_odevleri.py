"""Fashion-MNIST erken durdurma/2 katman ve KMNIST hiperparametre taraması.
Gereken paketler: tensorflow tensorflow-datasets scikit-learn pandas matplotlib
Not: TensorFlow bu hazırlama ortamında kurulu olmadığından bu bölüm burada eğitilmedi.
"""
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
ROOT=Path(__file__).resolve().parent
keras.utils.set_random_seed(42)

def build_model(hidden=(256,), learning_rate=.001):
    inputs=keras.Input(shape=(28,28))
    x=keras.layers.Flatten()(inputs)
    for i,n in enumerate(hidden):
        x=keras.layers.Dense(n,activation='relu',name=f'Hidden_{i+1}')(x)
    outputs=keras.layers.Dense(10,activation='softmax',name='Output')(x)
    model=keras.Model(inputs,outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

def plot_history(history,stem):
    h=history.history
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,keys,title in [(axes[0],('loss','val_loss'),'Kayıp'),(axes[1],('accuracy','val_accuracy'),'Doğruluk')]:
        for k in keys: ax.plot(range(1,len(h[k])+1),h[k],label=k)
        ax.set(xlabel='Epok',title=title);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/(stem+'.png'),dpi=140);plt.close(fig)

# ÖDEV 2: Fashion MNIST: tek katman erken durdurma, iki katman 20 epok
(x_train,y_train),(x_test,y_test)=keras.datasets.fashion_mnist.load_data()
x_train=x_train.astype('float32')/255.;x_test=x_test.astype('float32')/255.
rows=[]
for max_epochs in [10,20,40]:
    model=build_model((256,))
    early=keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)
    history=model.fit(x_train,y_train,epochs=max_epochs,batch_size=64,validation_split=.2,
                      callbacks=[early],verbose=0)
    loss,acc=model.evaluate(x_test,y_test,verbose=0)
    rows.append({'model':'Fashion tek gizli katman + early stopping','max_epochs':max_epochs,
                 'trained_epochs':len(history.history['loss']),'test_loss':loss,'test_accuracy':acc})
    plot_history(history,f'Fashion_1kat_EarlyStop_{max_epochs}')
model=build_model((256,256));h=model.fit(x_train,y_train,epochs=20,batch_size=64,validation_split=.2,verbose=0)
loss,acc=model.evaluate(x_test,y_test,verbose=0)
rows.append({'model':'Fashion iki gizli katman','max_epochs':20,'trained_epochs':20,'test_loss':loss,'test_accuracy':acc})
plot_history(h,'Fashion_2kat_20epok')
pd.DataFrame(rows).to_csv(ROOT/'Fashion_Sonuclar.csv',index=False)
print('Fashion sonuçları:\n',pd.DataFrame(rows).to_string(index=False))

# ÖDEV 3: KMNIST, ortak test seti, sabit ayrılmış validation, farklı hiperparametreler
train_ds,test_ds=tfds.load('kmnist',split=['train','test'],as_supervised=True,batch_size=-1)
x_full,y_full=tfds.as_numpy(train_ds); x_te,y_te=tfds.as_numpy(test_ds)
x_full=np.squeeze(x_full,-1).astype('float32')/255.
x_te=np.squeeze(x_te,-1).astype('float32')/255.
x_tr,x_val,y_tr,y_val=train_test_split(x_full,y_full,test_size=.2,random_state=42,stratify=y_full)
results=[]
# 2 mimari ve her biri için epoch, batch size, düğüm sayısı seçenekleri
for hidden in [(64,), (128,), (64,32), (128,64)]:
    for epochs in [5,10]:
        for batch_size in [64,128]:
            keras.utils.set_random_seed(42)
            model=build_model(hidden)
            history=model.fit(x_tr,y_tr,validation_data=(x_val,y_val),
                              batch_size=batch_size,epochs=epochs,verbose=0)
            # Seçim doğrulama ile yapılmalı. Testi seçimde kullanmayın.
            results.append({'hidden':str(hidden),'epochs':epochs,'batch_size':batch_size,
                            'val_accuracy':max(history.history['val_accuracy']),
                            'last_val_accuracy':history.history['val_accuracy'][-1]})
            plot_history(history,f'KMNIST_{str(hidden).replace(" ","")}_{epochs}_{batch_size}')
result=pd.DataFrame(results).sort_values('last_val_accuracy',ascending=False)
result.to_csv(ROOT/'KMNIST_Validation_Taramasi.csv',index=False)
best=result.iloc[0]
hidden=tuple(int(x.strip()) for x in best['hidden'].strip('()').split(',') if x.strip())
# Nihai model seçilen hiperparametrelerle yeniden eğitim ve test yalnızca sonda
keras.utils.set_random_seed(42)
final_model=build_model(hidden)
final_model.fit(x_tr,y_tr,validation_data=(x_val,y_val),epochs=int(best['epochs']),batch_size=int(best['batch_size']),verbose=0)
loss,acc=final_model.evaluate(x_te,y_te,verbose=0)
print('KMNIST en iyi validation konfigürasyonu:',best.to_dict())
print('KMNIST nihai test:', {'loss':float(loss),'accuracy':float(acc)})
