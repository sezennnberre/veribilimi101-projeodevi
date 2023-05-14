# -*- coding: utf-8 -*-
"""
Created on Sat May 13 00:59:38 2023

@author: HP
"""

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn import preprocessing as pre
import math
from sklearn import metrics
import re 
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor



weather = pd.read_csv("weather.csv")

#--------------------EKSİK VERİ SORGULAMASI----------------------------------

# veri setinde nan değerleri olan sütunu bulmak için yazılmış kod 
for n,c in weather.items():
    if is_numeric_dtype(c):
        if weather[n].isnull().sum():
            print(n)
            

#SimpleImputer komutu nan değerlerin yerine ortalamyı yazmamızı sağlar.            
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy= "mean")

#veri setinden istediğimiz yeri çektik.
temp_max = weather.iloc[:,2:3].values

#çektiğimiz kısmı SimpleImputer ile fit edip transform işlemi uyguladık. 
imputer = imputer.fit(temp_max)
temp_max = imputer.transform(temp_max)
print(temp_max)

#missing valuelerin içine sütunnun ortalamsını yazdırmak istiyoduk ve yukarıda bu işlemleri yaptık.
#bu işlemleri veri setimize yansıtmak için LabelEncoder kullanırız.
weather["temp_max"] = pre.LabelEncoder().fit_transform(temp_max)


#----------------------------------------- KATEGORİK DEĞİŞKEN DÖNÜŞÜMÜ---------------------------

# veri setinde ilgilendiğimiz sütunu category tipine çevirdik. sıralamayı kendi istediğim gibi yapmak 
#istedim onu ayarladık.
for _ in weather.items():
    weather["weather"] = weather["weather"].astype('category')
    weather["weather"] = weather["weather"].cat.set_categories(["sun","drizzle","rain","fog","snow"],ordered=True)
print(weather["weather"])

# istediğimiz gibi sıralayım categorylediğimiz veriyi code'a çevirerek k değişkenine atadık.
k= weather["weather"].cat.codes

#label encoder ile k değişkenini veri setinde encode ettiğimiz sütuna entegre ettik.
weather["weather"]= pre.LabelEncoder().fit_transform(k)


#--------------------------------ERROR METRİĞİ KARARI --------------------------------------

# en düşük sıcaklık (temp_min), en yüksek sıcaklık (temp_max) ve rüzgar hızı ile yağış miktarını tahmin edeceğiz.


ort_yags = weather.precipitation.mean()

ort_hata = weather.precipitation - ort_yags


RMSE = np.sqrt(np.square((weather.precipitation) - (ort_yags)).mean())


#İLK BÖLME

filt = ( weather.temp_min <= 15)

bölüm1 = weather[filt]   # 2den küçük olanlar 
bölüm2 = weather[~filt]    #♦ 2den büyük olanlar

ort_bölüm1 = bölüm1.precipitation.mean()
ort_bölüm2 = bölüm2.precipitation.mean()

RMSE1 = np.sqrt(np.square(weather.precipitation - ort_bölüm1) + np.square(weather.precipitation - ort_bölüm2)).mean()

# ikinci bölme 

# ikinci bir bölme yapamadım çünkü hangi şekilde denersem deneyeyim. RMSE1'den daha az sonuca ulaşamadım.
# ikinci bölme yaptığım zaman error hep büyük çıkıyor ne yapma  gerektiğini bilemedim bundan dolayı da en az sonucu bulduğum
#ilk bölmeyi yaparak bıraktım.



# ------------------------- TRAİN - TEST- VALİDATİON -----------------------------

#tahmin edilecek kısım
precipitation = weather.iloc[:,1:2].values
yagıs = pd.DataFrame(data= precipitation, index = range(1461), columns=["precipitation"] )

#tahmin için kullanılacak sütunlar veri datasından ayrı ayrı çekildi.
value = weather.iloc[:,2:].values
predict_values = pd.DataFrame(data = value, index = range(1461), columns= ["temp_max","temp_min","wind","weather"])


#RandomForestRegressor kullanarak
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predict_values,yagıs,test_size=0.33,random_state=0)

#ağaç sayısını arttırarak sonucun nasıl değiştiğini gördük..
m = RandomForestRegressor(n_estimators=1,bootstrap=False, n_jobs=-1)
m.fit(x_train,y_train)
r2 = m.score(x_train, y_train)

m = RandomForestRegressor(n_estimators=10,bootstrap=True, n_jobs=-1)
m.fit(x_train,y_train)
r2_1 = m.score(x_train, y_train)

m = RandomForestRegressor(n_estimators=30,bootstrap=True, n_jobs=-1)
m.fit(x_train,y_train)
r2_2 = m.score(x_train, y_train)

#LinerRegressor kullanarak
from sklearn.linear_model import LinearRegression
l = LinearRegression()
l.fit(x_train,y_train)
y_pred = l.predict(x_test)

# r^2 ile hata hesabı
from sklearn.metrics import r2_score
print("Random Forest R2 degeri")
print(r2_score(y_train,m.predict(x_train)))
print(r2_score(y_test, m.predict(x_test)))
#değerler her çalıştırmada farklılık gösterebilir neden bilmiyorum.

# RMSLE ile hata hesabı       squared false yapınca root oluyor.
from sklearn.metrics import mean_squared_log_error
print("Mean Squared Log Error degeri")
print(mean_squared_log_error(y_train,m.predict(x_train),squared=False))
print(mean_squared_log_error(y_test,m.predict(x_test),squared=False))

#RMSE ile hata hesabı
from sklearn.metrics import mean_squared_error
print("Mean Squared Error degeri")
print(mean_squared_error(y_train, m.predict(x_train),squared=False))
print(mean_squared_error(y_test, m.predict(x_test),squared=False))


#bu error değerleri random_satte = 42 olduğundan yani her döngüde yeniden train ve testler random olarak düzenlendiğinden 
#her döngüde farklı değerler çıkabilir.
#random_state = 0 yaparsak yine aynı şey oldu nasıl engelliycem bilemedim.
  













