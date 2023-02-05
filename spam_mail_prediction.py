#!/usr/bin/env python
# coding: utf-8

# # Kütüphane tanımlama (Identify library)

# In[6]:


import pandas as pd 
#Dizilerle çalışmak için kullanılır.Ayrıca doğrusal cebir ve matrisler alanında gerekli işlevlere sahiptir.

import numpy as np 
#Veri işlemesi ve analizi için Python programlama dilinde yazılmış olan bir kütüphanedir. 

from sklearn.model_selection import train_test_split 
# Sklearn-> Makine öğrenmesi modelleri oluşturmak için kullanılır.
"""
Burda train_test_split işleminden kısaca bahsedicek olursak; dizilerimizi veya matrislerimizi rastgele subsetlere 
ve test datalarına dönüştürüyoruz.
Sklearn ile test datası oluşturup ,modelimizi oluşturup tahmin üretme aşamasına geçeriz.
"""
from sklearn.feature_extraction.text import TfidfVectorizer 
"""
Kodlama adımında dokümanlarımızı bir TFxIDF özellik matrisine dönüştürebilmek için TfidfVectorizer sınıfını kullanacağız.
Metinsel verileri rakamsal değerlere çevirip, makine öğrenmesinin anlamasını sağlayabiliriz.
Eğer sadece metinsel veri ekler isek makine öğrenmesi anlayamaz. 
Bu sebepten ötürü metinsel veriyi vektörele yani rakamsal veriye çevirmeliyiz.
"""
from sklearn.linear_model import LogisticRegression
"""
Logistic Regression ( Lojistik Regresyon ) sınıflandırma işlemi yapmaya yarayan bir regresyon yöntemidir. Kategorik veya 
sayısal verilerin sınıflandırılmasında kullanılır. Bağımlı değişkenin yani sonucun sadece 2 farklı değer alabilmesi durumda
çalışır. ( Evet / Hayır, Erkek / Kadın, Şişman / Zayıf vs. )

Doğrusal sınıflandırma problemlerinde yaygın bir biçimde kullanılır. Bu sebeple Linear Regression ile çok benzemektedir.
"""
from sklearn.metrics import accuracy_score 

#sklearn.metrics}Regresyon,sınıflandırma ve kümeleme problemlerinde elde ettiğiniz sonuçları değerlendirmek için kullanılır.

# accuracy_score -> Doğruluk sınıflandırma puanını ortaya koyar.


# ## Veri Toplama & Ön İşlem (Data Collection & Pre-Processing)

# In[7]:


# csv dosyasında var olan veri pandas dataframe 'e yükleniyor
raw_mail_data = pd.read_csv(r'C:\Python\mail_data.csv')


# In[8]:


# boş değerleri(null values) boş stringler(null string) ile değiştirme işlemi 
# boş değer gördüğümüz yerde onu bir boş string ile değiştirmek istiyoruz.
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[9]:


# veriyi ekrana yazdırıyoruz.

mail_data

# Burada var olan ; input = message , output(target) = category 


# In[10]:


# ilk 5 veriyi ekrana yazdırıyoruz
mail_data.head()


# In[11]:


# veri tablosunun kaç satır ve sütundan oluştuğunu yazıyoruz.
mail_data.shape


# ## Etiket kodlaması (Label Encoding)

# In[12]:


# spam maili 0 olarak , spam olmayan maili 1 olarak ;
mail_data.loc[mail_data['Category'] == 'spam','Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham','Category',] = 1


# spam =0 
# 
# ham = 1

# In[13]:


#  verileri metin ve etiket olarak ayırma 

# input değeri X , output değeri Y olarak alabiliriz.

x = mail_data['Message']

y = mail_data['Category']


# In[14]:


print(x)


# In[15]:


print(y)


# ### Veriyi eğitim datası ve test datası olarak ayırma (Splitting the data into training data & test data )

# In[16]:


"""
***Çalışmamızda ki en önemli kısımlardan birisi olduğunu söyleyebiliriz. Tabi her ML projesindede durum böyledir.***

Şimdi burada x ve y 'training data' ve 'test data' olarak ayrılacak.
Burada verinin bir bölümünü eğitmek(training model) için kullanılırken, bir bölümü değerlendirmek (test model) için kullanılır. 
"""

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2, random_state=3)

# x_train = train data messages || y_train = label for train data 
# x_test = test data messages || y_test = label for test data
 
# test_size=0.2 } Var olan verimizin %20 si test data olar ak kullanılacak.
# random_state parametresi, sizin durumunuzda verilerin eğitim ve test endekslerine bölünmesine karar verecek olan dahili rasgele sayı üretecini başlatmak için kullanılır.


# In[17]:


print(x.shape) # kaç adet mesaj olduğunu gösterir.
print(x_train.shape) # kaç adet mesajın 'train data' olduğunu gösterir. -> ( train_size = 0.8 )
print(x_test.shape) # kaç adet mesajın 'test data' olduğunu gösterir.  -> ( test_size = 0.2 )  
print(x_train) # 'train data' olan mesajları gösterir.
print(x_test) # 'test data' olan mesajarı olduğunu gösterir.


# ## Özellik Çıkarma (Feature Extraction)

# In[18]:


# metin verilerini, lojistik regresyona 'input' olarak kullanılabilecek özellik vektörlerine dönüştürme
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True') 

# min_df=1 } Eğer kelimenin skoru (yani tekrar sayısı) 1'den az ise onu görmezden gel,kelime skoru 1'den fazla ise onu dahil et.
# Eğer kelime tekrarlamıyor ise (1 defa veya daha az) bunu kullanmak istemeyiz.Çünkü tahmin edebilmek için yeterli değildir.

# stop_words='english' } Çokça kez tekrarlayan resmi kelimeler (as,the,is,...) Anlamsız olan bu kelimeler kaldırılır.
# Data içerisindeki bu tarz kelimeleri görmezden gelmek isteriz. Gereksizdir.Çünkü tahmin için yeterli değildir.

# lowercase = 'True' } Var olan tüm harfleri küçük harfli yapar.

# Elde edilen bu vektörü veriyi dönüştürmek için fonksiyon olarak kullanmamız gerekir. ( x_train , x_test )

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)
# 2. kodda modelimizin 'x_test' te uygunluğunu görmek istemediğimiz için 'fit' kullanmadık.


# Var olan tüm mailleri eğer uygunsa dönüştürme işlemi yapar. 
"""
3 adımda ne yaptığımızı inceleyelim.

1.adım-> Veriyi (training data) vektörleştiriciye uyumlu hale getir. 
2.adım-> Bu vektörleştiriciyi kullanarak veriyi (x_train) dönüştür.
3.adım-> Aynı vektörleştiriciyi kullanarak veriyi(x_test) dönüştür.
"""

# outputları (y_train ve y_text) değerlerini 'integer' değere çevirme
 
y_train = y_train.astype('int')
y_test = y_test.astype('int')


# In[19]:


print(x_train_features)


# ## Modeli Eğitme (Training the Model)

# ### Logistic Regression

# In[20]:


"""
Lojistik regresyon, bir hedef değişkenin olasılığını tahmin etmek için kullanılan denetimli bir öğrenme sınıflandırma 
algoritmasıdır. Hedef veya bağımlı değişkenin doğası ikilidir, bu da yalnızca iki olası sınıf olacağı anlamına gelir.
"""
model = LogisticRegression()


# In[21]:


# Training data ile Lojistik Regresyonu eğitmek

model.fit(x_train_features,y_train)


# ### Eğitilen modelin değerlendirilmesi (Evaluating the trained model)

# In[22]:


# Training data üzerinde tahmin

prediction_on_training_data = model.predict(x_train_features)

# Geliştirdiğimiz model ile tahmin ettiğimiz sonucu kıyaslıyoruz.

accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)

# accuracy_score -> Doğruluk sınıflandırma puanını ortaya koyar.


# In[23]:


print('traning data nın doğruluğu : ', accuracy_on_training_data)


# Doğruluk tahmini %96 oranındadır.
# 
# Yani 100 tahmin üzerinden 96 başarı model için iyi bir sonuçtur.
# 
# 

# In[24]:


# Test data üzerinde tahmin

prediction_on_test_data = model.predict(x_test_features)

accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)

print('test data nın doğruluğu : ', accuracy_on_test_data)


# Gördüğümüz üzere 'training data' ve 'test data' üzerinden yapılan tahminler üzerinde büyük bir fark yok. Bu durum bizim 
# geliştirdiğimiz modelimizin iyi olduğunu gösterir.
# 
# Bunu bir nevi sınava giren öğrencinin soruları çalıştığı yerden değilde çalıştığı yerlere benzer yerlerden çıkınca da sınavdan 
# iyi bir sonuç almasına benzetebiliriz. Bu onun ezber yapmadığı işin mantığını kavradığı yani konuyu öğrendiği anlamına gelir.

# ## Tahmin Sistemi İnşa Etme (Building a Predictive System )

# In[25]:


# yeni bir mail de modelimizi deneyelim.

input_mail = [",Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
# Mail'i vektörize et ve rakamsal verilere çevir.

input_data_features = feature_extraction.transform(input_mail)

# tahmin etme

prediction = model.predict(input_data_features)

if (prediction[0] == 1):
    print('ham mail')
    
else:
    print('spam mail')


# In[27]:


input_mail = ["Eh u remember how 2 spell his name... Yes i did. He v naughty make until i v wet."]
# Mail'i vektörize et ve rakamsal verilere çevir.

input_data_features = feature_extraction.transform(input_mail)

# tahmin etme

prediction = model.predict(input_data_features)

if (prediction[0] == 1):
    print('ham mail')
    
else:
    print('spam mail')


# # Burada eğittiğimiz modelimizi yeni tahminler üzerinde denedik ve elde ettiğimiz sonuç başarılıdır.
