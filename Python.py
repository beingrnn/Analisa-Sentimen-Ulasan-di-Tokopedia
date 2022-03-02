import pandas as pd

pip install openpyxl

df = pd.read_excel("../input/hslscrape/hasil_scrape.xlsx",engine='openpyxl')


df.columns = df.iloc[0]
df = df[1:]

df.head()

df.info()

data_tokped = df[['Nama_Produk', 'Akun', 'Ulasan' ,'Rating_komentar']]

data_tokped.head()

data_tokped['Rating_komentarsp'] = data_tokped['Rating_komentar'].str.split(' ').str[1]

data_tokped['rate'] = data_tokped['Rating_komentarsp'].str.replace(r'data-testid="icnGivenRatingFilter','').str.replace(r'\s+', ' ')

data_tokped['rate'] = data_tokped['Rating_komentarsp'].astype(str).str[-2]

data = data_tokped.drop(['Rating_komentar', 'Rating_komentarsp'], axis=1)

data.head()

import nltk
nltk.download('all')


data['Ulasan'] = data['Ulasan'].astype(str)

#Case Folding
data['Ulasan'] = data['Ulasan'].str.lower()

data

#Tokenizing
import string 
import re 
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

def remove_links(text):
    # menghapus tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # menghapus non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # menghapus mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # menghapus URL
    return text.replace("http://", " ").replace("https://", " ")
                
data['Ulasan'] = data['Ulasan'].apply(remove_links)

#menghapus number
def remove_number(text):
    return  re.sub(r"\d+", " ", text)

data['Ulasan'] = data['Ulasan'].apply(remove_number)

#menghapus punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

data['Ulasan'] = data['Ulasan'].apply(remove_punctuation)

# menghapus single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", " ", text)

data['Ulasan'] = data['Ulasan'].apply(remove_singl_char)

# Tokenisasi 
def word_tokenize_wrapper(text):
    return word_tokenize(text)

data['Ulasan_tokenize'] = data['Ulasan'].apply(word_tokenize_wrapper)

# Menghitung Distibusi Persebaran Kata
def freqDist_wrapper(text):
    return FreqDist(text)

Ulasan_fqsist = data['Ulasan_tokenize'].apply(freqDist_wrapper)

print('Frequency Tokens : \n') 
print(Ulasan_fqsist.head().apply(lambda x : x.most_common()))

#Normalisasi
slank_word_dict = {
    "keduakali" : "kedua kali",
    "agak" : "sedikit",
    "pas" : "saat",
    "produkx" : "produknya",
    "sukamkasih" : "suka makasih",
    "gak" : "tidak",
    "cpt" : "cepat",
    "sdh" : "sudah",
    "recommend" : "rekomendasi",
    "bagusseller" : "bagus",
    "bagusssss" : "bagus",
    "bagussssss" : "bagus",
    "baguuussss" : "bagus",
    "cepatmakasih": "cepat",
    "cepatmantappp": "cepat",
    "cepatproduct": "cepat",
    "cepatrecomended": "cepat",
    "cepattoko": "cepat",
    "recomended": "rekomendasi",
    "recommended": "rekomendasi",
    "rekomendasi": "rekomendasi",
    "rekomended": "rekomendasi",
    }

def slank_normalized_term(document):
    return [slank_word_dict[term] if term in slank_word_dict else term for term in document]
    
normalizad_word = pd.read_csv("../input/hslscrape/kamus_alay.csv")

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1] 

def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]
    
data['Ulasan_normalized'] = data['Ulasan_tokenize'].apply(normalized_term).apply(slank_normalized_term)


#Filtering
from nltk.corpus import stopwords
list_stopwords = stopwords.words('indonesian')
#remove stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

data['Ulasan_stop_removed'] = data['Ulasan_normalized'].apply(stopwords_removal)

#Stemming
pip install Sastrawi

pip install swifter

# import Sastrawi package
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter


# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in data['Ulasan_stop_removed']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])
    
print(term_dict)
print("------------------------")


# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

data['Ulasan_Stemmed'] = data['Ulasan_stop_removed'].swifter.apply(get_stemmed_term)

data["Ulasan_clean"] = [' '.join(map(str, l)) for l in data['Ulasan_Stemmed']]


data.to_excel("tokped_text.xlsx")


#Word Distribution
ulasan = ' '.join(str(v) for v in data['Ulasan_clean'])
tokenize_ulasan = word_tokenize(ulasan)
tokenize_ulasan

fqdist = FreqDist(tokenize_ulasan)
fqdist

fqdist.most_common(15)

import matplotlib.pyplot as plt

# plotting
fqdist.plot(10,cumulative=False)
plt.show()

#Labeling
label = []
for index, row in data.iterrows():
    if row["rate"] == 5 or row["rate"] == 4:
        label.append(1)
    else:
        label.append(0)

data["label"] = label

data

data["label"].value_counts()

#Menghitung Kata Dengan TF-IDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
Ulasan = data['Ulasan_clean']
Ulasan.isnull().sum()

Ulasan = Ulasan.fillna('tidak ada komentar')
#untuk menghitung jumlah kata yang telah di steming
cv = CountVectorizer()
term_fit = cv.fit(Ulasan)

print (len(term_fit.vocabulary_))
term_fit.vocabulary_ #mengurutkan berdasarkan urutab abjad kata 

ulasan_tf = Ulasan[1] #memanggil kata pada index ke 1
print (ulasan_tf)

term_frequency = term_fit.transform([ulasan_tf]) #hanya menampilkan hasil document 1
print (term_frequency)

dokumen = term_fit.transform(Ulasan) #hasil perhitungan tf idf dalam 1 doc
tfidf_transformer = TfidfTransformer().fit(dokumen)
print (tfidf_transformer.idf_)

tfidf=tfidf_transformer.transform(term_frequency)
print (tfidf) #hasil manual dengan sistem pyhton

#NLP
data_label = data[["Nama_Produk", "Akun", "Ulasan_clean", "label"]]
data_label["Ulasan_clean"] = data_label["Ulasan_clean"].fillna("tidak ada komentar")

data_label.to_excel("data_label.xlsx")
sentimen_data=pd.value_counts(data_label["label"], sort= True)
sentimen_data.plot(kind= 'bar', color= ["green", "red"])
plt.title('Bar chart')
plt.show()

from wordcloud import WordCloud
#Ulasan Negatif

train_s0 = data_label[data_label["label"] == 0]
train_s0["Ulasan_clean"] = train_s0["Ulasan_clean"].fillna("tidak ada komentar")

all_text_s0 = ' '.join(word for word in train_s0["Ulasan_clean"])
wordcloud = WordCloud(colormap='Reds', width=1000, height=1000, mode='RGBA', background_color='white').generate(all_text_s0)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

#Ulasan Positif

train_s1 = data_label[data_label["label"] == 1]
train_s1["Ulasan_clean"] = train_s1["Ulasan_clean"].fillna("tidak ada komentar")

all_text_s1 = ' '.join(word for word in train_s1["Ulasan_clean"])
wordcloud = WordCloud(colormap='Blues', width=1000, height=1000, mode='RGBA', background_color='white').generate(all_text_s1)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Ulasan Positif")
plt.margins(x=0, y=0)
plt.show()

#Menyiapkan data train dan data test
data_label['Ulasan_clean'] = data_label['Ulasan_clean'].fillna("tidak ada komentar")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_label['Ulasan_clean'], data_label['label'], 
                                                    test_size=0.1, stratify=data_label['label'], random_state=30)
                                                    
#TFIDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(decode_error='replace', encoding='utf-8')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print(X_train.shape)
print(X_test.shape)

X_train = X_train.toarray()
X_test = X_test.toarray()

#Machine Learning
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

#deklarasi metode cross validation
cv_method = RepeatedStratifiedKFold(n_splits=5,  n_repeats=3, random_state=999)
#tuning hyperparameter menggunakan gridsearch

params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
gscv_nb = GridSearchCV(estimator=nb, 
                 param_grid=params_NB, 
                 cv=cv_method,   # use any cross validation technique 
                 verbose=1, 
                 scoring='accuracy') 

#Fitting ke Model
gscv_nb.fit(X_train,y_train)
#mendapatkan hyperparameters terbaik
gscv_nb.best_params_

nb = GaussianNB(var_smoothing=1.0)
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print('--------------------- confusion matrix  ----------------------------')
print(confusion_matrix(y_test, y_pred_nb))
print('--------------------- classification report  ----------------------------')
print(classification_report(y_test, y_pred_nb))
