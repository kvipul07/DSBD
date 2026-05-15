import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

file = open("sample.txt", "r")

text = file.read()

print("Original Text:")
print(text)

tokens = word_tokenize(text)

print("\nTokens:")
print(tokens)

pos = nltk.pos_tag(tokens)

print("\nPOS Tags:")
print(pos)

stop_words = stopwords.words("english")

filtered = [word for word in tokens if word.lower() not in stop_words]

print("\nAfter Stopword Removal:")
print(filtered)

ps = PorterStemmer()

stemmed = [ps.stem(word) for word in filtered]

print("\nStemmed Words:")
print(stemmed)

lm = WordNetLemmatizer()

lemmatized = [lm.lemmatize(word) for word in filtered]

print("\nLemmatized Words:")
print(lemmatized)

tfidf = TfidfVectorizer()

result = tfidf.fit_transform([text])

print("\nTF-IDF Values:")
print(result.toarray())
