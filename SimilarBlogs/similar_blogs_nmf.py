import sys

sys.setdefaultencoding('utf8')
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
import _pickle as pkl

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF

def letters_only(astr):
    return astr.isalpha()

raw_posts = pkl.load(open("post_dataset.pkl", "rb"))
posts = []

names = set(names.words())
cv = CountVectorizer(stop_words="english", max_features=500)
lemmatizer = WordNetLemmatizer()

for post in raw_posts:
    posts.append(" ".join([lemmatizer.lemmatize(word.lower()) for word in post.split() if letters_only(word) and word not in names]))


transformed = cv.fit_transform(posts)
#nmf = NMF(n_components=100, )







