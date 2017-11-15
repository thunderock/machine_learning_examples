# we are going to take eucledian distance between different posts to find out the nearest.

import scipy as sp


# function to calculate eucleadian distance between two arrays
def dist_raw(v1, v2):
    normalized_v1 = v1 / sp.linalg.norm(v1.toarray())
    normalized_v2 = v2 / sp.linalg.norm(v2.toarray())
    v1 = normalized_v1
    v2 = normalized_v2
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())


# term - frequency inverse document frequency
def tfidf(term, doc, docset):
    tf = float(doc.count(term))/sum(doc.count(w) for w in docset)
    # frequency of this word in this document / frequency of word in all documents
    idf = math.log(float(len(docset)) / (len([doc for doc in docset if term in doc])))
    # total number of documents / total number of docs with this term
    return tf * idf


import nltk.stem
english_stemmer = nltk.stem.SnowballStemmer('english')

from sklearn.feature_extraction.text import TfidfVectorizer
# Stemmer derived from TfidfVectorizer which has tfidf built in
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyser = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (
            english_stemmer.stem(w) for w in analyser(doc)
        )


import os, fnmatch

# loading datasets from 20 news
total_docs = []
for root, dirnames, filenames in os.walk("../Data/20_newsgroups"):
    for filename in fnmatch.filter(filenames, "*"):
        total_docs.append(os.path.join(root, filename))

posts = [open(os.path.join(f)).read() for f in total_docs]

#from sklearn.feature_extraction.text import CountVectorizer

# Count vectorizer will count the frequency of words in each blog and then will try to find out similarity between two bags on basis of
# maximum number of common words

# print(vectorizer) # min df will reject all words having frequencies less than 10.

vectorizer = StemmedTfidfVectorizer(min_df=10, stop_words='english', max_df=0.5, decode_error='ignore')
X_train = vectorizer.fit_transform(posts)
num_samples, num_features = X_train.shape

print (num_samples)
print (num_features)

#print(vectorizer.get_feature_names())

test_post = "To center the data (make it have zero mean and unit standard error), you subtract the mean and then divide the result by the standard deviation. x′=x−μσ You do that on the training set of data. But then you have to apply the same transformation to your testing set (e.g. in cross-validation), or to newly obtained examples before forecast. But you have to use the same two parameters μ and σ \
(values) that you used for centering the training set. Hence, every sklearns fit() just calculates the parameters (e.g. μ and σ in case of StandardScaler) and saves them as an internal objects state. Afterwards, you can call its transform() method to apply the transformation to a particular set of examples. fit_transform() joins these two steps and is used for the initial fitting of parameters on the training set x" \
            ", but it also returns a transformed x′. Internally, it just calls first fit() and then transform() on the same data."
test_post_vec = vectorizer.transform([test_post])
# print(test_post_vec)
#print(test_post_vec.toarray())

# using k means to divide posts into clusters
num_clusters = 50
from sklearn.cluster import KMeans
km = KMeans(n_clusters=num_clusters, init='random', n_init=1, verbose=1)
km.fit(X_train)
# this will give integer labels for all those clusters
print(km.labels_)
# this will return the integer value of the cluster to which the data belongs
nearest_cluster_index = km.predict(test_post_vec)
import numpy as np
nearest_posts = np.where(km.labels_ == nearest_cluster_index)
# code to use eucleadean distance and then calculating resembelence between posts
import sys
best_doc = None
best_dist = sys.maxint
best_i = None
for i in range(0, len(nearest_posts)):
    post_vec = X_train.getrow(i)  # gives count vector for sample i

    d = dist_raw(post_vec, test_post_vec)  # calculating the distance
    print "=== Post %i with dist=%.2f "%(i, d)
    if d < best_dist:
        best_dist = d
        best_i = i

print("Best post is %i with dist=%.2f"%(best_i, best_dist))

