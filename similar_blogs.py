# we are going to take eucledian distance between different posts to find out the nearest.
import os

posts = [open(os.path.join("Data/CountVectorizer", f)).read() for f in os.listdir("Data/CountVectorizer")]
from sklearn.feature_extraction.text import CountVectorizer

# Count vectorizer will count the frequency of words in each blog and then will try to find out similarity between two bags on basis of
# maximum number of common words

vectorizer = CountVectorizer()
# print(vectorizer) # min df will reject all words having frequencies less than 1.

X_train = vectorizer.fit_transform(posts)
num_samples, num_features = X_train.shape

# print (num_samples)
# print (num_features)

#print(vectorizer.get_feature_names())

test_post = "Imaging databases"
test_post_vec = vectorizer.transform([test_post])
# print(test_post_vec)
#print(test_post_vec.toarray())

import scipy as sp

# function to calculate eucleadian distance between two arrays
def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

import sys

best_doc = None

best_dist = sys.maxint
best_i = None
for i in range(0, num_samples):
    post = posts[i]
    if post == test_post:
        continue
    post_vec = X_train.getrow(i) # gives count vector for sample i

    d = dist_raw(post_vec, test_post_vec) # calculating the distance
    print "=== Post %i with dist=%.2f %s"%(i, d, post)
    if d < best_dist:
        best_dist = d
        best_i = i

print("Best post is %i with dist=%.2f"%(best_i, best_dist))




