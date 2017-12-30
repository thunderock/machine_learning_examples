import sys
reload(sys)
sys.setdefaultencoding('utf8')
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
import cPickle as pkl

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF




