import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, Birch, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from nltk.stem import PorterStemmer
from stop_words import get_stop_words
import re
from wordcloud import WordCloud
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import tweepy
from pprint import pprint
import warnings
from textblob import TextBlob

def cluster_text_data(data, col, decomposition=False, language='en'):
    print('Started clustering...')
    data = data.copy()
    texts = data[col].dropna()
    X = texts.copy()
    
    clean_text = lambda x: re.sub(r"""[(-,.;@#?!&$]+  \ *  """, " ", x.replace("\n",""), flags=re.VERBOSE) if type(x)==str else ''
    X = X.map(clean_text)
    X = X.map(str.lower)
    X = X.map(PorterStemmer().stem)

    if language == 'tr':
        stop_words = get_stop_words('turkish')
        X = TfidfVectorizer(stop_words=stop_words).fit_transform(X).toarray()
    else:
        X = TfidfVectorizer(stop_words='english').fit_transform(X).toarray()
    if decomposition:
        X = TruncatedSVD(n_components=100).fit_transform(X)
    X = MinMaxScaler().fit_transform(X)

    param_range = range(4,11,1)
    # min_samples = round(0.15*X.shape[0])
    # max_eps = 0.1**(1/X.shape[1])
    # almost_zero = 0.1**50
    # step = (max_eps-almost_zero)/20
    # param_range = np.arange(almost_zero, max_eps+almost_zero, step).tolist()
    models = len(param_range) * [None]
    scores = np.zeros(len(param_range))
    for i, param_val in enumerate(param_range):
        print('Run Param No:', i)
        # models[i] = KMeans(n_clusters=param_val)
        models[i] = Birch(threshold=0.1, n_clusters=param_val)
        # models[i] = DBSCAN(eps=param_val)
        models[i].fit(X)
        if len(list(set(models[i].labels_))) > 1:
            scores[i] = silhouette_score(X,models[i].labels_)
        else:
            scores[i] = -1
    model = models[np.argmax(scores)]
    print('Silhoutte Scores:', scores)
    print('Best Silhoutte Score:', np.max(scores))
    
    clustered_data = pd.DataFrame({'text': texts, 'cluster': model.labels_})
    clustered_data.to_excel('output/clustered_data.xlsx', index=False, engine='openpyxl')
    print('Clustering completed!')

    return clustered_data


def generate_word_clouds_and_summarize(clustered_data, query, stop_words=[]):
    warnings.filterwarnings("error")
    print('Generating word clouds and summarizing text...')
    stop_words = get_stop_words('english') + stop_words
    
    word_clouds = []
    summaries = []
    for cluster in clustered_data['cluster'].unique():
        print('Cluster No:', cluster)
        cluster_full_text = ''.join(clustered_data.loc[clustered_data['cluster']==cluster, 'text'])
        cluster_full_text = cluster_full_text.lower()
        word_cloud = WordCloud(width = 800, height = 800,
                        background_color ='white',
                        stopwords = stop_words,
                        min_font_size = 10).generate(cluster_full_text.replace(query.lower(),''))
        word_cloud.to_file(f'output/wordclouds/wc{cluster}.jpg')
        word_clouds.append(word_cloud)
        parser = PlaintextParser.from_string(cluster_full_text, Tokenizer('english'))
        lsa_summarizer = LsaSummarizer()
        lsa_summary = None
        try:
            lsa_summary = lsa_summarizer(parser.document, 10)
        except UserWarning:
            lsa_summary = f'Cluster{cluster} is not suitable for LSA summary.'
        summary_file = open('output/summaries/summary.txt', 'a')
        summary_file.write(f'\nCluster{cluster} Summary:\n{str(lsa_summary)}\n\n')
        summary_file.close()
        summaries.append(lsa_summary)

    print(f'''{len(word_clouds)} word clouds are generated and 
        cluster texts are summarized, if text is suitable!''')
    warnings.filterwarnings("default")
    return word_clouds, summaries

def show_word_cloud_and_summary(word_cloud, summary):
    print('LSA Summary:')
    pprint(summary)
    plt.figure(figsize = (7, 7), facecolor = None)
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()

def get_tweet_sentiment(tweet):
   # create TextBlob object of passed tweet text
    blob = TextBlob(tweet)
    polarity = []
    for sentence in blob.sentences:
        polarity.append(sentence.sentiment.polarity)   
    polarity = np.mean(polarity)
    # set sentiment
    if polarity > 0:
        print('positive')
    elif polarity == 0:
        print('neutral')
    else:
        print('negative')
    return polarity

def get_an_idea(query, language='en'):
    # Authenticate to Twitter
    auth = tweepy.OAuthHandler("YOUR CONSUMER KEY HERE!", 
        "YOUR CONSUMER SECRET HERE!")
    auth.set_access_token("YOUR ACCES TOKEN HERE!", 
        "YOUR ACCES TOKEN SECRET HERE!")

    api = tweepy.API(auth)

    try:
        api.verify_credentials()
        print("Authentication OK")
        search = api.search(q=query, lang=language, count=200)
    except:
        print("Error during authentication")
    tweets = []
    for tweet in search:
        tweets.append(tweet.text)
        # print(f"{tweet.text}")
    tweets = pd.DataFrame(data={'tweets':tweets})
    # full_text = ' '.join(tweets.review)
    # full_summary = LsaSummarizer()(PlaintextParser.from_string(full_text,Tokenizer('english')).document,5)
    summary_file = open('output/summaries/summary.txt', 'w')
    summary_file.write('') # summary_file.write(f'General LSA Summary:\n{str(full_summary)}\n\n')
    summary_file.close()
    # print('General LSA Summary:')
    # print(full_summary)
    print()
    clustered_tweets = cluster_text_data(tweets, 'tweets', decomposition=True, language=language)
    for i in range(clustered_tweets.shape[0]):
        clustered_tweets.loc[i, 'sentiment'] = get_tweet_sentiment(clustered_tweets.loc[i,'text'])
    cluster_sentiments = {}
    cluster_sentiments['general'] = clustered_tweets.sentiment.describe()
    for cluster in clustered_tweets['cluster'].unique():
        print('CLUSTER', cluster, ':')
        print(clustered_tweets.loc[clustered_tweets['cluster']==cluster, ['sentiment']].describe())
        cluster_sentiments[cluster] = clustered_tweets.loc[clustered_tweets['cluster']==cluster, ['sentiment']].describe()
    summary_file = open('output/summaries/sentiment_analysis.txt', 'w')
    summary_file.write(str(cluster_sentiments))
    summary_file.close()
    word_clouds, summaries = generate_word_clouds_and_summarize(clustered_tweets, query, [])
    for i in range(len(word_clouds)):
        print(f'Cluster {i}:')
        show_word_cloud_and_summary(word_clouds[i], summaries[i])
        print()
    for cluster in clustered_tweets['cluster'].unique():
        print(f"Cluster {cluster} n: {clustered_tweets.loc[clustered_tweets['cluster']==cluster].shape[0]}")


get_an_idea('batman', language='en')
