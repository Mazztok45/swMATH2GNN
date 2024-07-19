import pandas as pd
#from torch_geometric.nn import GCNConv
from torch_geometric.data import HeteroData
import torch
#import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools

articles_df = pd.read_csv('../articles_data/full_df.csv')
print("aarticles: ", articles_df.shape)
articles_dict = articles_df.to_dict()
software_df = pd.read_csv('../data/full_df.csv')
print("software: ", software_df.shape)
software_dict = software_df.to_dict()


def keywords_vocabulary(articles_dict, software_dict):
    software_keywords = []
    articles_keywords = []
    for value in articles_dict['keywords'].values():
        if isinstance(value, str):
            keywords = value.split(";")
            articles_keywords.append(keywords)

    for value in software_dict['keywords'].values():
        if isinstance(value, str):
            keywords = value.split(";")
            software_keywords.append(keywords)

    all_keywords = software_keywords + articles_keywords

    vectorizer = TfidfVectorizer()

    all_features = vectorizer.fit(all_keywords)
    software_features = vectorizer.transform(software_keywords).to_array()
    article_features = vectorizer.transform(articles_keywords).to_array()

    return article_features, software_features


def preprocess_heterodata(articles_dict, software_dict):
    data = HeteroData()

    articles_features, software_features = keywords_vocabulary(articles_dict, software_dict)

    data['articles'].x = articles_features
    data['software'].x = software_features

    print(data)



def build_gnn(articles_dict, software_dict):
    articles_features, software_features = keywords_vocabulary(articles_dict=articles_dict, software_dict=software_dict)


preprocess_heterodata(articles_dict, software_dict)
