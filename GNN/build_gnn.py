import pandas as pd
#from torch_geometric.nn import GCNConv
#import torch
#import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer
import itertools


articles_df = pd.read_csv('../articles_data/full_df.csv')
articles_dict = articles_df.to_dict()
software_df = pd.read_csv('../data/full_df.csv')
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

    flattened_articles = list(itertools.chain(*articles_keywords))
    flattened_software = list(itertools.chain(*software_keywords))

    all_keywords = flattened_articles + flattened_software
    mlb = MultiLabelBinarizer()

    # Fit the binarizer to all keywords
    mlb.fit(all_keywords)
    software_features = mlb.transform(software_keywords)
    articles_features = mlb.transform(articles_keywords)

    print("Software Features:\n", software_features)
    print("Paper Features:\n", articles_features)
    return articles_keywords


articles = keywords_vocabulary(articles_dict=articles_dict, software_dict=software_dict)