import pandas as pd
from torch_geometric.nn import GCNConv
from torch_geometric.data import HeteroData
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import re

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
            if value.startswith("zbMATH Open Web Interface contents unavailable due to conflicting licenses."):
                value = "not_available"
            else:
                value = value.replace(';', ' ')
            articles_keywords.append(value)

    for value in software_dict['keywords'].values():
        if isinstance(value, str):
            if value.startswith("zbMATH Open Web Interface contents unavailable due to conflicting licenses."):
                value = "not_available"
            else:
                value = value.replace(';', ' ')
            software_keywords.append(value)
        else:
            value = "nan"
            software_keywords.append(value)

    all_keywords = software_keywords + articles_keywords

    vectorizer = TfidfVectorizer(max_features=17500)

    all_features = vectorizer.fit(all_keywords)
    software_features = vectorizer.transform(software_keywords).toarray()
    article_features = vectorizer.transform(articles_keywords).toarray()
    print([index for index, value in enumerate(software_features[1]) if value != 0])
    print([index for index, value in enumerate(article_features[1]) if value != 0])

    return article_features, software_features


def preprocess_heterodata(articles_dict, software_dict):
    data = HeteroData()

    articles_features, software_features = keywords_vocabulary(articles_dict, software_dict)
    data['articles'].x = articles_features
    data['software'].x = software_features
    print(data)


def software_to_software_edges(software_dict):
    related_software = software_dict['related_software']
    print(related_software)

    for relation in related_software.values():
        if isinstance(relation, str):
            rels = relation.split(";")
            rels_dict = []
            for rel in rels:
                cleaned_string = rel[rel.index("(") + 1: rel.rindex(")")]

                # Step 2: Extract the key-value pairs using regex
                key_value_pairs = re.findall(r":(\w+)\s*=>\s*(\d+|\".*?\")", cleaned_string)

                # Step 3: Create the dictionary
                result_dict = {}
                for key, value in key_value_pairs:
                    # Convert numerical strings to integers
                    if value.isdigit():
                        result_dict[key] = int(value)
                    else:
                        result_dict[key] = value.strip('"')
                rel = result_dict


    return software_dict


def software_to_articles_edges(software_dict, articles_dict):
    return




def build_gnn(articles_dict, software_dict):
    articles_features, software_features = keywords_vocabulary(articles_dict=articles_dict, software_dict=software_dict)


#preprocess_heterodata(articles_dict, software_dict)

software_to_software_edges(software_dict)
