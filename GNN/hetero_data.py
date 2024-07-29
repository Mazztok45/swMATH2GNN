import pandas as pd
from torch_geometric.data import HeteroData
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import torch_geometric.transforms as T



def preprocess_heterodata(articles_dict, software_dict):
    data = HeteroData()

    articles_features, software_features = keywords_vocabulary(articles_dict, software_dict)
    data['article'].x = articles_features
    data['software'].x = software_features

    software_edges = software_to_software_edges(software_dict)
    data['software', 'related', 'software'].edge_index = software_edges

    sof_art_edges = software_to_articles_edges(software_dict, articles_dict)
    data['software', 'mentioned_in', 'article'].edge_index = sof_art_edges

    article_edges = article_to_article_edges(articles_dict)
    data['article', 'references', 'article'].edge_index = article_edges

    data = T.ToUndirected()(data)
    return data


def keywords_vocabulary(articles_dict, software_dict):
    """
    Returns tfidf matrix for the keywords. Treats the keywords of each software/article as a sentence
    :param articles_dict: Dictionary with articles
    :param software_dict: Dictionary with software
    :return: TfIdf-Matrix for articles and software
    """
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

    vectorizer = TfidfVectorizer(max_features=15000)

    all_features = vectorizer.fit(all_keywords)
    software_features = vectorizer.transform(software_keywords).toarray()
    article_features = vectorizer.transform(articles_keywords).toarray()

    return article_features, software_features


def software_to_software_edges(software_dict):
    related_software = software_dict['related_software']
    # values of this are a julia dict converted to a string, needs reshaping
    for key, relation in related_software.items():
        if isinstance(relation, str):
            ids = re.findall(r':id\s*=>\s*(\d+)', relation)
            ids = list(map(int, ids))
            related_software[key] = ids
        else:
            related_software[key] = []

    edge_index = create_edge_index(related_software)

    return edge_index


def software_to_articles_edges(software_dict, articles_dict):
    standard_articles = software_dict['standard_articles']
    for key, relation in standard_articles.items():
        if isinstance(relation, str):
            ids = re.findall(r':id\s*=>\s*(\d+)', relation)
            ids = list(map(int, ids))
            standard_articles[key] = ids
        else:
            standard_articles[key] = []

    edge_index = create_edge_index(standard_articles)

    return edge_index


def article_to_article_edges(articles_dict):
    references = articles_dict['ref_ids']

    for key, value in references.items():
        if isinstance(value, str):
            ref_ids = value.split("; ")
            integer_list = [int(num) for num in ref_ids]
            references[key] = integer_list
        else:
            references[key] = []

    edge_index = create_edge_index(references)
    return edge_index


def create_edge_index(edge_dict):
    source_indices = []
    target_indices = []

    for source, targets in edge_dict.items():
        for target in targets:
            source_indices.append(source)
            target_indices.append(target)

    source_tensor = torch.tensor(source_indices, dtype=torch.long)
    target_tensor = torch.tensor(target_indices, dtype=torch.long)

    edge_index = torch.stack([source_tensor, target_tensor], dim=0)

    return edge_index