from torch_geometric.data import HeteroData
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import torch_geometric.transforms as T
import scipy
import dask.array as da

def preprocess_heterodata(articles_dict, software_dict):
    """

    :param articles_dict: Dictionary made from articles_data/full_df.csv
    :param software_dict: Dictionary made from data/full_df.csv
    :return: HeteroData() object with nodes and edges
    """
    data = HeteroData()

    # create nodes for all articles and software with the features from the called function
    articles_features, software_features = keywords_vocabulary(articles_dict, software_dict)
    data['article'].x = articles_features
    data['software'].x = software_features

    # create edges between the software
    software_edges = software_to_software_edges(software_dict)
    data['software', 'related', 'software'].edge_index = software_edges

    # create edges between software and articles
    sof_art_edges = software_to_articles_edges(software_dict, articles_dict)
    data['software', 'mentioned_in', 'article'].edge_index = sof_art_edges

    # create edges between article and article
    article_edges = article_to_article_edges(articles_dict)
    data['article', 'references', 'article'].edge_index = article_edges

    data = T.ToUndirected()(data)
    print("Built HeteroData Dataset")
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

    # max_features might need to be changed according to the RAM of the computer/ numpy version
    vectorizer = TfidfVectorizer(max_features=15000)

    all_features = vectorizer.fit(all_keywords)
    software_features = vectorizer.fit_transform(software_keywords).toarray()
    articles_features = vectorizer.fit_transform(articles_keywords).toarray()

    # Convert the sparse matrix to a Dask array
    articles_features_dask = da.from_array(articles_features.toarray(), chunks=(1000, 15000))

    # Articles features dimensionality reduction
    pca = PCA(n_components=5000)
    articles_features_dense = articles_features_dask.toarray()
    articles_features_reduced = pca.fit_transform(articles_features_dense)
    articles_features_reduced_sparse = sparse.csr_matrix(articles_features_reduced)
    article_features = articles_features_reduced_sparse

    software_features_dask = da.from_array(software_features.toarray(), chunks=(1000, 15000))
    # Software features dimensionality reduction
    software_features_dense = software_features_dask.toarray()
    software_features_reduced = pca.fit_transform(software_features_dense)
    software_features_reduced_sparse = sparse.csr_matrix(software_features_reduced)
    software_features = software_features_reduced_sparse
    return article_features, software_features


def software_to_software_edges(software_dict):
    """

    :param software_dict: Dictionary for software
    :return: edge_index for the edges between the software (torch.stack)
    """
    related_software = software_dict['related_software']

    # values of this are a julia dict converted to a string, needs reshaping
    for key, relation in related_software.items():
        if isinstance(relation, str):
            ids = re.findall(r':id\s*=>\s*(\d+)', relation)
            ids = list(map(int, ids))
            related_software[key] = ids
        else:
            related_software[key] = []

    mapped_dict = ids_mapping(software_dict)
    edge_index = create_edge_index(related_software, mapped_dict)

    return edge_index


def software_to_articles_edges(software_dict, articles_dict):
    """

    :param software_dict: Dictionary for software
    :param articles_dict: dictionary for articles
    :return: edge_index for edges between articles and software
    """
    standard_articles = software_dict['standard_articles']

    # values of this are a julia dict converted to a string, needs reshaping
    for key, relation in standard_articles.items():
        if isinstance(relation, str):
            ids = re.findall(r':id\s*=>\s*(\d+)', relation)
            ids = list(map(int, ids))
            standard_articles[key] = ids
        else:
            standard_articles[key] = []
    mapped_dict = ids_mapping(articles_dict)
    edge_index = create_edge_index(standard_articles, mapped_dict)

    return edge_index


def article_to_article_edges(articles_dict):
    """

    :param articles_dict: Dictionary for articles
    :return: edge_index for edges between articles
    """
    references = articles_dict['ref_ids']

    for key, value in references.items():
        if isinstance(value, str):
            ref_ids = value.split("; ")
            integer_list = [int(num) for num in ref_ids]
            references[key] = integer_list
        else:
            references[key] = []

    mapped_dict = ids_mapping(articles_dict)
    edge_index = create_edge_index(references, mapped_dict)
    return edge_index


def ids_mapping(mapping_dict):
    """
    The ids that are saved in the swMATH/zbMATH database are different from the indexes in the csv files.
    The indexes from the csv files are the ids of the nodes.
    The information on the relations between software/software, article/article, software/article are
    based on the ids in database. Therefore we need to map the ids from one to another, to create
    the correct edges.

    :param mapping_dict: the dictionary that needs mapping
    :return: mapped dictionary
    """
    ids = mapping_dict['id']
    inverted_dict = {value: key for key, value in ids.items()}
    return inverted_dict


def create_edge_index(edge_dict, mapped_dict):
    """
    The Approximate Algorithm for Set Cover Problem
Last Updated : 14 Jun, 2023

Given a universe U of n elements, a collection of subsets of U say S = {S1, S2…,Sm} where every subset Si has function that actually creates the edge_index (for less code repition)
    :param edge_dict: The dict that holds the information on the edges
    :param mapped_dict: the mapped ids from database ids to dataset indecies
    :return: edge_index for the certain edge
    """
    source_indices = []
    target_indices = []

    for source, targets in edge_dict.items():
        for target in targets:
            if target in mapped_dict.keys():
                source_indices.append(source)
                target_map = mapped_dict[target]
                target_indices.append(target_map)

    source_tensor = torch.tensor(source_indices, dtype=torch.long)
    target_tensor = torch.tensor(target_indices, dtype=torch.long)

    edge_index = torch.stack([source_tensor, target_tensor], dim=0)

    return edge_index
