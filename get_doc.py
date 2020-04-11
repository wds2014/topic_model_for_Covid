import os
import json
from pprint import pprint
from copy import deepcopy

import numpy as np
import pandas as pd
import tqdm

def format_name(author):
    middle_name = " ".join(author['middle'])
    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))
    
    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)

def format_authors(authors, with_affiliation=False):
    name_ls = []
    
    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)
    
    return ", ".join(name_ls)

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""
    for section, text in texts_di.items():
        body += text

    # for section, text in texts_di.items():
        # body += section
        # body += "\n\n"
        # body += text
        # body += "\n\n"
    
    return body

def format_bib(bibs):
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []
    
    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'], 
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))

    return "; ".join(formatted)
	

def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)
    
    return raw_files

def generate_clean_df(all_files):
    cleaned_files = []
    
    for file in tqdm(all_files):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'], 
                           with_affiliation=True),
            format_body(file['abstract']),
            format_body(file['body_text']),
            format_bib(file['bib_entries']),
            file['metadata']['authors'],
            file['bib_entries']
        ]

        cleaned_files.append(features)

    col_names = ['paper_id', 'title', 'authors',
                 'affiliations', 'abstract', 'text', 
                 'bibliography','raw_authors','raw_bibliography']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()
    
    return clean_df
    
if __name__ == '__main__' :
    biorxiv_dir = 'pmc_custom_license/pmc_custom_license/'
    filenames = os.listdir(biorxiv_dir)
    print("Number of articles retrieved from biorxiv:", len(filenames))
    all_files = []

    for filename in filenames:
        filename = biorxiv_dir + filename
        file = json.load(open(filename, 'rb'))
        all_files.append(file)
        
    cleaned_files = []
    doc_num = 0
    with open('pmc_custom_license.txt','w') as f:
        for file in all_files:
            doc = format_body(file['abstract']) + format_body(file['abstract'])
            if len(doc) > 20:
                f.write(doc)
                f.write('\n')
                doc_num +=1
    print('doneeeee', doc_num)
    
    # features = [
        # file['paper_id'],
        # file['metadata']['title'],
        # format_authors(file['metadata']['authors']),
        # format_authors(file['metadata']['authors'], 
                       # with_affiliation=True),
        # format_body(file['abstract']),
        # format_body(file['body_text']),
        # format_bib(file['bib_entries']),
        # file['metadata']['authors'],
        # file['bib_entries']
    # ]
