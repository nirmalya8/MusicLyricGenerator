import pandas as pd
import os
import numpy as np

def get_data():
    datadir_path = os.path.join(os.path.dirname(__file__),"Data")
    data_path = os.path.join(datadir_path,"taylor_swift.tsv")
    print("========Loading Data========")
    print(f'Path of Data Directory: {datadir_path}')
    print(f'Path of Data File: {data_path}')
    data = pd.read_csv(data_path, sep='\t').drop(columns=['index'])
    print(f'========Loaded Data========')
    return data

def get_unique_albums(data):
    print("========Getting Unique Albums========")
    unique_albums = data.album.drop_duplicates().tolist()
    print(f'Number of Unique Albums: {unique_albums}')
    print(f'5 examples: {unique_albums[0:5]}')
    print("========Unique Albums Obtained========")
    return unique_albums

def del_words(words,unique_albums):
    print("========Deleting Problematic Words========")
    indices = []
    count = 0
    for album in unique_albums:
        album = album.split() 
        if len(album) != 0:
            for a in album:
                if a in words:
                    indices.append(count)
        count += 1
    print("========Deleted Problematic Words========")
    return indices

def get_new_album_list(indices,unique_albums):
    print("========Getting Updated Album List========")
    albums = []
    for i in range(len(unique_albums)):
        if i not in indices:
            albums.append(unique_albums[i])
    albums+=['folklore']
    albums = pd.DataFrame(albums).drop_duplicates(keep='first')
    albums = albums.drop([15,12])[0].tolist()
    print(f'Number of New Albums: {albums}')
    print(f'5 examples: {albums[0:5]}')
    print("========Obtained List of Albums========")
    return albums

def get_album_titles(albums):
    print("========Getting Album Titles========")
    titles = []
    for album in albums:
        a = album.split()
        titles.append(a)
    print(f'Number of Album Titles: {titles}')
    print(f'5 examples: {titles[0:5]}')
    print("========Obtained Titles========")
    return titles

if __name__ == '__main__':
    data = get_data()
    unique_albums = get_unique_albums(data)
    
    words = ['Live','Genius','Demo','folklore']
    indices = del_words(words,unique_albums)

    albums = get_new_album_list(indices,unique_albums)

    titles = get_album_titles(albums)
