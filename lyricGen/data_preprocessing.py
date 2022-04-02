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

def get_choruses(data,albums):
    chorus_data=data[(data.album.isin(albums)) & (data.lyric=='[Chorus]')].drop_duplicates(subset='song_title').index.tolist()
    lyrics = data.lyric.tolist()
    stop=[]
    for l in lyrics:
        if l[0]=='[':
            stop.append(1)
        else:
            stop.append(0)
    data['S']=stop
    chorus = []
    for i in chorus_data:
    #Pick the chorus till the following line
        new_d=data[i+1::]
        ss=new_d.S.tolist()
        for q in range(len(ss)):
            if ss[q]==1:
                break
        new_d=data[i+1:i+1+q]
        chorus.append(new_d.lyric.tolist())

    return chorus

def return_index(chorus,lyrics):
    return chorus.index([lyrics])

def get_chorus_text(chorus):
    text=''
    for c in range(len(chorus)):
        for i in range(len(chorus[c])):
            if c%10==0:
                print(".",sep=" ")
            text= text+ chorus[c][i]+ ' \n '
    print(len(text))
    print(text[35000:])
    return text

def write_into_file(text):
    datadir_path = os.path.join(os.path.dirname(__file__),"Data")
    file_path = os.path.join(datadir_path,"chorus.txt")
    f = open(file_path, "w",encoding="utf-8")
    f.write(text)
    f.close()

if __name__ == '__main__':
    data = get_data()
    unique_albums = get_unique_albums(data)
    
    words = ['Live','Genius','Demo','folklore']
    indices = del_words(words,unique_albums)

    albums = get_new_album_list(indices,unique_albums)

    titles = get_album_titles(albums)

    chorus = get_choruses(data,albums)
    lyrics = "I didn't bring her up so they could cut her down\t8\tNone\n197\tUnreleased Songs\tBrought Up That Way \tI didn't bring her here so they could shut her out\t9\tNone\n197\tUnreleased Songs\tBrought Up That Way \tI live my whole damn life to see that little girl's smile\t10\tNone\n197\tUnreleased Songs\tBrought Up That Way \tSo why are tears pouring down that sweet face?\t11\tNone\n197\tUnreleased Songs\tBrought Up That Way \tShe wasn't brought up that way"

    print(return_index(chorus,lyrics))
    text = get_chorus_text(chorus)
    
    write_into_file(text)