import requests
import re
import os
import warnings
warnings.simplefilter("ignore")


def get_artists():
    artists = []

    for letter in 'abcdefghijklmnopqrstuvwxyz':
        page = 0

        while True:
            page += 1
            tmp = requests.get(f'https://www.wikiart.org/en/alphabet/{letter}/?json=3&layout=new&page={page}', verify=False).json()['Artists']

            artists += tmp

            if tmp == []:
                break

    artists = [artist['artistUrl'] for artist in artists]

    return artists


def get_artist_paintings(artist):
    paintings = []

    page = 0

    while True:
        page += 1
        tmp = requests.get(f'https://www.wikiart.org/{artist}/mode/all-paintings?json=2&layout=new&page={page}', verify=False).json()['Paintings']

        if isinstance(tmp, type(None)):
            break

        else:
            paintings += tmp

    paintings = [{'url': painting['paintingUrl'], 'image': painting['image']} for painting in paintings]

    return paintings


def get_painting_info(painting):
    html = requests.get(f'https://www.wikiart.org/{painting}', verify=False).content.decode('utf-8')

    date = re.findall('<li>\s*<s>Date:</s>\s*.*<span itemprop="dateCreated">(.+?)</span>\s*</li>', html)

    styles = re.findall('<li class="dictionary-values\s*"\s*>\s*<s>Style:\s*</s>\s*<span\s*>([\s\S]+?)</span>[\s\S]+?</li>', html)
    styles = re.findall('<a target="_self" href="/en/paintings-by-style/.+?">(.+?)</a>', styles[0])

    genres = re.findall('<li class="dictionary-values\s*"\s*>\s*<s>Genre:\s*</s>\s*<span >([\s\S]+?)</span>\s*</li>', html)
    genres = re.findall('<a target="_self" href="/en/paintings-by-genre/.+?"><span itemprop="genre">(.+?)</span></a>', genres[0])
    
    tags = re.findall('<a target="_self" class="tags-cheaps__item__ref" href=".+?">\s*(.+?)\s*</a>', html)

    return date[0], styles, genres, tags


def get_dataset():
    os.mkdir('wikiart')

    artists = get_artists()
    i = 0

    for artist in artists:
        paintings = get_artist_paintings(artist)

        for painting in paintings:
            info = get_painting_info(painting['url'])

            image = requests.get(painting['image'], verify=False).content

            with open(f'wikiart/image_{i}.jpg', 'wb') as f:
                f.write(image)

            with open('wikiart.txt', 'a+') as f:
                f.write(f'wikiart/image_{i}.jpg,{artist},{painting},{info[0]},{"|".join(info[1])},{"|".join(info[2])},{"|".join(info[3])}\n')

            i += 1


print(get_painting_info('en/raphael/the-veiled-woman-or-la-donna-velata'))

#if __name__ == '__main__':
    #get_dataset()
