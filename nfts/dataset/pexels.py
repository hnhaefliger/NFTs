import requests


def get_dataset(query, key='pexels.txt'):
    with open(key, 'r') as f:
        key = f.read().split('\n')[0]

    headers = {
        'Authorization': key
    }

    page = 0

    while True:
        page += 1
        temp = requests.get(f'https://api.pexels.com/v1/search?query={query}&per_page80&page={page}', headers=headers).json()
        total_results = temp['total_results']
        
        with open(f'images.txt', 'a+') as f:
            f.write('\n'.join([image['src']['large'] for image in temp['photos']]) + '\n')

        if total_results <= 80 * page:
            break


get_dataset('abstract')