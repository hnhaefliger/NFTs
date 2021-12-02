import requests
import asyncio

def get_results(query, size='q', target=20000):
    '''
    Sizes:
        sq: 75 x 75
        q: 150 x 150
        t: 100 x ? or ? x 100
        s: 240 x ? or ? x 240
        n: 320 x ? or ? x 320
        w: 400 x ? or ? x 400
        m: 500 x ? or ? x 500
        z: 640 x ? or ? x 640
        c: 800 x ? or ? x 800
        l: 1024 x ? or ? x 1024
    '''
    def get_photos(page):
        response = requests.get(
            'https://api.flickr.com/services/rest',
            params={
                'sort': 'relevance',
                'content_type': 7,
                'extras': f'url_{size}',
                'per_page': min(target, 500),
                'page': page,
                'text': query,
                'method': 'flickr.photos.search',
                'api_key': '973475ba6159373c1b5431949b1bc8df',
                'format': 'json',
                'nojsoncallback': 1,
            }
        )

        return [image for image in response.json()['photos']['photo'] if f'url_{size}' in image]

    page = 0
    results = []

    while len(results) < target:
        page += 1

        temp = get_photos(page)

        if len(temp) == 0:
            break

        results += [image[f'url_{size}'] for image in temp]

    return results
    

async def save_image(url, name, location):
    image = requests.get(url).content

    with open(f'{location}/{name}', 'wb') as f:
        f.write(image)

    return True


async def _get_images(images, location):
    for i, image in enumerate(images):
        await save_image(image, f'image_{i}.jpg', location)

    return True
    

def get_images(images, location):
    loop = asyncio.new_event_loop()

    loop.run_until_complete(_get_images(images, location))

    return True


if __name__ == '__main__':
    import sys

    get_images(get_results(sys.argv[0], target=int(sys.argv[2])), sys.argv[3])
