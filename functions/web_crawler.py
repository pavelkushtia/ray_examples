import ray

@ray.remote
def crawl(url, depth=0, maxdepth=1, maxlinks=4):
    links = []
    link_futures = []
    import requests
    from bs4 import BeautifulSoup 
    try:
        f = requests.get(url) 
        links += [(url, f.text)] 
        if (depth > maxdepth):
            return links # base case
        soup = BeautifulSoup(f.text, 'html.parser') 
        c=0
        for link in soup.find_all('a'):
            try: 
                c=c+1
                link_futures += [crawl.remote(link["href"], depth=(depth+1), maxdepth=maxdepth)]
                # Don't branch too much; we're still in local mode and the web is big
                if c > maxlinks: 
                    break
            except: 
                pass
        for r in ray.get(link_futures): 
            links += r
        return links
    except requests.exceptions.InvalidSchema:
        return [] # Skip nonweb links
    except requests.exceptions.MissingSchema:
        return [] # Skip nonweb links
        
ray.get(crawl.remote("http://holdenkarau.com/"))
