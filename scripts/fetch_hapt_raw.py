"""Download the original UCI HAPT recordings and verify archive paths."""
import argparse,hashlib,io,json,urllib.request,zipfile
from pathlib import Path
URL='https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip'
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='data/HAPT');a=p.parse_args();out=Path(a.out)
    if out.exists():raise ValueError('Output exists; use a new directory')
    with urllib.request.urlopen(URL,timeout=120) as r:data=r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for n in z.namelist():
            path=Path(n)
            if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe archive member')
        out.mkdir(parents=True)
        for n in z.namelist():
            if n.startswith('RawData/') or 'readme' in n.lower() or 'activity_labels' in n.lower():z.extract(n,out)
    (out/'source.json').write_text(json.dumps(dict(url=URL,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)),indent=2))
    print('Downloaded original RawData to',out)
if __name__=='__main__':main()
