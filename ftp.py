import zipfile
from pathlib import Path
import tqdm

root = zipfile.ZipFile('/home/bingliang/data/WebVid2.5M/WebVid.zip', 'w', zipfile.ZIP_STORED)

path = '/home/bingliang/data/WebVid2.5M/videos/'

for i in tqdm.trange(1, 10001, 50):
    p = '{:06d}_{:06d}'.format(i, i+49)
    for sub_p in tqdm.tqdm(Path(path + p).glob('*')):
        root.write(str(sub_p), p+'/'+sub_p.name)
root.close()