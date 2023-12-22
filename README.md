# PYSEG2

Read/Write seg2 files in python
This program is stand alone but it can export obspy streams in seg2 files.

Usage :  
```
from pyseg2 import Seg2file

seg2 = Seg2File()
with open('./input_file.seg2', 'rb') as fid:
    seg2.load(fid)

seg2.seg2traces = seg2.seg2traces[:2]

with open('output_file.seg2', 'wb') as fil:
    fil.write(seg2.pack())

seg2re = Seg2File()
with open('output_file.seg2', 'rb') as fid:
    seg2re.load(fid)

```
