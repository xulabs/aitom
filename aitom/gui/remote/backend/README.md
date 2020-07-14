# Backend mrc loader for online visulizatin
## Introduction
This project was originally designed for scaling a mrc file, and getting a block of data.
## Usage
```python
from MrcLoader import MrcLoader

m = MrcLoader('test.mrc')

m.read((0,0,0), (100, 100, 100), 0, True)
```
- load a mrc file
- read a cubic data with scale coefficient. For example, If original mrc file is (200, 200, 200), and code is `m.read((0,0,0), (25, 25, 25), 2)`, MrcLoader will do `SCALE_BASE`x reduction 2 times(`SCALE_BASE` is in `config.py`). If `SCALE_BASE == 2`, mrc will down-sample file to (50, 50, 50) and get this file (0, 0, 0)-(25, 25, 25). i.e. Original file (0, 0, 0)-(100, 100, 100)

## Future works
- Merge into backend code.
