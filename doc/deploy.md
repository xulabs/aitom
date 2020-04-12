# Deploy

```
git clone https://github.com/xulabs/aitom
cd aitom
python3 setup.py sdist 
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```


# Note

1. Pypi will not pack some files by default(.hpp .pyx). 'MANIFEST.in' is used to include all files in source distributions.

2. Dependent packages in 'requirements.txt' will be installed automatically when 'pip install aitom'. However, 'cython' and 'numpy' are used in 'setup.py' so they should be installed first.
