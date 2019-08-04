rm -rf build
find . -type d -name __pycache__ | xargs -I {} rm -rf {} 
find . -type f -name "*.so" | xargs -I {} rm {}
