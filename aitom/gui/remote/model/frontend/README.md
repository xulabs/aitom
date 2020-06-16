# demo webpage for mrc file visulization
## Introduction
This demo webpage is designed for online model visulization. Mrc file will be pre-processed on server, and send model only to webpage to display.
## Usage
- For simpler use, models and website page are saved locally
- Start server locally, e.g. `python -m http.server 8000`. Noticing that in `index.html`, line 20: `server = 'http://localhost:8000/'`. This url should correspond with command url.
- Access webpage. There are two available buttons `2` and `3`. Clicking each button will load a different model. `vtk` Model can be generated from `backend/contour.py`
## Future works
- Integrate with backend code(actually it is partial realized).
- Add zoom and shift buttons, and integrate with backend code.
