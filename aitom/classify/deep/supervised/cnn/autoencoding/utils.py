import pickle
import aitom.image.vol.util as AIVU
import aitom.image.io as AIIO


def visualize(img, path):
    t = AIVU.cub_img(img)['im']
    AIIO.save_png(t, path)
