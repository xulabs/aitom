'''

imput dataset format should be:

demo dataset is a dict,key is the label, value is a matrix size of (number * size * size * size)

    {'label1':[[[]]],
    'label2':[[[]]]
    }
use data from following location for testing:
tutorials/000_general.md


Then we convert it to
    [
    {'pdb_id':'label1',
    'v':[[[]]]
    },
    {'pdb_id':'label2',
    'v':[[[]]]
    },
    {'pdb_id':'label1',
    'v':[[[]]]
    }
    ]
using the function SD.convert_demo_data_format(dj, label_list)

So you can use either of the two format above.


'''



import aitom.classify.deep.supervised.cnn.subdivide as SD
import pickle

with open('./aitom_demo_subtomograms.pickle', 'rb') as f:
    dj = pickle.load(f, encoding='iso-8859-1')

label_list = ['5T2C_data', '1KP8_data']#here's the label of the dataset, you need to add manually
dj = SD.convert_demo_data_format(dj, label_list)

pdb_id_map = SD.pdb_id_label_map([_['pdb_id'] for _ in dj])

model = SD.inception3D(image_size=dj[0]['v'].shape[0], num_labels=len(pdb_id_map))
# model = SD.dsrff3D(image_size=dj[0]['v'].shape[0], num_labels=len(pdb_id_map))
model = SD.compile(model)
SD.train_validation(model=model, dj=dj, pdb_id_map=pdb_id_map, nb_epoch=10, validation_split=0.2)
