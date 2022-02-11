
import h5py
from pprint import pprint
import json

f = h5py.File('serialized/model0.9740557074546814.h5', 'r')
attributes = f.attrs.get('model_config')

pprint( attributes )