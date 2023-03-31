from .flow_model import Flow_Model
from .depth_model import Depth_Model

def model_loader(cfgs):
    if cfgs['model'] == 'flow':
        net = Flow_Model(cfgs=cfgs) 
    else:
        net = Depth_Model(cfgs=cfgs)
    return net