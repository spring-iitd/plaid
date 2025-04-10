import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ids

def get_model(model_name):
    for model_class in ids.__all_classes__:
        if model_class.__name__ == model_name:
            return model_class()
    raise Exception(f"{model_name} not yet implemented")