from torch.optim import Adam
from neurobranch_simp import create_simple_model
from loaddata import create_data_loader
import json

def full_applying_pipeline():
    data_loader = create_data_loader()
    
    net = create_simple_model()

    result = net.apply(data_loader)
    
    return result

if __name__ == "__main__":

    training_results = full_applying_pipeline()
    print("训练完成！")