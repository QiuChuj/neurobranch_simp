from torch.optim import Adam
from neurobranch_simp import create_simple_model
from loaddata import create_data_loaders
import json

def full_training_pipeline(config):
    train_loader, val_loader = create_data_loaders(data_dir=config['data_dir'], label_dir=config['label_dir'], batch_size=config['batch_size'], val_split=config['val_split'])
    
    net = create_simple_model()
    optimizer = Adam(net.parameters(), lr = config['learning_rate'])

    train_loss,val_loss = net.train_epoch(train_loader, val_loader, optimizer, epochs=config['epochs'])
    
    return train_loss, val_loss

if __name__ == "__main__":
    with open("/home/richard/project/neurobranch_simp/configs/config.json") as f:
        config = json.load(f)
    
    training_results = full_training_pipeline(config)
    print("训练完成！")