from models.DBN import DBN
import torch


train_data = None

dbn_mnist = DBN(visible_units=28 * 28,
                hidden_units=[400, 500, 800],
                k=1,
                learning_rate=0.1,
                learning_rate_decay=False,
                initial_momentum=0.5,
                final_momentum=0.95,
                weight_decay=0.0001,
                xavier_init=False,
                increase_to_cd_k=False,
                use_gpu=torch.cuda.is_available())  # example DBN model

num_epochs = 50
batch_size = 125

dbn_mnist.train_static(
    train_data.data,
    train_data.targets,
    num_epochs,
    batch_size
)  # Learning procedure for the DBN model
