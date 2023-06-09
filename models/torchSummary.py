import torch
from torchsummary import summary
from models.dense_u import dense_u

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense = dense_u().to(device)
    summary(dense, input_size=(1, 128, 128, 128))