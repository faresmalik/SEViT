import torch.nn as nn 
import torch

class Classifier(nn.Module): 
    """
    MLP classifier. 
    Args:
        num_classes -> number of classes 
        in_feature -> features dimension

    return logits. 
    
    """
    def __init__(self,num_classes=2 ,in_features = 768*196):
        
        super().__init__()
        self.linear1 = nn.Linear(in_features= in_features, out_features= 4096)
        self.linear2 = nn.Linear(in_features= 4096, out_features= 2048)
        self.linear3 = nn.Linear(in_features= 2048, out_features= 128)
        self.linear4 = nn.Linear(in_features= 128, out_features= num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self,x):
        x= x.reshape(-1, 196*768)
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = self.linear4(x)
        return x