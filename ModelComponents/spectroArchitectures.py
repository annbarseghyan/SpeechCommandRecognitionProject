import torch
import torch.nn as nn

class CNN_spectro(nn.Module):
    def __init__(self, num_classes=6):
        super(CNN_spectro, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(20,8), stride=(1,3))
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=(10,4), stride=(1,1))
        self.fc1 = nn.Linear(68224, 32)


        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        

        x = torch.flatten(x,start_dim=1)
        
    
        x = self.relu(self.fc1(x))
        
        x = self.relu(self.fc2(x))
        
        x=self.fc3(x)
        
        
        return x
    
    
class MLP_spectro(nn.Module):
    def __init__(self, num_classes=6,dropout_p=0.3):
        super(MLP_spectro, self).__init__()
        
        self.fc1 = nn.Linear(201*101, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.fc4(x)
        return x