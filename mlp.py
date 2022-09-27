import torch
import numpy as np
import torch.nn.functional as F

class MLP(torch.nn.Module):
    '''
    Multilayer Perceptron.
    dims: [input, h1, h2, ..., hn]
    '''
    def __init__(self, dims, activates, device, lr):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            input_dim,output_dim = dims[i],dims[i+1]
            layers.append(torch.nn.Linear(input_dim, output_dim))
            if i < len(activates):
                layers.append(activates[i]())
        self.layers = torch.nn.Sequential(*layers)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def update_learning_rate(self, lr):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        
    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

def suffle(X, Y, batch_size=1.0):
    assert len(X) == len(Y)
    assert type(batch_size)==float or type(batch_size)==int
    indice = None
    if type(batch_size)==float:
        assert batch_size > 0 and batch_size <= 1.0
        indice = np.random.choice(len(X), int(batch_size*len(X)), replace=False)
    else:
        assert batch_size > 0 and batch_size <= len(X)
        indice = np.random.choice(len(X), batch_size, replace=False)
    X_,Y_ = np.array(X)[indice], np.array(Y)[indice]
    return X_,Y_

def train(X, Y, model, epochs, batch_size=1.0, weight=None, gap=0.01):
    ITERA = int(len(X)/len(suffle(X, Y, batch_size)[0]))
    gap = int(epochs * ITERA * gap)
    print('start training for epochs: ' + str(epochs) + ' itera: ' + str(ITERA) + ' gap: ' + str(gap))
    for epoch in range(epochs):
        for it in range(ITERA):
            model.train()
            model.zero_grad()
            X_,Y_ = suffle(X, Y, batch_size)
            X_,Y_ = torch.tensor(X_, dtype=torch.float32),torch.tensor(Y_, dtype=torch.long)
            if weight is not None:
                weight = torch.tensor(weight, dtype=torch.float32)
            logits = model.forward(X_)
            loss = F.cross_entropy(logits, Y_, weight=weight)
            loss.backward()
            model.optimizer.step()
            if (epoch*ITERA+it) % gap == 0:
                loss,acc = evaluate(X_, Y_, model)
                print(str(np.round((epoch*ITERA+it)*100/(epochs*ITERA), 4)) + '%', np.round(loss, 4), np.round(acc, 4))
        
def evaluate(X, Y, model):
    with torch.no_grad():
        model.eval()
        logits = model.forward(X)
        predict_prob = F.softmax(logits, dim=1)
        predicts = torch.tensor([pred.item() for pred in torch.argmax(predict_prob, dim=1)])
        loss = F.cross_entropy(logits, Y).item()
        acc = ((predicts==Y).float().sum()/len(predicts)).item()
        return loss, acc
    
def reference(X, model):
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        model.eval()
        logits = model.forward(X)
        predict_prob = F.softmax(logits, dim=1)
        predicts = [pred.item() for pred in torch.argmax(predict_prob, dim=1)]
        return predicts,predict_prob