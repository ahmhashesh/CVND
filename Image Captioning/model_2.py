import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.randn(1, 1, self.hidden_size), torch.randn(1, 1, self.hidden_size)) 
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, features, captions):
        captions = captions[:, :-1]
        embed = self.embed(captions)
        #print(features.size(),embed.size())
        #print(self.embed_size,self.hidden_size,self.vocab_size)
        inputs = torch.cat((features.unsqueeze(1), embed), dim = 1)
        #print(inputs.size(),embed.size())
        out,_ = self.lstm(inputs)
        out = self.linear(out)
        out = self.softmax(out)
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        for x in range(max_len):
            out,states = self.lstm(inputs,states)
            out = out.squeeze(1)
            outputs = self.linear(out)
            predict = self.softmax(outputs)
            #print(predict)
            sentence.append(predict.item())
            #print(sentence)
            inputs = self.embed(predict).unsqueeze(1)
        return sentence