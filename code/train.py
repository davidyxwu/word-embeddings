from fetch_datasets import fetch_IMDB, fetch_CoLA
from model import GRU
from torchsummary import summary

model = GRU(25004, 300, 256, 64, 2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
