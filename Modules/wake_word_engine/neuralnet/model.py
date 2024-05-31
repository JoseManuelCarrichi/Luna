"""Modelo de detección de wake word con LSTM y PyTorch."""
import torch
import torch.nn as nn


class LSTMWakeWord(nn.Module):

    def __init__(self, num_classes, feature_size, hidden_size,
                num_layers, dropout, bidirectional, device='cpu'):
        super(LSTMWakeWord, self).__init__()
        # Número de capas y tamaño de la capa oculta
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # Calcula el número de direcciones basado en si la LSTM es bidireccional
        self.directions = 2 if bidirectional else 1
        self.device = device
        # Capa de normalización por lotes para normalizar las características de entrada
        self.layernorm = nn.LayerNorm(feature_size)
        # Capa LSTM que toma características de entrada y produce una salida oculta
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional)
        # Capa lineal que clasifica la salida oculta de la LSTM en clases
        self.classifier = nn.Linear(hidden_size*self.directions, num_classes)

    # Método para inicializar el estado oculto de la LSTM
    def _init_hidden(self, batch_size):
        n, d, hs = self.num_layers, self.directions, self.hidden_size
        # Crea tensores de ceros para el estado oculto y de celda de la LSTM
        return (torch.zeros(n*d, batch_size, hs).to(self.device),
                torch.zeros(n*d, batch_size, hs).to(self.device))

    # Método de paso hacia adelante del modelo
    def forward(self, x):
        # x.shape => seq_len, batch, feature
        x = self.layernorm(x)
        hidden = self._init_hidden(x.size()[1])
        # Pasa las características de entrada a través de la LSTM
        out, (hn, cn) = self.lstm(x, hidden)
        # Pasa el estado oculto final a través de la capa clasificadora
        out = self.classifier(hn)
        return out