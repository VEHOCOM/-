import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=True,
                           batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, text, lengths):
        embedded = self.embedding(text)
        
        # Pack padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), 
                                                          batch_first=True, 
                                                          enforce_sorted=False)
        
        # Run through LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Concatenate the final forward and backward hidden states
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        
        # Pass through linear layer
        output = self.fc(hidden)
        
        return output