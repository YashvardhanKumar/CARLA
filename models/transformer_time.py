import torch
import torch.nn as nn

import torch
import torch.nn as nn

# class TimeSeriesTransformer(nn.Module):
#     def __init__(self, input_dim, output_dim, num_layers=3, num_heads=4, d_model=32, dropout=0.1):
#         super(TimeSeriesTransformer, self).__init__()

#         self.input_projection = nn.Linear(input_dim, d_model)
#         self.positional_encoding = PositionalEncoding(d_model)

#         self.transformer_layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
#             for _ in range(num_layers)
#         ])

#         self.output_projection = nn.Linear(d_model, output_dim)

#     def forward(self, x):
#         # x shape: (batch_size, seq_len, input_dim)
#         batch_size, seq_len, _ = x.shape

#         x = self.input_projection(x) # Linear projection to d_model
#         x = self.positional_encoding(x) # Add positional encodings

#         for layer in self.transformer_layers:
#             x = layer(x)

#         # Average pooling over the sequence dimension
#         x = torch.mean(x, dim=1)

#         x = self.output_projection(x) # Linear projection to output_dim
#         return x

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe = torch.zeros(1, max_len, d_model)
#         pe[0, :, 0::2] = torch.sin(position * div_term)
#         pe[0, :, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor, shape [batch_size, seq_len, embedding_dim]
#         """
#         x = x + self.pe[:, :x.size(1), :]
#         return x


# # Example usage, matching the ResNet's input/output dimensions:
# input_dim = 55  # Matches the input channels of the ResNet
# output_dim = 8   # Matches the output channels of the ResNet
# seq_len = 128 # Example sequence length. You should adapt this to your data.

# # Example input tensor
# input_tensor = torch.randn(32, seq_len, input_dim) # Batch size 32

# transformer_model = TimeSeriesTransformer(input_dim, output_dim)
# output_tensor = transformer_model(input_tensor)

# print("Input shape:", input_tensor.shape)
# print("Output shape:", output_tensor.shape)

# import torch
# from transformers import AutoModel, AutoTokenizer

# class RelativePositionalEncoding(nn.Module):
#     def __init__(self, max_len, embedding_dim):
#         super(RelativePositionalEncoding, self).__init__()
#         self.embedding = nn.Embedding(max_len, embedding_dim)

#     def forward(self, x):
#         type(x)
#         batch_size, seq_len, _ = x.size()
#         # positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
#         positions = torch.arange(seq_len, device=x.device).long().expand(batch_size, seq_len) 

#         return self.embedding(positions)

# class TransformerAnomalyDetector(nn.Module):
#     def __init__(self):
#         super(TransformerAnomalyDetector, self).__init__()
#         self.model = AutoModel.from_pretrained('distilbert-base-uncased')
#         self.rpe = RelativePositionalEncoding(max_len=55, embedding_dim=55)
#         self.fc = nn.Linear(55, 8)  # 8 output classes
        
#         # self.model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=8)
#     def forward(self, x):
#         print(len(x[0]))
#         print(len(self.rpe(x)[0]))
#         x = self.rpe(x) + x
#         outputs = self.model(x)
#         anomaly_scores = torch.mean(outputs.last_hidden_state, dim=1)
#         outputs = self.fc(anomaly_scores)
#         return outputs
#     # def __init__(self, input_dim=55, output_dim=8, d_model=128, nhead=8, num_layers=2):  # Added hyperparameters
#     #     super(TransformerAnomalyDetector, self).__init__()
#     #     self.input_projection = nn.Linear(input_dim, d_model) # project input_dim to d_model
#     #     self.rpe = RelativePositionalEncoding(max_len=512, embedding_dim=d_model) #increased max len
#     #     self.transformer_encoder = nn.TransformerEncoder(
#     #         nn.TransformerEncoderLayer(d_model, nhead, batch_first=True), num_layers=num_layers
#     #     )
#     #     self.output_projection = nn.Linear(d_model, output_dim)


#     # def forward(self, x):
#     #     x = self.input_projection(x)
#     #     x = self.rpe(x) + x  # Add relative positional encodings
#     #     x = self.transformer_encoder(x) # Pass through the encoder layers
#     #     x = torch.mean(x, dim=1) #average pooling
#     #     x = self.output_projection(x)
#     #     return x

# # Initialize the model
# # model = TransformerAnomalyDetector()


# def transformer_ts(**kwargs):
    # return {
    #         'backbone': TransformerAnomalyDetector(),
    #         'dim': kwargs['embed_dim']  # Match the embedding dimension
    #     }

class TransformerBackbone(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, seq_len):
        super(TransformerBackbone, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.position_encoding = nn.Parameter(torch.randn(seq_len, embed_dim))
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)
        self.output_dim = embed_dim

    def forward(self, x):
        # x is expected to be [batch, seq_len, input_dim]
        # ensure that x matches this format
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x + self.position_encoding  # Add positional encoding [seq_len, embed_dim]

        # Transformer expects [seq_len, batch, embed_dim], so transpose:
        x = x.permute(1, 0, 2)  # [seq_len, batch, embed_dim]

        x = self.transformer(x)  # [seq_len, batch, embed_dim]
        x = x.mean(dim=0)  # Global average pooling along seq_len [batch, embed_dim]
        return x


def transformer_ts(in_channels, seq_len, embed_dim=64, num_heads=4, num_layers=2):
    """
    Initializes a transformer backbone. 
    in_channels corresponds to the input feature dimension (similar to the old 'in_channels' from ResNet).
    seq_len is the length of the input sequence.
    embed_dim, num_heads, and num_layers are Transformer hyperparameters.
    """
    backbone = TransformerBackbone(input_dim=in_channels, 
                                   embed_dim=embed_dim, 
                                   num_heads=num_heads, 
                                   num_layers=num_layers,
                                   seq_len=seq_len)
    return {'backbone': backbone, 'dim': embed_dim}

# Example:
# dummy_input = torch.randn(32, 55, 200)
# model_info = transformer_ts()
# model = model_info['backbone']
# out = model(dummy_input)
# print(out.shape) # should be (32, 8)

# Example usage:
# old_backbone_dict = resnet_ts(in_channels=55, mid_channels=4)  # old code
# new_backbone_dict = transformer_ts(in_channels=55, seq_len=200, embed_dim=8, num_heads=2, num_layers=2)

# Test shape:
# dummy_input = torch.randn(32, 55, 200)  # (B, C, L)
# model = new_backbone_dict['backbone']
# output = model(dummy_input)
# print(output.shape)  # Should be (32, 8)
