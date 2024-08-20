
# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/


import torch
import torch.nn as nn
import torch.nn.functional as F


def load_model(model_path, config, device):
    hidden_dim = config['hidden_dim']
    n_layers = config['n_layers']
    dropout = config['dropout']
    num_heads = config['num_heads']

    encoder = Encoder(config['input_dim'], hidden_dim, n_layers, num_heads)
    decoder = Decoder(config['output_dim'], hidden_dim, n_layers, num_heads)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    return model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src):
        with torch.no_grad(): 
            encoder_outputs = self.encoder(src)
            output = self.decoder(encoder_outputs)
        return output

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, num_heads, use_norm=False):
        super(Encoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)  
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.use_norm = use_norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        outputs = self.transformer_encoder(x)
        outputs = outputs.transpose(0, 1)
        
        if self.use_norm:
            outputs = self.layer_norm(outputs)
        
        return outputs  

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, num_heads, use_norm=False):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.cross_attention = MultiHeadAttention(hidden_dim, num_heads)  
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads) 
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.use_norm = use_norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, encoder_outputs):
        input = encoder_outputs.transpose(0, 1)
        input = self.pos_encoder(input)
        cross_attn_output, _ = self.cross_attention(input, encoder_outputs, encoder_outputs)
        decoder_input = input + cross_attn_output
        decoder_output = self.transformer_decoder(decoder_input, encoder_outputs.transpose(0, 1))
        decoder_output = decoder_output.transpose(0, 1)
        
        if self.use_norm:
            decoder_output = self.layer_norm(decoder_output)
       
        prediction = self.fc_output(decoder_output)
        
        return prediction

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by the number of heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Flash Attention requires PyTorch >= 2.0")

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if self.flash:
            attn_output = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0)
            attn_weights = None
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_linear(attn_output)
        
        return output, attn_weights