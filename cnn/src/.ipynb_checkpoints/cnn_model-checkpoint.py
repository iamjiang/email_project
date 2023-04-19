import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler

class CNN_Model(nn.Module):
    def __init__(self, batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab, vocab_size, embedding_length, \
                 pretrained_emb, mode):
        super(CNN_Model, self).__init__()
        """
        Arguments
        ---------
        batch_size : Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
        out_channels : Number of output channels after convolution operation performed on the input matrix
        kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
        keep_probab : Probability of retaining an activation node during dropout operation
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embedding dimension of GloVe word embeddings
        pretrained_emb : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
        --------
        """
        self.batch_size = batch_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.stride = stride
        self.padding = padding
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.mode=mode

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
#         self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
        if self.mode=="rand":
            rand_embed_init=torch.Tensor(vocab_size, embedding_length).uniform_(-0.25,0.25)
            self.word_embeddings.weight.data.copy_(rand_embed_init)
            self.word_embeddings.weight.requires_grad = True
        elif self.mode=="static":
            self.word_embeddings.weight.data.copy_(pretrained_emb)
            self.word_embeddings.weight.requires_grad = False
        elif self.mode=="non-static":
            self.word_embeddings.weight.data.copy_(pretrained_emb)
            self.word_embeddings.weight.requires_grad = True
        elif self.mode=="multi-channel":
            self.static_embed=nn.Embedding.from_pretrained(pretrained_emb, freeze=True)
            self.non_static_embed=nn.Embedding.from_pretrained(pretrained_emb, freeze=False)
        else:
            print("Unsupport Mode")
            exit()

        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length), stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride, padding)
        # self.conv4 = nn.Conv2d(in_channels, out_channels, (kernel_heights[3], embedding_length), stride, padding)
        self.dropout = nn.Dropout(keep_probab)
        self.label = nn.Linear(len(kernel_heights)*out_channels, output_size)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)

        return max_out
    
    def forward(self, input_sentences,device):

        """
        The idea of the Convolutional Neural Netwok for Text Classification is very simple. We perform convolution operation on the embedding matrix 
        whose shape for each batch is (num_seq, embedding_length) with kernel of varying height but constant width which is same as the embedding_length.
        We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor 
        and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors. This output is then fully connected
        to the output layers consisting two units which basically gives us the logits for both positive and negative classes.
        Parameters
        ----------
        input_sentences: input_sentences of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.
        logits.size() = (batch_size, output_size)
        """
        if self.mode in ["rand","static","non-static"]: 
            input = self.word_embeddings(input_sentences)
            # input.size() = (batch_size, num_seq, embedding_length)
            input = input.unsqueeze(1).to(device)
            # input.size() = (batch_size, 1, num_seq, embedding_length)
        elif self.mode=="multi-channel":
            static_input=self.static_embed(input_sentences)
            non_static_input=self.non_static_embed(input_sentences)
            input=torch.stack([static_input,non_static_input],dim=1).to(device)
            # input.size() = (batch_size, input_channel=2, num_seq, embedding_length)
        else:
            print("Unsupport Mode")
            exit()

        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)
        # max_out4 = self.conv_block(input, self.conv4)
        
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        # all_out = torch.cat((max_out1, max_out2, max_out3, max_out4), 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        # fc_in.size()) = (batch_size, num_kernels*out_channels)
        logits = self.label(fc_in)

        return logits
    