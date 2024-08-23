#!/usr/bin/env python3

import torch


def make_embedder(
    architecture: str='GPT',
    training_style: str='CSM',
    in_dim: int=1024,
    embed_dim: int=768,
    num_hidden_layers: int=1,
    dropout: float=0.1,
    n_positions: int=512
    ) -> torch.nn.Module:
    """
    Make an embedder object.
    
    The embedder is used to prepare an input batch 
    (as generated by src.batcher) for training and 
    compute the model's training loss, given the 
    specified training style.

    Args:
    -----
    architecture: str
        The model architecture to use.
        One of: 'GPT', 'BERT', 'NetBERT', autoencoder',
        'PretrainedGPT', 'PretrainedBERT', 'LinearBaseline'.
    training_style: str
        The used training style (ie., framework).
        One of: 'BERT', 'CSM', 'NetBERT', 'autoencoder',
        'decoding'.
    in_dim: int
        The input dimension (ie., # networks) of the
        parcelated BOLD data.
    embed_dim: int
        The dimension of the used embedding space.
    num_hidden_layers: int
        The number of hidden layers of the embedding
        model. If more than one layers are used, all
        layers except the last one are activated through
        Gelu activation (see src.base.EmbeddingModel).
    dropout: float
        Dropout rate used emebdding model.
    n_positions: int
        The maximum number of sequence elements that
        the model can handle (in sequence elements).

    Core methods: 
    -----
    prep_batch(batch):  
        Makes all training-style specific edits of input batch 
        (as generated by src.batcher); 
        i.e., projection of input BOLD sequences into an 
        embedding space (as defined by embed_dim) 
        and addition of all training-style specific tokens to 
        the input data 
    
    loss(batch, outputs):
        Compute the training-style specific loss,
        given batch (as generated by prep_batch) and 
        the the full model's (see src.model) output 
        (as generated by model.forward) 

    switch_decoding_mode(is_decoding_mode):
        Switch the embedder to decoding mode (is_decoding_mode=True).
        This function is needed to adapt a pre-trained model
        to a downstream decoding task.
    """

    kwargs = {
        "in_dim": in_dim,
        "embed_dim": embed_dim,
        "num_hidden_layers": num_hidden_layers,
        "dropout": dropout,
        "n_positions": n_positions
    }

    if training_style == 'CSM_causal':
        from embedder.csm_causal import CSMEmbedder
        embedder = CSMEmbedder(**kwargs)

    elif training_style == 'CSM':
        from embedder.csm import CSMEmbedder
        embedder = CSMEmbedder(**kwargs)
    
    elif training_style == 'decoding':

        if architecture in {'GPT', 'PretrainedGPT2','DistilledGPT2'}:
            from embedder.csm import CSMEmbedder
            embedder = CSMEmbedder(**kwargs)
        
        else:
            raise ValueError('unkown architecture')

    else:
        raise ValueError('unknown training style.')
    
    return embedder
