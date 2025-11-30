"""
Text embedding utilities for category names.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Union

try:
    from transformers import BertTokenizer, BertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class CategoryTextEmbedder:
    """
    A utility class for generating text embeddings for category names.
    Supports multiple embedding types: BERT, CLIP, and GloVe.
    """
    
    def __init__(self, 
                 embedding_type: str = "bert",
                 embed_dim: int = 512,
                 device: str = "cpu"):
        """
        Initialize the category text embedder.
        
        Args:
            embedding_type: Type of embedding to use ("bert", "clip", "glove")
            embed_dim: Dimension of the output embedding
            device: Device to run the models on
        """
        self.embedding_type = embedding_type.lower()
        self.embed_dim = embed_dim
        self.device = device
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize the embedding models based on the embedding type."""
        if self.embedding_type == "bert":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers is required for BERT embeddings")
            
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.model = BertModel.from_pretrained("bert-base-cased")
            self.model.to(self.device)
            
            # Freeze the model parameters
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Projection layer to match desired embedding dimension
            self.projection = nn.Linear(768, self.embed_dim).to(self.device)
            
        elif self.embedding_type == "clip":
            if not CLIP_AVAILABLE:
                raise ImportError("clip is required for CLIP embeddings")
            
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            
            # Freeze the model parameters
            for param in self.model.parameters():
                param.requires_grad = False
                
            # CLIP text encoder outputs 512-dim embeddings
            if self.embed_dim != 512:
                self.projection = nn.Linear(512, self.embed_dim).to(self.device)
            else:
                self.projection = nn.Identity()
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")
    
    def embed_category_names(self, category_names: List[str]) -> torch.Tensor:
        """
        Generate embeddings for a list of category names.
        
        Args:
            category_names: List of category names to embed
            
        Returns:
            torch.Tensor: Embeddings of shape (len(category_names), embed_dim)
        """
        if self.embedding_type == "bert":
            return self._embed_with_bert(category_names)
        elif self.embedding_type == "clip":
            return self._embed_with_clip(category_names)
        elif self.embedding_type == "glove":
            return self._embed_with_glove(category_names)
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")
    
    def _embed_with_bert(self, category_names: List[str]) -> torch.Tensor:
        """Generate embeddings using BERT."""
        embeddings = []
        
        with torch.no_grad():
            for category in category_names:
                # Tokenize the category name
                inputs = self.tokenizer(
                    category, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True,
                    max_length=64
                ).to(self.device)
                
                # Get BERT embeddings
                outputs = self.model(**inputs)
                # Use [CLS] token representation
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (1, 768)
                
                # Project to desired dimension
                projected = self.projection(cls_embedding)  # Shape: (1, embed_dim)
                embeddings.append(projected)
        
        return torch.cat(embeddings, dim=0).cpu().numpy()  # Shape: (len(category_names), embed_dim)
    
    def _embed_with_clip(self, category_names: List[str]) -> torch.Tensor:
        """Generate embeddings using CLIP."""
        embeddings = []
        
        with torch.no_grad():
            for category in category_names:
                # Prepare text input for CLIP
                text_input = clip.tokenize([f"a {category}"]).to(self.device)
                
                # Get CLIP text embeddings
                text_features = self.model.encode_text(text_input)
                text_features = text_features.float()  # Shape: (1, 512)
                
                # Project to desired dimension
                projected = self.projection(text_features)  # Shape: (1, embed_dim)
                embeddings.append(projected)
        
        return torch.cat(embeddings, dim=0)  # Shape: (len(category_names), embed_dim)
    
    def _embed_with_glove(self, category_names: List[str]) -> torch.Tensor:
        """Generate embeddings using GloVe."""
        embeddings = []
        
        for category in category_names:
            # Split category name into words and get GloVe embeddings
            words = category.lower().split()
            word_embeddings = []
            
            for word in words:
                if hasattr(self.glove, 'stoi') and word in self.glove.stoi:
                    word_embeddings.append(self.glove[word])
                else:
                    # Use zero vector for unknown words
                    word_embeddings.append(torch.zeros(50))
            
            if word_embeddings:
                # Average word embeddings
                category_embedding = torch.stack(word_embeddings).mean(dim=0)
            else:
                # Use zero vector if no words found
                category_embedding = torch.zeros(50)
            
            # Project to desired dimension
            projected = self.projection(category_embedding.unsqueeze(0).to(self.device))
            embeddings.append(projected)
        
        return torch.cat(embeddings, dim=0)  # Shape: (len(category_names), embed_dim)
    
    def create_category_embedding_dict(self, category_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        Create a dictionary mapping category names to their embeddings.
        
        Args:
            category_names: List of category names
            
        Returns:
            Dict mapping category names to their embeddings
        """
        embeddings = self.embed_category_names(category_names)
        return {
            name: embedding 
            for name, embedding in zip(category_names, embeddings)
        }


def get_category_text_embeddings(
    category_names: List[str],
    embedding_type: str = "bert",
    embed_dim: int = 512,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Convenience function to get text embeddings for category names.
    
    Args:
        category_names: List of category names to embed
        embedding_type: Type of embedding to use ("bert", "clip", "glove")
        embed_dim: Dimension of the output embedding
        device: Device to run the models on
        
    Returns:
        torch.Tensor: Embeddings of shape (len(category_names), embed_dim)
    """
    embedder = CategoryTextEmbedder(
        embedding_type=embedding_type,
        embed_dim=embed_dim,
        device=device
    )
    return embedder.embed_category_names(category_names) 