"""DINOv2 ViT-S/14 embedding generator for image feature extraction with M1 optimization."""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from PIL import Image
import io

# Register HEIF/HEIC decoder so Image.open handles .heic files
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass
import numpy as np
import platform


class EmbeddingGenerator:
    """Generate 384-dimensional embeddings using the DINOv2 ViT-S/14 model
    (resized to 224x224 for speed; chosen over the larger ViT-L/14 to keep
    CPU/MPS inference fast and the vectors ~14x smaller)."""
    
    def __init__(self, device: str = None):
        """Initialize DINOv2 model for embedding generation.
        
        Args:
            device: torch device ('cuda', 'cpu', 'mps', or None for auto-detection)
        """
        # Auto-detect device with M1/Metal support
        if device is None:
            self.device = self._detect_device()
        else:
            self.device = device
        
        # Load DINOv2 ViT-S14 model (outputs 384-dim vectors, ~14x smaller than ViT-L14)
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Store model metadata
        self.model_name = 'dinov2_vits14'
        self.embedding_dim = 384
    
    def _detect_device(self) -> str:
        """Detect optimal device: MPS (M1/M2) > CUDA > CPU."""
        # Check for Apple Silicon with Metal Performance Shaders
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        # Fall back to CUDA if available
        if torch.cuda.is_available():
            return 'cuda'
        # Default to CPU
        return 'cpu'
    
    def _preprocess_image(self, image_data: bytes) -> torch.Tensor:
        """Convert image bytes to normalized tensor for model input.
        
        Args:
            image_data: Raw image bytes
        
        Returns:
            Preprocessed image tensor (1, 3, 518, 518)
        """
        # Load image from bytes
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Resize to 224x224 (smaller than DINOv2 standard 518, much faster on CPU)
        image = image.resize((224, 224), Image.Resampling.BICUBIC)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # Normalize with ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
    
    def generate_embedding(self, image_data: bytes) -> Tuple[List[float], float]:
        """Generate embedding for a single image.
        
        Args:
            image_data: Raw image bytes
        
        Returns:
            Tuple of (embedding vector as list, confidence score)
        """
        with torch.no_grad():
            # Preprocess image
            image_tensor = self._preprocess_image(image_data)
            
            # Generate embedding with device-specific optimizations
            if self.device == 'mps':
                # MPS requires explicit casting for stability
                embedding = self.model(image_tensor.float())
            else:
                embedding = self.model(image_tensor)
            
            # Calculate confidence as the norm of the embedding before normalization
            # (higher norm indicates stronger feature activation)
            confidence = float(torch.norm(embedding, p=2).item())
            
            # Normalize embedding to unit length (L2 normalization)
            embedding = F.normalize(embedding, p=2, dim=1)
            
            # Convert to list and return with confidence
            embedding_list = embedding.squeeze(0).cpu().tolist()
            return embedding_list, confidence
    
    def generate_embeddings_batch(self, image_data_list: List[bytes]) -> List[Tuple[List[float], float]]:
        """Generate embeddings for multiple images in batch.
        
        Args:
            image_data_list: List of raw image bytes
        
        Returns:
            List of tuples (embedding vector as list, confidence score)
        """
        results = []
        
        with torch.no_grad():
            # Process images in batch
            image_tensors = []
            for image_data in image_data_list:
                image_tensor = self._preprocess_image(image_data)
                image_tensors.append(image_tensor)
            
            # Stack all images into single batch tensor
            batch_tensor = torch.cat(image_tensors, dim=0)
            
            # Generate embeddings for entire batch
            embeddings = self.model(batch_tensor)
            
            # Calculate confidence scores (norm before normalization)
            confidences = torch.norm(embeddings, p=2, dim=1)
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Convert to list format with confidence scores
            for i in range(len(image_data_list)):
                embedding_list = embeddings[i].cpu().tolist()
                confidence = float(confidences[i].item())
                results.append((embedding_list, confidence))
        
        return results
    
    async def generate(self, file_path: str) -> List[float]:
        """Async wrapper: read image from path and return its embedding vector.

        Used by the job queue worker which calls this per photo via asyncio.
        The heavy lifting (preprocess + model forward) is pushed to a thread
        so it does not block the event loop.
        """
        import asyncio

        def _run() -> List[float]:
            with open(file_path, "rb") as f:
                image_data = f.read()
            embedding, _confidence = self.generate_embedding(image_data)
            return embedding

        return await asyncio.to_thread(_run)

    def get_model_info(self) -> Dict[str, any]:
        """Return metadata about the embedding model.
        
        Returns:
            Dictionary with model name and embedding dimension
        """
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device
        }

