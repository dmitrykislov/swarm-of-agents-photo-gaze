"""Unit tests for DINOv2 embedding generator."""
import pytest
import torch
from PIL import Image
import io
import numpy as np
from app.embedding_generator import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test DINOv2 embedding generation functionality."""
    
    @pytest.fixture
    def generator(self):
        """Provide an EmbeddingGenerator instance for testing."""
        return EmbeddingGenerator(device='cpu')
    
    @pytest.fixture
    def sample_image_bytes(self):
        """Create a sample image as bytes for testing."""
        # Create a simple RGB image (100x100 pixels)
        image = Image.new('RGB', (100, 100), color=(73, 109, 137))
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        return image_bytes.getvalue()
    
    @pytest.fixture
    def sample_image_bytes_list(self, sample_image_bytes):
        """Create multiple sample images for batch testing."""
        images = []
        for i in range(3):
            # Create images with different colors
            color = (50 + i * 50, 100 + i * 30, 150 - i * 40)
            image = Image.new('RGB', (100, 100), color=color)
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            images.append(image_bytes.getvalue())
        return images
    
    @pytest.mark.unit
    def test_model_initialization(self, generator):
        """Verify DINOv2 model loads successfully."""
        assert generator.model is not None
        assert generator.model_name == 'dinov2_vits14'
        assert generator.embedding_dim == 384
    
    @pytest.mark.unit
    def test_model_device_detection(self):
        """Verify device detection works correctly."""
        generator = EmbeddingGenerator(device='cpu')
        assert generator.device == 'cpu'
    
    @pytest.mark.unit
    def test_get_model_info(self, generator):
        """Verify model metadata is accessible."""
        info = generator.get_model_info()
        assert info['model_name'] == 'dinov2_vits14'
        assert info['embedding_dim'] == 384
        assert info['device'] == 'cpu'
    
    @pytest.mark.unit
    def test_generate_embedding_returns_correct_dimension(self, generator, sample_image_bytes):
        """Verify single image embedding has correct dimension (384)."""
        embedding, confidence = generator.generate_embedding(sample_image_bytes)
        assert isinstance(embedding, list)
        assert len(embedding) == 384
    
    @pytest.mark.unit
    def test_generate_embedding_returns_confidence_score(self, generator, sample_image_bytes):
        """Verify confidence score is returned and is a positive float."""
        embedding, confidence = generator.generate_embedding(sample_image_bytes)
        assert isinstance(confidence, float)
        assert confidence > 0
    
    @pytest.mark.unit
    def test_generate_embedding_normalized(self, generator, sample_image_bytes):
        """Verify embedding is L2-normalized (norm close to 1)."""
        embedding, confidence = generator.generate_embedding(sample_image_bytes)
        embedding_tensor = torch.tensor(embedding)
        norm = torch.norm(embedding_tensor, p=2).item()
        # L2 norm should be close to 1 (allowing small floating point error)
        assert 0.99 < norm < 1.01
    
    @pytest.mark.unit
    def test_generate_embeddings_batch_returns_correct_count(self, generator, sample_image_bytes_list):
        """Verify batch processing returns correct number of embeddings."""
        results = generator.generate_embeddings_batch(sample_image_bytes_list)
        assert len(results) == 3
    
    @pytest.mark.unit
    def test_generate_embeddings_batch_each_has_correct_dimension(self, generator, sample_image_bytes_list):
        """Verify each batch embedding has correct dimension."""
        results = generator.generate_embeddings_batch(sample_image_bytes_list)
        for embedding, confidence in results:
            assert isinstance(embedding, list)
            assert len(embedding) == 384
            assert isinstance(confidence, float)
            assert confidence > 0
    
    @pytest.mark.unit
    def test_generate_embeddings_batch_normalized(self, generator, sample_image_bytes_list):
        """Verify batch embeddings are L2-normalized."""
        results = generator.generate_embeddings_batch(sample_image_bytes_list)
        for embedding, confidence in results:
            embedding_tensor = torch.tensor(embedding)
            norm = torch.norm(embedding_tensor, p=2).item()
            assert 0.99 < norm < 1.01
    
    @pytest.mark.unit
    def test_different_images_produce_different_embeddings(self, generator, sample_image_bytes_list):
        """Verify different images produce different embeddings."""
        results = generator.generate_embeddings_batch(sample_image_bytes_list)
        embeddings = [r[0] for r in results]
        
        # Calculate cosine similarity between first two embeddings
        emb1 = torch.tensor(embeddings[0])
        emb2 = torch.tensor(embeddings[1])
        similarity = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        
        # Different images should have similarity < 1.0 (not identical)
        assert similarity < 1.0
    
    @pytest.mark.unit
    def test_same_image_produces_same_embedding(self, generator, sample_image_bytes):
        """Verify same image produces identical embeddings."""
        embedding1, conf1 = generator.generate_embedding(sample_image_bytes)
        embedding2, conf2 = generator.generate_embedding(sample_image_bytes)
        
        # Embeddings should be identical (deterministic model in eval mode)
        assert embedding1 == embedding2
        assert conf1 == conf2
    
    @pytest.mark.unit
    def test_batch_vs_single_consistency(self, generator, sample_image_bytes):
        """Verify batch processing produces same results as single processing."""
        # Single processing
        single_embedding, single_conf = generator.generate_embedding(sample_image_bytes)
        
        # Batch processing with one image
        batch_results = generator.generate_embeddings_batch([sample_image_bytes])
        batch_embedding, batch_conf = batch_results[0]
        
        # Results should be identical
        assert single_embedding == batch_embedding
        assert single_conf == batch_conf

