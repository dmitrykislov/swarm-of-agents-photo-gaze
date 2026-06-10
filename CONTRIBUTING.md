# Contributing Guide



Thank you for your interest in contributing to the Photo Similarity Finder! This guide will help you get started with development and testing.



## Development Setup



### Prerequisites



- Python 3.9+

- Node.js 18+

- Docker & Docker Compose (recommended)

- Git



### Backend Development



```bash

# Clone repository

git clone <repo-url>

cd <repo-dir>



# Create virtual environment

python3 -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate



# Install dependencies

pip install -r requirements.txt

pip install -r requirements-dev.txt



# Set up environment

cp .env.example .env

# Edit .env with your local database and Qdrant URLs



# Start PostgreSQL and Qdrant (using Docker)

docker-compose up -d postgres qdrant



# Run migrations

alembic upgrade head



# Start backend server

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

```



### Frontend Development



```bash

# Navigate to frontend directory

cd frontend



# Install dependencies

npm install



# Start development server

npm run dev



# Frontend will be available at http://localhost:5173

```



## Code Style



### Python



- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)

- Use type hints for all function parameters and return values

- Maximum line length: 100 characters

- Use docstrings for all public functions and classes



```python

def process_photo(file_path: str, threshold: float) -> dict:

    """Process a photo and return metadata.

    

    Args:

        file_path: Path to the photo file

        threshold: Similarity threshold (0-1)

    

    Returns:

        Dictionary with photo metadata and embedding

    

    Raises:

        FileNotFoundError: If file does not exist

        ValueError: If threshold is out of range

    """

    pass

```



### TypeScript/React



- Use TypeScript for all new code

- Use functional components with hooks

- Use meaningful variable and function names

- Maximum line length: 100 characters

- Use JSDoc comments for complex functions



```typescript

/**

 * Fetch similarity groups from the backend.

 * @param threshold - Similarity threshold (0-1)

 * @returns Promise resolving to array of similarity groups

 * @throws Error if request fails

 */

export async function fetchSimilarityGroups(threshold: number): Promise<SimilarityGroup[]> {

  // Implementation

}

```



## Testing



### Backend Tests



All new features must include unit and integration tests.



```bash

# Run all tests

pytest -v



# Run specific test file

pytest tests/test_workflow_integration.py -v



# Run with coverage

pytest --cov=app --cov-report=html



# Run in watch mode

pytest-watch

```



**Test Structure**:



```python

import pytest

from app.module import function_to_test



class TestFunctionToTest:

    """Test suite for function_to_test."""

    

    def test_happy_path(self):

        """Test successful execution."""

        result = function_to_test(valid_input)

        assert result == expected_output

    

    def test_error_handling(self):

        """Test error handling."""

        with pytest.raises(ValueError):

            function_to_test(invalid_input)

```



### Frontend Tests



```bash

# Run all tests

npm test



# Run in watch mode

npm run test:watch



# Run with coverage

npm test -- --coverage

```



**Test Structure**:



```typescript

import { render, screen } from '@testing-library/react';

import { MyComponent } from './MyComponent';



describe('MyComponent', () => {

  it('renders correctly', () => {

    render(<MyComponent />);

    expect(screen.getByText('Expected text')).toBeInTheDocument();

  });

});

```



## Git Workflow



### Branch Naming



- `feature/description` - New features

- `bugfix/description` - Bug fixes

- `docs/description` - Documentation updates

- `refactor/description` - Code refactoring



### Commit Messages



Use clear, descriptive commit messages:



```

feat: Add similarity threshold slider



- Implement ThresholdInput component

- Add threshold validation (0-1)

- Pass the threshold to the /similarity-groups query (min_similarity)

- Add tests for threshold filtering



Closes #123

```



### Pull Request Process



1. Create a feature branch: `git checkout -b feature/my-feature`

2. Make your changes and commit: `git commit -m "feat: description"`

3. Push to your fork: `git push origin feature/my-feature`

4. Open a pull request with:

   - Clear description of changes

   - Reference to related issues

   - Screenshots (for UI changes)

   - Test results

5. Address review feedback

6. Merge when approved



## Documentation



### Code Comments



Write comments that explain **why** code exists, not what it does:



```python

# ✓ Good: Explains the reason

# Use cosine similarity instead of Euclidean distance

# because embeddings are normalized and cosine is more efficient

similarity = cosine_similarity(embedding1, embedding2)



# ✗ Bad: Just describes what the code does

# Calculate cosine similarity

similarity = cosine_similarity(embedding1, embedding2)

```



### Documentation Files



- Update `README.md` for user-facing changes

- Update `docs/API.md` for API endpoint changes

- Update `docs/ARCHITECTURE.md` for architectural changes

- Update `docs/TROUBLESHOOTING.md` for known issues



## Performance Considerations



### Backend



- Use async/await for I/O operations

- Batch database queries when possible

- Cache expensive computations

- Profile code with `cProfile` for bottlenecks



### Frontend



- Use React.memo for expensive components

- Implement virtualization for large lists

- Lazy load images and components

- Monitor bundle size with `npm run build`



## Security



- Never commit secrets or credentials

- Use environment variables for sensitive data

- Validate all user input

- Use parameterized queries (SQLAlchemy ORM)

- Keep dependencies up to date: `pip list --outdated`



## Debugging



### Backend



```python

# Use pdb for interactive debugging

import pdb; pdb.set_trace()



# Or use breakpoint() (Python 3.7+)

breakpoint()



# View logs

docker-compose logs -f backend

```



### Frontend



```typescript

// Use browser DevTools

console.log('Debug message:', variable);

debugger;  // Pause execution



// View logs

npm run dev  // Check terminal output

```



## Reporting Issues



When reporting bugs, include:



1. **Description**: What is the issue?

2. **Steps to reproduce**: How can we reproduce it?

3. **Expected behavior**: What should happen?

4. **Actual behavior**: What actually happens?

5. **Environment**: OS, Python/Node version, Docker version

6. **Logs**: Relevant error messages and stack traces

7. **Screenshots**: For UI issues



## Questions?



- Check existing issues and documentation

- Ask in project discussions

- Open an issue with the `question` label

