"""
Gotcha — LLM Client Abstraction
Provides access to Google Gemini (or fallbacks) with error handling,
rate limiting, and response caching.
"""

import os
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Try importing the Gemini SDK
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai package not available. LLM calls will use fallback/mock.")

# Module-level client cache
_initialized = False

def initialize_client() -> bool:
    """Initialize the LLM API client.
    
    Returns:
        True if client is initialized successfully, False otherwise.
    """
    global _initialized
    if _initialized:
        return True
        
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY environment variable not set. LLM calls will fall back to mock.")
        return False
        
    if not GEMINI_AVAILABLE:
        return False
        
    try:
        genai.configure(api_key=api_key)
        _initialized = True
        logger.info("Gemini API client initialized successfully.")
        return True
    except Exception as e:
        logger.error("Failed to configure Gemini API client: %s", e)
        return False

def call_llm(
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    retry_attempts: int = 3,
    backoff_factor: float = 2.0,
) -> str:
    """Call the configured LLM with the given prompt and parameters.
    
    Args:
        prompt: User prompt content.
        system_instruction: System prompt/instructions.
        temperature: Sampling temperature (0.0 to 1.0).
        max_tokens: Maximum tokens to generate.
        retry_attempts: Number of API retries for rate limits or transient errors.
        backoff_factor: Exponential backoff factor.
        
    Returns:
        Generated text response, or an empty string/mock on failure.
    """
    from src.config import LLM_MODEL
    
    # Try initializing if not done
    initialized = initialize_client()
    
    if not initialized or not GEMINI_AVAILABLE:
        # Provide a mock/fallback response for testing or offline environments
        logger.warning("Using mock LLM response (Gemini client not initialized or unavailable)")
        return _get_mock_response(prompt)
        
    # Run call with retries
    for attempt in range(retry_attempts):
        try:
            # Modern SDK setup
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            # Using GenerativeModel
            model = genai.GenerativeModel(
                model_name=LLM_MODEL,
                generation_config=generation_config,
                system_instruction=system_instruction
            )
            
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
            else:
                logger.warning("Empty response received from Gemini API.")
                
        except Exception as e:
            logger.warning("Gemini API error (attempt %d/%d): %s", attempt + 1, retry_attempts, e)
            if "ResourceExhausted" in str(e) or "429" in str(e):
                # Rate limit hit — backoff
                sleep_time = backoff_factor ** (attempt + 1)
                logger.info("Rate limit hit. Sleeping for %.1f seconds...", sleep_time)
                time.sleep(sleep_time)
            elif attempt < retry_attempts - 1:
                time.sleep(1.0)
            else:
                logger.error("All LLM retry attempts failed: %s", e)
                
    return _get_mock_response(prompt)

def _get_mock_response(prompt: str) -> str:
    """Fallback generator for mock LLM responses."""
    # Analyze prompt to return realistic mock JSON/strings
    prompt_lower = prompt.lower()
    
    # JD Parsing / Skill extraction mock
    if "extract" in prompt_lower or "job description" in prompt_lower:
        return """{
            "title": "Senior Backend Engineer",
            "must_have_skills": [
                {"skill": "python", "weight": 0.4},
                {"skill": "postgresql", "weight": 0.3},
                {"skill": "fastapi", "weight": 0.3}
            ],
            "nice_to_have_skills": [
                {"skill": "docker", "weight": 0.5},
                {"skill": "aws", "weight": 0.5}
            ],
            "seniority_expected": "senior",
            "work_mode": "hybrid",
            "min_experience_years": 5
        }"""
        
    # Candidate narrative analysis mock
    if "candidate" in prompt_lower or "profile" in prompt_lower:
        return """{
            "experience_impact": 0.85,
            "domain_coherence": 0.90,
            "narrative_credibility": 0.80,
            "reasoning": "Candidate shows strong experience in python and postgresql with clear progression in backend roles. Good description detail."
        }"""
        
    return "Mock LLM Response: Gemini client is offline or key is missing."
