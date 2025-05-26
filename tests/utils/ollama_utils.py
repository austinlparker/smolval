"""Utilities for Ollama integration testing."""

import subprocess
import time
from typing import Optional

import requests


def is_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running at the specified URL."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def get_available_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Get list of available models in Ollama."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            return [model["name"] for model in models.get("models", [])]
        return []
    except requests.RequestException:
        return []


def is_model_available(model_name: str, base_url: str = "http://localhost:11434") -> bool:
    """Check if a specific model is available in Ollama."""
    available_models = get_available_models(base_url)
    return model_name in available_models


def pull_model_if_needed(model_name: str, base_url: str = "http://localhost:11434") -> bool:
    """Pull a model if it's not already available."""
    if is_model_available(model_name, base_url):
        return True
    
    try:
        # Use ollama CLI to pull the model
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for model download
        )
        
        if result.returncode == 0:
            # Wait a moment for the model to be registered
            time.sleep(2)
            return is_model_available(model_name, base_url)
        
        return False
    except (subprocess.SubprocessError, subprocess.TimeoutExpired):
        return False


def start_ollama_if_needed() -> bool:
    """Start Ollama if it's not running (requires ollama to be installed)."""
    if is_ollama_running():
        return True
    
    try:
        # Try to start ollama serve in background
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait for service to start
        for _ in range(10):  # Wait up to 10 seconds
            time.sleep(1)
            if is_ollama_running():
                return True
        
        return False
    except subprocess.SubprocessError:
        return False


def setup_ollama_for_testing(
    model_name: str = "qwen2.5:3b",
    base_url: str = "http://localhost:11434",
    auto_start: bool = False,
    auto_pull: bool = False
) -> tuple[bool, str]:
    """
    Set up Ollama for testing.
    
    Returns:
        tuple[bool, str]: (success, error_message)
    """
    # Check if Ollama is running
    if not is_ollama_running(base_url):
        if auto_start:
            if not start_ollama_if_needed():
                return False, "Failed to start Ollama service"
        else:
            return False, f"Ollama not running at {base_url}. Start with: ollama serve"
    
    # Check if model is available
    if not is_model_available(model_name, base_url):
        if auto_pull:
            if not pull_model_if_needed(model_name, base_url):
                return False, f"Failed to pull model {model_name}"
        else:
            return False, f"Model {model_name} not available. Pull with: ollama pull {model_name}"
    
    return True, "Ollama setup successful"


def get_model_info(model_name: str, base_url: str = "http://localhost:11434") -> Optional[dict]:
    """Get detailed information about a model."""
    try:
        response = requests.post(
            f"{base_url}/api/show",
            json={"name": model_name},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None


def test_model_generation(
    model_name: str,
    prompt: str = "Hello! Please respond with just 'OK'.",
    base_url: str = "http://localhost:11434"
) -> tuple[bool, str]:
    """Test that a model can generate responses."""
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "")
            return True, generated_text
        else:
            return False, f"API error: {response.status_code}"
    
    except requests.RequestException as e:
        return False, f"Request failed: {e}"