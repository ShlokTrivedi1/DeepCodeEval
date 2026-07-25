import os
import asyncio
from litellm import acompletion
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------
# Each entry:  display_name -> (litellm_model_id, env_var_for_api_key)
MODEL_CATALOGUE = {
    "claude": {
        "display": "Claude Sonnet 5",
        "litellm_id": "claude-sonnet-5",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "display": "OpenAI GPT-5.6 Terra",
        "litellm_id": "openai/gpt-5.6-terra",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "display": "Gemini 3.6 Flash",
        "litellm_id": "gemini/gemini-3.6-flash",
        "api_key_env": "GEMINI_API_KEY",
    },
    "openrouter": {
        "display": "OpenRouter Auto-Free",
        "litellm_id": "openrouter/openrouter/free",
        "api_key_env": "OPENROUTER_API_KEY",
    },
}


async def get_model_response_async(
    model_key: str,
    prompt: str,
    context: Dict[str, Any],
    api_keys: Dict[str, str] = None,
):
    """Stream code generation from a single model. Yields text chunks.

    Args:
        model_key: Key in MODEL_CATALOGUE (e.g. "claude", "openai").
        prompt: User's code-generation prompt.
        context: Ingested repository context dict.
        api_keys: Optional mapping of model_key -> API key supplied at runtime
                  (from the UI).  Falls back to the environment variable when
                  a key is absent or empty.
    """
    user_prompt = f"""
    You are an expert code generator. Your task is to generate code based on the following repository context:

    Repository Context:
    {context['content']}

    Instructions:
    1. Generate code that strictly follows the repository's existing patterns and conventions
    2. Use the same coding style, naming conventions, and structure as the codebase
    3. Include clear, concise docstrings and comments explaining key functionality
    4. Ensure the code integrates seamlessly with existing components
    5. Focus on maintainability and readability

    User query:
    {prompt}

    Output only the code implementation without explanations or additional text.
    """

    messages = [
        {"role": "user", "content": user_prompt}
    ]

    meta = MODEL_CATALOGUE[model_key]

    # Prefer runtime key supplied from the UI; fall back to .env / environment.
    runtime_key = (api_keys or {}).get(model_key, "").strip()
    api_key = runtime_key if runtime_key else os.getenv(meta["api_key_env"])

    if not api_key:
        yield (
            f"⚠️ No API key found for **{meta['display']}**.\n"
            f"Please enter your `{meta['api_key_env']}` in the sidebar."
        )
        return

    try:
        # Get streaming response from the model using LiteLLM asynchronously.
        response = await acompletion(
            model=meta["litellm_id"],
            messages=messages,
            api_key=api_key,
            max_tokens=2000,
            stream=True
        )
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        yield f"Error generating response: {str(e)}"


def get_parallel_responses(
    prompt: str,
    context: Dict[str, Any],
    selected_models: List[str] = None,
    api_keys: Dict[str, str] = None,
) -> Dict[str, Any]:
    """
    Return a dict of {model_key: async_generator} for the selected models.
    This is a plain (non-async) function — generators are created lazily;
    no I/O happens until the caller iterates each generator.
    Defaults to all four models when selected_models is None.

    Args:
        api_keys: Optional mapping of model_key -> API key supplied at runtime
                  (from the UI). Forwarded to each model generator.
    """
    if selected_models is None:
        selected_models = list(MODEL_CATALOGUE.keys())

    return {
        model_key: get_model_response_async(model_key, prompt, context, api_keys)
        for model_key in selected_models
    }
