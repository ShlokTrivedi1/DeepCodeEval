# DeepCodeEval: Multi Model AI Code Generation & Evaluation Benchmarking

DeepCodeEval is an interactive benchmarking platform designed to compare the code generation capabilities of leading Large Language Models side by side. By ingesting context directly from any GitHub repository, the system prompts multiple AI models simultaneously and evaluates their generated code using automated DeepEval metrics.

Whether you are testing proprietary models like Claude and OpenAI or leveraging free models via Gemini and OpenRouter, DeepCodeEval gives you real time streaming comparisons and visual metrics.

## Key Features

* **Parallel Streaming Generation**: Prompt multiple models simultaneously and observe live streaming responses in a clean multi column layout.
* **Repository Context Ingestion**: Easily import code structure and context from any public GitHub repository using Gitingest.
* **Dynamic API Key Management**: Provide model API keys directly within the Streamlit sidebar during runtime or load them from environment files.
* **Automated DeepEval Benchmark**: Grade generated code across Correctness, Readability, and Best Practices with detailed reasoning and overall performance scores.
* **Visual Analytics**: Interactive Plotly bar graphs compare model scores side by side for quick analysis.

## Technology Stack

* **Streamlit**: Web interface for configuration, chat interactions, and visual analytics.
* **LiteLLM**: Unified model orchestration layer supporting Anthropic, OpenAI, Google Gemini, and OpenRouter.
* **DeepEval**: Unit testing framework for evaluating LLM outputs against tailored rubrics.
* **Gitingest**: Automated repository ingestion tool to extract context for prompt enhancement.
* **Plotly**: Data visualization library for interactive performance comparisons.

## Quick Setup Guide

Ensure Python 3.12 or higher is installed on your system.

1. Clone the repository and navigate into the workspace directory.
2. Install project dependencies using uv:

```bash
uv sync
```

3. Optional: Configure default API keys by creating a `.env` file based on `.env.example`:

```env
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

Note: API keys can also be entered securely in the application sidebar at runtime.

4. Launch the Streamlit web application:

```bash
streamlit run app.py
```

## How to Use DeepCodeEval

1. Open the sidebar configuration panel.
2. Select the AI models you wish to evaluate and enter your corresponding API keys.
3. Paste a GitHub repository URL into the Repository section and click **Ingest Repository**.
4. Type your coding task or requirement into the main chat box.
5. Watch as the selected models generate python code side by side in real time.
6. Click **Evaluate Generated Code** to trigger automated DeepEval scoring.
7. Review the generated performance charts and detailed metric breakdowns.

## Evaluation Framework

DeepCodeEval scores model outputs across three primary dimensions:

1. **Code Correctness**: Checks functional completeness, logic accuracy, and edge case handling.
2. **Code Readability**: Assesses naming conventions, formatting, structural organization, and clarity of documentation.
3. **Best Practices**: Evaluates error handling, security awareness, and modular design.

Each dimension is scored from 0 to 10. The overall benchmark score is computed as the equal average of these three scores.

## License and Contributions

Contributions, feature requests, and pull requests are welcome. Feel free to open an issue or submit your enhancements.
