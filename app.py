import asyncio
import nest_asyncio
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from model_service import get_parallel_responses, MODEL_CATALOGUE
from code_ingestion import ingest_github_repo
from code_evaluation import evaluate_code

# Allow asyncio.run / run_until_complete to work inside Streamlit's event loop
nest_asyncio.apply()

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Code Generation Model Comparison",
    layout="wide"
)

# Custom CSS for responsive code containers
st.markdown("""
<style>
    .stMarkdown {
        width: 100%;
    }
    pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        max-width: 100% !important;
    }
    code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        max-width: 100% !important;
    }
    .streamlit-expanderContent {
        width: 100% !important;
    }
    div[data-testid="stCodeBlock"] {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        max-width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper: model keys in a stable order
# ---------------------------------------------------------------------------
ALL_MODEL_KEYS = list(MODEL_CATALOGUE.keys())   # ["claude","openai","gemini","openrouter"]

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'context' not in st.session_state:
    st.session_state.context = None
if 'reference_code' not in st.session_state:
    st.session_state.reference_code = None
if 'last_generated_code' not in st.session_state:
    st.session_state.last_generated_code = {k: None for k in ALL_MODEL_KEYS}
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = {k: None for k in ALL_MODEL_KEYS}
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = ALL_MODEL_KEYS[:]   # default: all selected
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {k: "" for k in ALL_MODEL_KEYS}  # runtime API keys

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    # --- Model selection ---
    st.write("### 🤖 Select Models to Compare")
    st.caption("Pick at least 1 model. Free options: Gemini & OpenRouter.")

    selected = []
    for key in ALL_MODEL_KEYS:
        meta = MODEL_CATALOGUE[key]
        checked = st.checkbox(
            meta["display"],
            value=(key in st.session_state.selected_models),
            key=f"chk_{key}"
        )
        if checked:
            selected.append(key)
            # API key input shown right below the checkbox (collapsed by default)
            env_var = meta["api_key_env"]
            import os as _os
            has_env = bool(_os.getenv(env_var))
            placeholder = "Set via .env ✓" if has_env else f"Enter {env_var}…"
            entered = st.text_input(
                f"🔑 {env_var}",
                value=st.session_state.api_keys.get(key, ""),
                type="password",
                placeholder=placeholder,
                key=f"apikey_{key}",
                help=(
                    f"Paste your API key for {meta['display']} here. "
                    "It is stored only in your browser session and never saved to disk."
                ),
            )
            st.session_state.api_keys[key] = entered

    if len(selected) < 1:
        st.warning("Select at least 1 model.")
    else:
        st.session_state.selected_models = selected

    st.divider()

    # --- Repository ingestion ---
    st.write("### 📦 Repository")
    github_repo = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/username/repository"
    )

    if st.button("Ingest Repository"):
        if github_repo:
            with st.spinner("Ingesting repository..."):
                st.session_state.context = ingest_github_repo(github_repo)
            st.success("Repository ingested successfully!")
        else:
            st.error("Please enter a valid repository URL")

    st.session_state.reference_code = st.text_area(
        "Reference Code (Optional)",
        help="Enter reference/ground truth code to compare against",
        height=200
    )

    # --- Evaluation ---
    st.write("### 📊 Evaluation")
    active_models = st.session_state.selected_models
    all_generated = all(
        st.session_state.last_generated_code.get(k) for k in active_models
    )

    if st.button("Evaluate Generated Code"):
        if all_generated:
            with st.spinner("Evaluating code..."):
                for k in active_models:
                    meta = MODEL_CATALOGUE[k]
                    
                    # Fetch the API key (prefer runtime sidebar key, fallback to env)
                    runtime_key = st.session_state.api_keys.get(k, "").strip()
                    import os as _os
                    api_key = runtime_key if runtime_key else _os.getenv(meta["api_key_env"])

                    st.session_state.evaluation_results[k] = evaluate_code(
                        st.session_state.last_generated_code[k],
                        st.session_state.reference_code if st.session_state.reference_code else None,
                        model_id=meta["litellm_id"],
                        api_key=api_key
                    )
            st.success("Evaluation complete!")
        else:
            missing = [MODEL_CATALOGUE[k]["display"] for k in active_models
                       if not st.session_state.last_generated_code.get(k)]
            st.error(f"No generated code yet for: {', '.join(missing)}")

    # --- API key status summary ---
    st.divider()
    st.write("### 🔑 API Key Status")
    import os as _os
    for k in active_models:
        meta = MODEL_CATALOGUE[k]
        env_var = meta["api_key_env"]
        runtime = st.session_state.api_keys.get(k, "").strip()
        from_env = bool(_os.getenv(env_var))
        if runtime:
            st.success(f"✅ {meta['display']}: key entered in sidebar")
        elif from_env:
            st.info(f"ℹ️ {meta['display']}: using `{env_var}` from .env")
        else:
            st.error(f"❌ {meta['display']}: no API key — enter it above!")


# ---------------------------------------------------------------------------
# Async chat handler
# ---------------------------------------------------------------------------
async def handle_chat_input_async(prompt: str):
    """Core async logic: stream responses from all selected models in parallel."""
    active_models = st.session_state.selected_models
    api_keys = st.session_state.api_keys  # runtime keys from sidebar

    with st.chat_message("assistant"):
        # Create one column per model
        cols = st.columns(len(active_models))
        # Store st.empty() placeholders (NOT the return value of .code())
        placeholders = {}
        for col, key in zip(cols, active_models):
            with col:
                st.write(f"##### {MODEL_CATALOGUE[key]['display']}")
                placeholders[key] = st.empty()
                placeholders[key].code("", language="python")

        # get_parallel_responses is a plain sync function returning {key: async_gen}
        generators = get_parallel_responses(
            prompt,
            st.session_state.context,
            active_models,
            api_keys=api_keys,
        )

        async def stream_model(key):
            """Stream one model and update its placeholder live."""
            gen = generators[key]
            response_text = ""
            cleaned = ""   # initialise so it's always defined
            async for chunk in gen:
                response_text += chunk
                cleaned = (
                    response_text.strip()
                    .removeprefix("```python")
                    .removeprefix("```")
                    .removesuffix("```")
                    .strip()
                )
                # Update the placeholder in-place
                placeholders[key].code(cleaned, language="python")
            return cleaned

        # Run all model streams concurrently
        results = await asyncio.gather(*[stream_model(k) for k in active_models])

        final_responses = dict(zip(active_models, results))

        message = {
            "role": "assistant",
            "content": "",
            "model_responses": final_responses,
            "active_models": active_models,
        }
        st.session_state.chat_history.append(message)

        for k, text in final_responses.items():
            st.session_state.last_generated_code[k] = text


def handle_chat_input(prompt: str):
    """Sync wrapper so Streamlit can call the async handler."""
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(handle_chat_input_async(prompt))


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------
st.title("🤖 AI Code Generation Model Comparison")
st.caption("Compare Claude, OpenAI, Gemini, and OpenRouter side-by-side — powered by DeepEval")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    if message["role"] == "assistant":
        resp_dict = message.get("model_responses", {})
        active = message.get("active_models", list(resp_dict.keys()))
        if active:
            cols = st.columns(len(active))
            for col, key in zip(cols, active):
                with col:
                    st.write(f"##### {MODEL_CATALOGUE[key]['display']}")
                    st.code(resp_dict.get(key, ""), language="python")

# Chat input
if prompt := st.chat_input("What code would you like to generate?"):
    if not st.session_state.context:
        st.error("Please ingest a GitHub repository first!")
    elif not st.session_state.selected_models:
        st.error("Please select at least one model in the sidebar!")
    else:
        handle_chat_input(prompt)

# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------
active_models = st.session_state.selected_models
eval_ready = active_models and all(
    st.session_state.evaluation_results.get(k) for k in active_models
)

if eval_ready:
    st.write("---")
    st.header("📊 Evaluation Results (powered by DeepEval)")

    # Build plot data
    metrics_keys = ["correctness", "readability", "best_practices"]
    metric_labels = ["Correctness", "Readability", "Best Practices", "Overall Score"]

    plot_rows = []
    for key in active_models:
        res = st.session_state.evaluation_results[key]
        if "error" in res:
            st.error(f"Error evaluating {MODEL_CATALOGUE[key]['display']}: {res['error']}")
            
        for label, mk in zip(metric_labels[:3], metrics_keys):
            plot_rows.append({
                "Metric": label,
                "Model": MODEL_CATALOGUE[key]["display"],
                "Score": res.get("detailed_metrics", {}).get(mk, {}).get("score", 0.0)
            })
        plot_rows.append({
            "Metric": "Overall Score",
            "Model": MODEL_CATALOGUE[key]["display"],
            "Score": res.get("overall_score", 0.0)
        })

    plot_df = pd.DataFrame(plot_rows)

    # Enough colours for up to 4 models
    palette = ['#00CED1', '#FF69B4', '#7CFC00', '#FF8C00']
    colors = palette[:len(active_models)]

    fig = px.bar(
        plot_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Model Performance Comparison",
        template="plotly_dark",
        color_discrete_sequence=colors
    )
    fig.update_layout(
        xaxis_title="Evaluation Metrics",
        yaxis_title="Score",
        legend_title="Models",
        plot_bgcolor='rgba(32, 32, 32, 1)',
        paper_bgcolor='rgba(32, 32, 32, 1)',
        bargap=0.2,
        bargroupgap=0.1,
        font=dict(color='#E0E0E0'),
        title_font=dict(color='#E0E0E0'),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(32, 32, 32, 0.8)',
            bordercolor='rgba(255, 255, 255, 0.3)',
            borderwidth=1
        )
    )
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', zerolinecolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', zerolinecolor='rgba(128,128,128,0.2)')
    st.plotly_chart(fig, use_container_width=True)

    # Per-model detail tables
    for key in active_models:
        res = st.session_state.evaluation_results[key]
        st.write(f"### {MODEL_CATALOGUE[key]['display']} — Detailed Metrics")
        
        table_data = []
        for mk in metrics_keys:
            metric_data = res.get("detailed_metrics", {}).get(mk, {})
            table_data.append({
                "Metric": mk.replace("_", " ").title(),
                "Score": f"{metric_data.get('score', 0.0):.2f}",
                "Reasoning": metric_data.get("reason", "N/A - Evaluation failed")
            })
        table_data.append({
            "Metric": "Overall Score",
            "Score": f"{res.get('overall_score', 0.0):.2f}",
            "Reasoning": "Final weighted average"
        })
        df = pd.DataFrame(table_data)
        st.dataframe(
            df,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="small"),
                "Score": st.column_config.TextColumn("Score", width="small"),
                "Reasoning": st.column_config.TextColumn("Reasoning", width="large"),
            },
            hide_index=True,
            use_container_width=True
        )
