# mini-tool-lm-chat
Streamlit chat for the from-scratch mini-tool-lm v2 model (13.8M params) with Composio tools. Code only: the weights are
downloaded at startup from the private HF model repo `davolu/mini-tool-lm-v2` (needs `HF_TOKEN`), and the tool retriever
(potion-base-8M, public) from the Hugging Face Hub.
Main file: `streamlit_app.py`. Python 3.12 recommended. Secrets (names only): see `.streamlit/secrets.toml.example`.
Risky tools (send/post/create/update/delete) always wait for an Approve/Reject click.
