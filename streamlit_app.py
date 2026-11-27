"""Entry point for Streamlit Community Cloud: main file = streamlit_app.py. Runs v2/app_streamlit.py."""
import runpy, os
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2", "app_streamlit.py"), run_name="__main__")
