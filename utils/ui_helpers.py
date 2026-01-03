import json
import time
from contextlib import contextmanager
from pathlib import Path
import streamlit as st


def show_toast(message: str, kind: str = "info") -> None:
    """Show a toast or fallback to standard alerts."""
    try:
        st.toast(message, icon=None)
        return
    except Exception:
        try:
            if kind == "success":
                st.success(message)
            elif kind == "error":
                st.error(message)
            elif kind == "warning":
                st.warning(message)
            else:
                st.info(message)
        except Exception:
            # Last resort: print to console if UI methods fail
            print(f"[{kind.upper()}] {message}")


@contextmanager
def timed_spinner(label: str = "Working..."):
    """Context manager that measures elapsed time while showing a spinner."""
    start = time.perf_counter()
    with st.spinner(label):
        yield lambda: time.perf_counter() - start


def run_task(label: str, func, *args, success_message: str | None = None, error_message: str | None = None, toast: bool = True, **kwargs):
    """Execute a callable with spinner, timing, and optional toast notifications.

    Returns a tuple: (result, elapsed_seconds, error)
    """
    start = time.perf_counter()
    try:
        with st.spinner(label):
            result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if toast:
            show_toast(success_message or f"Completed in {elapsed:.2f}s", kind="success")
        st.caption(f"Completed in {elapsed:.2f}s")
        return result, elapsed, None
    except (ValueError, TypeError, RuntimeError, KeyError, AttributeError, ZeroDivisionError) as exc:
        elapsed = time.perf_counter() - start
        if toast:
            show_toast(error_message or "Operation failed", kind="error")
        st.error(error_message or f"Error: {exc}")
        return None, elapsed, exc


def copy_button(text: str, key: str, label: str = "Copy result"):
    """Render a small HTML button that copies text to clipboard."""
    safe_text = json.dumps(text)
    safe_key = ''.join(c if c.isalnum() else '_' for c in key)
    btn_html = f"""
    <button id="copy-{key}" style="padding: 0.25rem 0.75rem; border-radius: 6px; border: 1px solid #3d4148; background: #1E88E5; color: white; cursor: pointer;">
        {label}
    </button>
    <script>
        const btn{safe_key} = document.getElementById('copy-{key}');
        if (btn{safe_key}) {{
            btn{safe_key}.onclick = () => navigator.clipboard.writeText({safe_text});
        }}
    </script>
    """
    st.markdown(btn_html, unsafe_allow_html=True)

def load_css(css_file: str = "styles.css") -> None:
    """Load CSS from file and apply to Streamlit app.
    
    Args:
        css_file: Path to CSS file relative to the app directory
    """
    try:
        css_path = Path(__file__).parent.parent / css_file
        if css_path.exists():
            with open(css_path, 'r') as f:
                css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        else:
            st.warning(f"CSS file not found: {css_path}")
    except Exception as e:
        st.warning(f"Error loading CSS: {str(e)}")