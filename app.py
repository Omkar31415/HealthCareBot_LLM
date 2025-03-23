import streamlit as st
from physical_health import run_physical_health
from mental_health import run_mental_health

# Set up page config for a consistent header and favicon
st.set_page_config(
    page_title="AI Health Diagnostic Assistant",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="auto",
)

# Add some custom CSS for styling
st.markdown(
    """
    <style>
    /* Center the title and adjust margins */
    .main > div:first-child {
        text-align: center;
    }
    /* Custom button style (uses st.markdown to override) */
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        margin: 10px;
        transition: background-color 0.3s;
    }
    div.stButton > button:hover {
        background-color: #45a049;
    }
    /* Page background style */
    .reportview-container {
        background: #f9f9f9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize navigation state
if 'page' not in st.session_state:
    st.session_state.page = "main"

# Main page UI
if st.session_state.page == "main":
    st.title("AI Health Diagnostic Assistant 🤖")
    st.markdown("### Your personalized health companion")
    st.markdown("#### Welcome! Please select a diagnostic tool below:")
    
def main():
    # Main menu page
    if st.session_state.page == "main":
        # Two-column layout for buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Mental Health Assessment"):
                st.session_state.page = "mental"
                st.rerun()
        with col2:
            if st.button("Physical Health Assessment"):
                st.session_state.page = "physical"
                st.rerun()

    # Mental Health interface
    elif st.session_state.page == "mental":
        run_mental_health()

    # Physical Health interface
    elif st.session_state.page == "physical":
        run_physical_health()

if __name__ == "__main__":
    main()
