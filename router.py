import streamlit as st

def require_role(allowed_roles):
    role = st.session_state.get("role")
    if role not in allowed_roles:
        st.error("🚫 Unauthorized")
        st.stop()


def get_page():
    params = st.query_params
    return params.get("page", "chat")


def set_page(page_name):
    st.query_params["page"] = page_name