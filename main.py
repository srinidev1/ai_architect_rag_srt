import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from router import get_page, set_page
from views import ingestview,chatview,evaluationview
#from views import ingestview, chatview, evaluationview

#from router import set_page

def loginForm():
    # Load the YAML file
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    # Extract the credentials and cookie settings
    credentials = config['credentials']
    cookie = config['cookie']

    # Create an authenticator object
    authenticator = stauth.Authenticate(
        credentials,
        cookie['name'],
        cookie['key'],
        cookie['expiry_days']
    )

    if "role" not in st.session_state:
        st.session_state.role = None

    col1, col2, col3 = st.columns([6, 1, 1])

    # Render the login widget
    #name, authentication_status, username = 
    authenticator.login('main')
    menu = []
    authentication_status = st.session_state.get('authentication_status')
    username = st.session_state.get('username')
    if authentication_status is not None:
        if authentication_status:
            st.write(f'Welcome *{st.session_state["name"]}*')
            st.session_state.role = config["credentials"]["usernames"][username]["role"]
            if st.session_state["role"] == "admin":
                #st.write("You have admin privileges.")
                menu = ["Ingest", "Chat", "Evaluation"]
                st.sidebar.markdown("### 👑 Admin Menu")
                if st.sidebar.button("📥 Ingest"):
                    set_page("ingest")
                if st.sidebar.button("💬 Chat"):
                    set_page("chat")
                if st.sidebar.button("📊 Evaluation"):
                    set_page("evaluation")
            else:
                #st.write("You have user privileges.")
                menu = ["Chat"]
            #choice = st.sidebar.selectbox("Navigation", menu)    
            #choice = st.sidebar.radio("Go to", menu)
            #with col3:
            authenticator.logout('🚪 Logout', 'sidebar')
            #authenticator.logout('Logout', 'main')
        else:
            st.error('Username/password is incorrect')
    else:
        st.warning('Please enter your username and password')
        
    #print(f"Authentication status: {authentication_status}")


    page = get_page()
    #print(f"Current page: {page}")
    if page == "ingest":
       ingestview.render()
    elif page == "chat":
        chatview.render()   
    elif page == "evaluation":
        evaluationview.render()   
    elif page == "logout":
        authenticator.logout('main')
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
    

    #if authentication_status:
     #   st.write(f'Welcome *{name}*')
      #  st.title('Some content')
       # authenticator.logout('Logout', 'main')
    #elif authentication_status == False:
     #   st.error('Username/password is incorrect')
    #elif authentication_status == None:
     #   st.warning('Please enter your username and password')
    

if __name__ == "__main__":
    #import sys
    #print(f"Python executable: {sys.executable}")
    loginForm()
