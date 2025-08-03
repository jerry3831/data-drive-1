import streamlit as st
import re
import requests

# !demo purposes
WEBHOOK_URL = r"https://www.google.com"

# simple email validation
def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email) is not None

# simple phone number validation
def is_valid_phone(phone):
    pattern = r"^(08|09|01)\d{8}$"
    return re.match(pattern, phone) is not None

# ?separate into two columns
colA, colB = st.columns(2)
with colA:
    st.header("About Developers")
    st.markdown("""The application was devloped by the foolwing level three data science students from mzuni.""")
    st.markdown("""
                - Jeremia Nkosi
                - Queen Sosola
                - Elizabeth Mfune
                - Maxwel Mwala
                - Tamandani Yona
                """)

with colB:
    # TODO: include all input fields in a form to avoid unecessary reruns
    st.header("Contact us")
    with st.form("Contact_form"):
        name = st.text_input("Name")
        phone = st.text_input("Phone Number")
        email = st.text_input("Email Adress")
        comment = st.text_area("Comment or Feedback")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if not WEBHOOK_URL:
                st.error("Email service is not set up. Please try again later")
                st.stop()
            if not name:
                st.error("Please provide your name.")
                st.stop()
            if not is_valid_email:
                st.error("Please provide a valid email in the format **johndoe12@gmail.com**")
                st.stop()
            if not is_valid_phone:
                st.error("Please provide a valid phone number.")
                st.stop()
            if not comment:
                st.error("The comment / Feedback area cannot be blank.")
                st.stop()

            # TODO: preparing the data payload for the webhook
            data = {
                "name": name, 
                "email": email,
                "phone": phone,
                "comment": comment
            }
            response = requests.post(WEBHOOK_URL, json=data)

            if response.status_code == 200:
                st.success("Your comment / Feedback has been submitted successifully")
            else:
                st.error("There was an error while sending your message.")


