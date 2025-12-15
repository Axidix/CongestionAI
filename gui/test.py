import streamlit as st
import requests

url = st.secrets["BACKEND_API_URL"] + "/forecast"
key = st.secrets["BACKEND_API_KEY"]
headers = {"X-API-Key": key}
try:
    r = requests.get(url, headers=headers, timeout=30)
    st.write("Status:", r.status_code)
    st.write("Headers:", r.headers)
    st.write("First 500 chars:", r.text[:500])
except Exception as e:
    st.error(str(e))