from main import get_query_chain

import streamlit as st

st.title('T Shirt Database QnA')

question = st.text_input('Question: ')
if question:
    chain = get_query_chain()
    ans = chain.run(question)
    st.header('Answer')
    st.write(ans)