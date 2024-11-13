import streamlit as st
import pickle

model=pickle.load(open('spam.pkl','rb'))
cv=pickle.load(open('vectorizer.pkl','rb'))

st.title("Email Spam Classification Application")
st.write("This is Machine learning Application to classify emailsa as spam or ham")

user_input=st.text_area("Enter Email user_input: ",height=200)

if st.button("Cassify"):
    if user_input:
        data=[user_input]
        vect=cv.transform(data).toarray()
        pred=model.predict(vect)
        
        if pred[0]==0:
            st.success("This Email is not Spam")
        else:
            st.error("This is Spam Email")

    else:
        st.error("please write Email")

    
# streamlit run  app.py
