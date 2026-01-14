import streamlit as st
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt



@st.cache_resource
def load_assets():

    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return tfidf, model


tfidf, model = load_assets()


st.set_page_config(page_title="Shield AI", page_icon="🛡️")
st.title("🛡️ Shield AI: Spam Guard")
st.write("Professional SMS & Email Classification")

message = st.text_area("Enter message to analyze:", placeholder="Type or paste text here...", height=150)

if st.button("Analyze Message", use_container_width=True):

    if not message.strip():
        st.warning("Please enter a message to analyze.")
    else:

        vectorized_input = tfidf.transform([message])
        prediction = model.predict(vectorized_input)[0]
        probabilities = model.predict_proba(vectorized_input)[0]


        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("### 🚨 Result: SPAM")
            else:
                st.success("### ✅ Result: HAM")
                st.balloons()

        with col2:
            conf = probabilities[1] if prediction == 1 else probabilities[0]
            st.metric("AI Confidence", f"{conf * 100:.1f}%")

        if prediction == 1:
            st.write("---")
            st.subheader("Spam Keywords Detected:")
            wc = WordCloud(background_color='white', colormap='Reds', width=800, height=400).generate(message)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)