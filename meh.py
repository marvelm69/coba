
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import streamlit as st

# Fungsi untuk memproses teks
def preprocess_text_simple(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Membaca dataset kursus
df = pd.read_csv('/kaggle/input/linkedin-job-postings/postings.csv')

# Menghapus duplikat berdasarkan 'job_posting_url'
df = df.drop_duplicates(subset=['job_posting_url'])



# Menggabungkan teks dari kolom 'company_name', 'title', dan 'description'
df['Combined'] = df['company_name'] + ' ' + df['title'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['skills_desc'].fillna('')
df['Combined'] = df['Combined'].apply(preprocess_text_simple)

# Vektorisasi teks gabungan menggunakan TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Combined'])
df.fillna("Unknown", inplace=True)
# Fungsi untuk merekomendasikan kursus berdasarkan input pengguna tanpa clustering
def recommend_course_non_clustering(user_input, df, vectorizer, tfidf_matrix, experience_levels, work_types, company_name):
    # Filter dataset berdasarkan pilihan pengguna
    filtered_df = df.copy()
    if experience_levels:
        filtered_df = filtered_df[filtered_df['formatted_experience_level'].isin(experience_levels)]
    if work_types:
        filtered_df = filtered_df[filtered_df['formatted_work_type'].isin(work_types)]
    if company_name:
        filtered_df = filtered_df[filtered_df['company_name'].str.contains(company_name, case=False, na=False)]
    
    # Preprocess input pengguna
    user_input_processed = preprocess_text_simple(user_input)
    user_tfidf = vectorizer.transform([user_input_processed])
    
    # Menghitung kemiripan kosinus di seluruh dataset
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix[filtered_df.index]).flatten()
    
    # Mendapatkan indeks kursus teratas
    top_course_indices = cosine_similarities.argsort()[-50:][::-1]  # Mengambil 50 kursus teratas
    
    # Membuat DataFrame dengan penomoran
    top_courses = filtered_df.iloc[top_course_indices].copy()
    top_courses.reset_index(drop=True, inplace=True)
    top_courses.index = top_courses.index + 1  # Menomori dari 1 hingga 50
    
    # Menambahkan kolom kemiripan kosinus
    top_courses['cosine_similarity'] = cosine_similarities[top_course_indices]
    
    return top_courses


st.set_page_config(page_title="Job Recommendation System", page_icon="📚")
st.title("📚 Job Recommendation System")

# Membuat area filter
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .st-expander {
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
    }
    .st-expander p {
        margin: 0;
    }
    .search-box {
        width: 100%;
        padding: 5px;
        margin-top: 10px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

st.subheader("Filters")

# Experience Level filter
st.write("**Experience Level**")
experience_levels = df['formatted_experience_level'].unique().tolist()
selected_experience_levels = []
cols = st.columns(2)  # Membuat dua kolom
for i, exp in enumerate(experience_levels):
    if i % 2 == 0:
        with cols[0]:
            if st.checkbox(exp, key=f"exp_{exp}"):
                selected_experience_levels.append(exp)
    else:
        with cols[1]:
            if st.checkbox(exp, key=f"exp_{exp}"):
                selected_experience_levels.append(exp)

# Work Type filter
st.write("**Work Type**")
work_types = df['formatted_work_type'].unique().tolist()
selected_work_types = []
cols = st.columns(2)  # Membuat dua kolom
for i, work in enumerate(work_types):
    if i % 2 == 0:
        with cols[0]:
            if st.checkbox(work, key=f"work_{work}"):
                selected_work_types.append(work)
    else:
        with cols[1]:
            if st.checkbox(work, key=f"work_{work}"):
                selected_work_types.append(work)

company_name = st.text_input("🔍 Search Company Name")

user_input = st.text_input("🔍 Enter your job interest (e.g., data scientist, machine learning engineer)")

if st.button("Get Recommendations"):
    recommendations = recommend_course_non_clustering(user_input, df, vectorizer, tfidf_matrix, selected_experience_levels, selected_work_types, company_name)
    st.session_state.recommendations = recommendations
    st.session_state.page = 0

if 'recommendations' in st.session_state:
    recommendations = st.session_state.recommendations
    page = st.session_state.page
    start_index = page * 5
    end_index = start_index + 5
    st.write("### 🎯 Recommendations:")
    for i, row in recommendations.iloc[start_index:end_index].iterrows():
        st.markdown(f"#### **{start_index + i + 1}. {row['title']}**")
        st.markdown(f"**🏢 Company Name**: {row['company_name']}")
        st.markdown(f"**📍 Location**: {row['location']}")
        st.markdown(f"[🔗 View Job Posting]({row['job_posting_url']})")
        with st.expander("📄 More Info"):
            st.markdown(f"**📝 Description**: {row['description']}")
            st.markdown(f"**💰 Min Salary**: {row['min_salary']}")
            st.markdown(f"**💵 Max Salary**: {row['max_salary']}")
            st.markdown(f"**🕒 Work Type**: {row['formatted_work_type']}")
            st.markdown(f"**🎓 Experience Level**: {row['formatted_experience_level']}")        
        st.markdown("---")

    cols = st.columns(2)
    if start_index > 0:
        if cols[0].button("Previous"):
            st.session_state.page -= 1
    if end_index < len(recommendations):
        if cols[1].button("Next"):
            st.session_state.page += 1
