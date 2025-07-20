import pandas as pd
import openai
import faiss
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential

# 🔑 Set your API key
api_key = ""  # 🔑 Replace with your actual OpenAI key or set in environment variable
OPENAI_API_KEY = api_key or os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)
openai.api_key = OPENAI_API_KEY
MODEL_NAME = "gpt-3.5-turbo"
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 10
TOP_FINAL_MATCHES = 3

bench_excel = "sample_employees.xlsx"          # Your bench data sheet
jd_excel = "sample_job_requirements.xlsx"       # Your job description sheet
output_excel = "MatchedResults.xlsx"    # Output file name

# ========== 1. Load Excel Data ==========
def load_data(jd_file: str, employee_file: str):
    jd_df = pd.read_excel(jd_file)
    emp_df = pd.read_excel(employee_file)
    return jd_df, emp_df

# ========== 2. Preprocessing ==========
def prepare_candidate_text(row):
    return (
        f"Skills: {row['Skills']}\n"
        f"Domains: {row['Domains']}\n"
        f"Experience Breakdown: {row['Experience']}\n"
        f"Title: {row['Job Title']}\n"
        f"Project Description: {row['Current Project Description']}\n"
        f"Years of Experience: {row['Experience.1']}\n"
        f"Peer Ratings: {row['PeerRatings']}"
    )

def prepare_jd_text(row):
    return (
        f"Title: {row['Job Title']}\n"
        f"Required Skills: {row['RequiredSkills']}\n"
        f"Experience: {row['Experience In Domains']}\n"
        f"Description: {row['Description']}"
    )

# ========== 3. Batch Embedding ==========

def get_embeddings(texts, model="text-embedding-3-small", batch_size=100):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches"):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings).astype("float32")

# ========== 4. FAISS Setup ==========
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def find_top_k(index, query_embedding, k=10):
    D, I = index.search(query_embedding, k)
    return I[0]

# ========== 5. LLM JD-to-Candidate Matching ==========
@retry(wait=wait_random_exponential(min=1, max=TOP_FINAL_MATCHES), stop=stop_after_attempt(5))
def ask_llm_to_match(jd_text, employee_profiles):
    prompt = f"""
You are a talent matcher AI. A Job Description is given below. 
Then 10 employee profiles are provided. Match them by ranking from best to least fit based on skills, experience, domain, and relevance.
Only return top {TOP_FINAL_MATCHES} ranked list with Employee IDs and reasons for ranking.

Job Description:
{jd_text}

Candidate Profiles:
{employee_profiles}

Output format:
1. Employee ID - reason
2. Employee ID - reason
...
"""
    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def ask_llm_to_match_old(jd_text, candidate_texts):
    messages = [
        {
            "role": "system",
            "content": "You are a job recommender system. You will rank candidates from best to least suitable based on job descriptions."
        },
        {
            "role": "user",
            "content": f"Job Description:\n{jd_text}\n\nHere are candidate profiles:\n\n"
                       + "\n\n".join([f"Candidate {i + 1}:\n{c}" for i, c in enumerate(candidate_texts)])
        }
    ]


    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": messages}
        ],
        temperature=0.3,
        max_tokens=800
    )
    return response.choices[0].message.content

# ========= 6. Main ==========
def main(jd_file: str, employee_file: str, top_k=TOP_K):
    jd_df, emp_df = load_data(jd_file, employee_file)

    emp_texts = emp_df.apply(prepare_candidate_text, axis=1).tolist()
    emp_ids = emp_df['Employee Id'].tolist()

    print("🔍 Generating candidate embeddings...")
    emp_embeddings = get_embeddings(emp_texts)
    faiss_index = build_faiss_index(emp_embeddings)

    results = []

    for i, jd_row in jd_df.iterrows():
        print(f"\n📄 Matching JD #{jd_row['JobID']} - {jd_row['Job Title']}")
        jd_text = prepare_jd_text(jd_row)
        jd_embedding = get_embeddings([jd_text])
        top_k_indices = find_top_k(faiss_index, jd_embedding, k=top_k)

        top_candidates = [emp_texts[idx] for idx in top_k_indices]
        top_candidate_ids = [emp_ids[idx] for idx in top_k_indices]
        combined_profiles = "\n\n".join(top_candidates)

        ranked_output = ask_llm_to_match(jd_text, combined_profiles)
        print(f"🏆 Top {TOP_FINAL_MATCHES} Matches:\n{ranked_output}")
        print("=" * 80)

        results.append({
            "JobID": jd_row["JobID"],
            "Job Title": jd_row["Job Title"],
            "LLM Top Matches": ranked_output
        })

    # Save to Excel
    result_df = pd.DataFrame(results)
    result_df.to_excel(output_excel, index=False)
    print(f"\n✅ Matching complete. Results saved to {output_excel}")

# Run
if __name__ == "__main__":
    main(jd_excel, bench_excel, top_k=TOP_K)
