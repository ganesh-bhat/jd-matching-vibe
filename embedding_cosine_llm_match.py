import os
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Set your API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # or directly: openai.api_key = "sk-..."

api_key = ""  # 🔑 Replace with your actual key
bench_excel = "/Users/ganeshbhat/PycharmProjects/employee-job-matching/data/bench_employee/sample_employees.xlsx"          # Your bench data sheet
jd_excel = "/Users/ganeshbhat/PycharmProjects/employee-job-matching/data/job_requirements/sample_job_requirements.xlsx"       # Your job description sheet


# Constants
CANDIDATE_FILE = "/Users/ganeshbhat/PycharmProjects/employee-job-matching/data/bench_employee/candidates.xlsx"
JD_FILE = "/Users/ganeshbhat/PycharmProjects/employee-job-matching/data/job_requirements/job_descriptions.xlsx"
OUTPUT_FILE = "/Users/ganeshbhat/PycharmProjects/employee-job-matching/data/matched_jobs/MatchedResults.xlsx"    # Output file name
TOP_N = 10
EMBED_MODEL = "text-embedding-ada-002"

def read_data():
    candidates_df = pd.read_excel(CANDIDATE_FILE)
    jds_df = pd.read_excel(JD_FILE)
    return candidates_df["Profile"].tolist(), jds_df["JD"].tolist()

def get_embeddings(texts, model=EMBED_MODEL, batch_size=100):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        try:
            response = openai.embeddings.create(model=model, input=batch)
            batch_embeddings = [r.embedding for r in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Embedding error at batch {i}: {e}")
            embeddings.extend([[0]*1536]*len(batch))  # fallback
    return np.array(embeddings)

def rank_candidates(jd_embeddings, candidate_embeddings):
    sim_matrix = cosine_similarity(jd_embeddings, candidate_embeddings)
    top_matches = np.argsort(-sim_matrix, axis=1)[:, :TOP_N]
    return top_matches, sim_matrix

def call_llm(jd, profiles, model="gpt-4"):
    profile_block = "\n\n".join([f"Candidate {i+1}:\n{p}" for i, p in enumerate(profiles)])
    prompt = f"""
You are a hiring assistant.

Job Description:
{jd}

10 Candidate Profiles:
{profile_block}

For each candidate:
- Assess skill fit
- Note major gaps
- Match score (out of 10)
- Short reasoning

Give a ranked list with match score and comment per candidate.
"""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"

def main():
    print("📄 Reading input files...")
    candidate_profiles, jds = read_data()

    print("🔍 Creating embeddings...")
    candidate_embs = get_embeddings(candidate_profiles)
    jd_embs = get_embeddings(jds)

    print("🤝 Matching candidates to JDs...")
    top_matches, sim_matrix = rank_candidates(jd_embs, candidate_embs)

    results = []
    for jd_idx, jd_text in enumerate(tqdm(jds, desc="LLM Matching")):
        matched_indices = top_matches[jd_idx]
        matched_profiles = [candidate_profiles[i] for i in matched_indices]
        match_output = call_llm(jd_text, matched_profiles)
        results.append({
            "JD_Index": jd_idx,
            "Job_Description": jd_text,
            "Top_Candidate_Indices": matched_indices.tolist(),
            "LLM_Response": match_output
        })

    pd.DataFrame(results).to_excel(OUTPUT_FILE, index=False)
    print("✅ Done! Results saved to final_matching_results.xlsx")

if __name__ == "__main__":
    main()
