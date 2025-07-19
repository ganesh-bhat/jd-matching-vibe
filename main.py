import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# === CONFIGURATION ===
api_key = ""  # 🔑 Replace with your actual key
bench_excel = "/Users/ganeshbhat/PycharmProjects/employee-job-matching/data/bench_employee/sample_employees.xlsx"          # Your bench data sheet
jd_excel = "/Users/ganeshbhat/PycharmProjects/employee-job-matching/data/job_requirements/sample_job_requirements.xlsx"       # Your job description sheet
output_excel = "/Users/ganeshbhat/PycharmProjects/employee-job-matching/data/matched_jobs/MatchedResults.xlsx"    # Output file name

# === LOAD DATA ===
bench_df = pd.read_excel(bench_excel)
jd_df = pd.read_excel(jd_excel)

# === FILTER TOP N MATCHES PER JD ===
def filter_top_n_candidates(jd_row, employee_df, top_n=20):
    jd_text = f"{jd_row['RequiredSkills']} {jd_row['Job Title']} {jd_row['Domains']} {jd_row['Experience In Domains']} {jd_row['Description']} "
    candidate_texts = employee_df['Skills'].fillna('') + " " + \
                      employee_df['Domains'].fillna('') + " " + \
                      employee_df['Experience'].fillna('') + " " + \
                      employee_df['Job Title'].fillna('') + " " + \
                      employee_df['Current Project Description'].fillna('') + " " + \
                      employee_df['Experience'].fillna('')

    tfidf = TfidfVectorizer().fit_transform([jd_text] + candidate_texts.tolist())
    cosine_scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    top_indices = cosine_scores.argsort()[-top_n:][::-1]
    return employee_df.iloc[top_indices]

# === MATCH USING GPT ===
def match_employee_to_jd(jd, filtered_employees):
    prompt = f"""You are an expert in talent matching. Match the following job description to the best employee(s) from this list. Explain your choices.

Job Description:
{jd['Description']}
Required Skills: {jd['RequiredSkills']}
Job Title {jd['Job Title']}
Experience: {jd['Experience In Domains']} years
Domains: {jd['Domains']}

Candidates:
"""
    for _, emp in filtered_employees.iterrows():
        prompt += f"""
- Employee Id: {emp['Employee Id']}
  Domain: {emp['Domains']}
  Skills: {emp['Skills']}
  Experience: {emp['Experience']} years
  Job Title: {emp['Job Title']}
  Peer Ratings: {emp['PeerRatings']}
  Current Project Description: {emp['Current Project Description']}
"""

    prompt += "\nGive top 1-2 matches with explanation."

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

# === PROCESS ALL JOBS ===
results = []

count = 0
for _, jd in jd_df.iterrows():
    filtered = filter_top_n_candidates(jd, bench_df, top_n=5)
    match = match_employee_to_jd(jd, filtered)
    results.append({
        "JobID": jd["JobID"],
        "MatchedCandidates": match
    })
    count = count+1
    if count == 3:
        break;

# === EXPORT TO EXCEL ===
output_df = pd.DataFrame(results)
output_df.to_excel(output_excel, index=False)
print(f"Matching complete. Results saved to {output_excel}")
