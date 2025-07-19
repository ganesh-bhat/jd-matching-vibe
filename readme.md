# 🔍 GenAI-Powered Smart JD-to-Candidate Matcher

This project demonstrates a **Generative AI-based approach** to intelligently match **Job Descriptions (JDs)** with **candidate skills** using OpenAI's LLMs — avoiding traditional fuzzy or keyword search. The entire solution is built using **low-code tools and vibe coding** style for rapid experimentation and simplicity.

## ✨ Key Features

- ✅ LLM-based intelligent matching for **complex, real-world JDs**
- ✅ Filters large candidate datasets automatically to avoid context length issues
- ✅ Supports **batch processing** using smart pre-filtering (e.g. cosine similarity)
- ✅ Outputs results into Excel for easy analysis
- ✅ Runs with **low-code tools** like Power Automate and Excel integrations
- ✅ Python backend using OpenAI API for deeper control (optional)


## 📁 Project Structure

```

📦 smart-jd-matcher/
┣ 📜 data\bench_employee           ← Sample dataset of candidates (skills, summary)
┣ 📜 data\job_requirements         ← Sample dataset of 50+ job descriptions
┣ 📜 matched_jobs                  ← Output of matched candidates
┣ 📜 requirements.txt              ← Python dependencies
┣ 📜 main.py                       ← Python script to batch process and match
┣ 📜 README.md                     ← You're here!

````

## ⚙️ How It Works

1. **Candidate and JD data** are stored in Excel sheets.
2. Python script filters and selects only relevant candidates per JD (using embeddings or similarity).
3. A prompt is constructed and sent to the OpenAI API.
4. LLM returns the match quality or reasoning.
5. Results are saved back to Excel.

## 🛠 Tools Used
- 🧠 [OpenAI GPT-4](https://openai.com/)
- 📊 Microsoft Excel
- 🐍 Python (optional)
- 🧪 Cosine Similarity via `scikit-learn` or `sentence-transformers` (optional pre-filter)

## 🚀 How to Run (Python Version)

1. Install dependencies:

```bash
pip install -r requirements.txt
````
2. Add your OpenAI API key as an environment variable or in the script.
3. Run the script:

```bash
python match_jds_with_candidates.py
```

4. Output will be saved in `matched_output.xlsx`.

## 📈 Sample Use Cases

* Internal resume screening in companies
* Smart filtering before sending resumes to hiring managers
* JD matching for gig platforms or talent marketplaces
* Research into LLM-based HR automation
