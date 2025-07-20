# 💼 JD-Employee Matching with Embeddings + OpenAI LLM

This project matches employee profiles to job descriptions (JDs) using a hybrid approach:

* **Phase 1**: Semantic search via **FASIS embeddings**
* **Phase 2**: Contextual reasoning via **OpenAI GPT (LLM)**

---

## 🧠 How It Works

1. **Extract Data**
   Load job descriptions and employee data from Excel files.

2. **Semantic Embedding Matching**
   Generate vector embeddings (using `sentence-transformers`) for JDs and employee profiles. Compute cosine similarity to shortlist top N candidates for each job.

3. **LLM-based Refinement**
   Use `gpt-3.5-turbo` or `gpt-4o` to contextually match JDs to employees based on:

   * Skills
   * Experience
   * Domain relevance
   * Project descriptions

4. **Final Output**
   Excel sheet showing top-matched employees per JD with reasoning and score.

---

## 📂 Project Structure

```
.
├── data/
│   ├── jds.xlsx               # JD Excel file with headings like JobID, Title, RequiredSkills...
│   └── employees.xlsx         # Employee Excel file with skills, domains, project desc...
│
├── main.py     # Main pipeline script
├── requirements.txt
└── README.md
```

---

## 📄 Input Format

### Job Descriptions (`jds.xlsx`)

| JobID | Title       | RequiredSkills                                   | Experience | Description                                                |
| ----- | ----------- | ------------------------------------------------ | ---------- | ---------------------------------------------------------- |
| 1001  | ML Engineer | Machine Learning, Angular, Microservices, Django | 9          | Looking for a strong ML Engineer with hands-on experience. |

### Employees (`employees.xlsx`)

| Employee Id | Skills                                      | Domains        | Experience                       | Job Title         | Current Project Description                                          | Experience | PeerRatings |
| ----------- | ------------------------------------------- | -------------- | -------------------------------- | ----------------- | -------------------------------------------------------------------- | ---------- | ----------- |
| Employee 1  | Node.js, Java, GCP, Machine Learning, Spark | E-Commerce, AI | Node.js:5, Java:4, ML:2, Spark:1 | Backend Developer | Developed scalable APIs on GCP integrating ML recommendation systems | 3.14       | 2           |

---

## 🚀 Running the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Requires Python 3.8+

### 2. Set OpenAI Key

```bash
export OPENAI_API_KEY=your-api-key-here
```

### 3. Run the Pipeline

```bash
python fasis_embedding_llm.py
```

---

## ⚙️ Configuration

You can customize:

* Number of top candidates to consider via embeddings
* Model used for LLM matching (`gpt-3.5-turbo` recommended for cost)
* Output formats and verbosity

---

## ✅ Output

The result is a ranked list of employees per job, saved as:

```
output/matches.xlsx
```

Each entry includes:

* Job ID
* Matched Employee ID
* LLM Matching Score
* Reason for match

---

## 💸 Costs

* 💡 **Use `gpt-3.5-turbo`** for cheapest inference:
  \~\$0.002 per request if prompts are <1K tokens.

* Optimize by:

  * Limiting LLM to top 5 matches
  * Keeping prompt compact

---

## 📌 Dependencies

```text
openai>=1.0.0
pandas
numpy
sentence-transformers
tqdm
tenacity
scikit-learn
```

---

## 📈 Future Enhancements

* Add streamlit UI for upload + visualization
* Add feedback loop for match quality
* Support multilingual matching
