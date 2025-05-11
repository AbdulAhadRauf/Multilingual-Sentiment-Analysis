# NPPE-1 Multilingual Sentiment Analysis – Fine-Tuning LLaMA 3.1‑8B-Instruct

## 📄 Competition Link

[Multilingual Sentiment Analysis on Kaggle](https://www.kaggle.com/competitions/multi-lingual-sentiment-analysis/overview)

## 🏆 Overview

Fine-tune **LLaMA 3.1-8B-Instruct** via LoRA to classify sentiment (Positive/Negative) in text across **13 Indian languages**. All work must be performed inside Kaggle Notebooks—no external data or custom-trained models permitted.

* **Task:** Binary sentiment classification (Positive vs. Negative)
* **Model:** LLaMA 3.1-8B-Instruct + LoRA
* **Start:** February 14, 2025
* **Close:** February 18, 2025
* **Evaluation:** F1 Score (macro)

## 📂 Dataset Description

| Split | Samples | Fields                                                    |
| ----: | ------: | :-------------------------------------------------------- |
| Train |   1,000 | `ID`, `sentence`, `label` (Positive/Negative), `language` |
|  Test |     100 | `ID`, `sentence`, `language` (predict `label`)            |

Supported languages (`language` codes):

```
as, bd, bn, gu, hi, kn, ml, mr, or, pa, ta, te, ur
```

### Example (Train)

| ID | sentence                                        | label    | language |
| -: | :---------------------------------------------- | :------- | :------- |
|  1 | কর্মীদের ভাল আচরণ এবং খাবারের পাশাপাশি পানীয়…  | Positive | bn       |
|  5 | जुथानि थाखाय जायगा गैया। गुबुन मुवा सोग्रा जाय… | Negative | bd       |

## 🛠 Environment & Dependencies

* **Python 3.8+**
* **PyTorch**
* **Transformers**, **Datasets** (Hugging Face)
* **PEFT** (LoRA)
* **Tokenizers**
* **scikit-learn**
* **pandas**, **numpy**
* **tqdm**

Install via:

```bash
pip install torch transformers datasets peft scikit-learn pandas numpy tqdm
```

## 🔍 Exploratory Data Analysis (EDA)

1. **Class Balance:** count of Positive vs. Negative overall and per language
2. **Sentence Length:** distribution of token counts
3. **Language Distribution:** sample counts by language
4. **Sample Inspection:** review text complexity and script variations

## 🏗️ Model & Fine‑Tuning Approach

### 1. Base Model: LLaMA 3.1-8B-Instruct

* Pretrained instruction-tuned model supporting Hindi tokenizer extended to 13 languages.

### 2. LoRA (Low-Rank Adaptation)

* **Library:** `peft`
* **Configuration:** apply LoRA to query/key/value projections
* **Ranks & Alpha:** typically `r=8`, `alpha=16`
* **Benefits:** parameter-efficient adaptation (\~0.1% of model parameters)

### 3. Training Strategy

* **Data Formatting:** prompt–response pairs, e.g.:

  ```text
  Prompt: "Classify the sentiment (Positive/Negative): <sentence>"
  Response: "Positive"
  ```
* **Loss:** cross-entropy on token “Positive”/“Negative”
* **Optimizer:** AdamW (lr=2e-5)
* **Batch Size:** 8–16
* **Epochs:** 3–5
* **Validation Split:** 10% of training for early stopping
* **Gradient Accumulation:** if GPU memory is limited

## 🎓 Training & Validation Pipeline

1. **Load Dataset:** using Hugging Face `load_dataset('csv', ...)` for train/test CSVs.
2. **Tokenization:** `LlamaTokenizer` with padding/truncation.
3. **PEFT Setup:** wrap model with `get_peft_model` and LoRA config.
4. **Trainer API:** Hugging Face `Trainer` with compute metrics callback for F1.
5. **Checkpointing:** save best LoRA adapters by validation F1 Score.

## 📊 Evaluation

* **Primary Metric:** macro F1 Score on test set.
* Use `sklearn.metrics.f1_score(y_true, y_pred, average='macro')`.

## 🚀 Inference & Submission

1. **Load LoRA Adapters:** merge or keep separate for inference.
2. **Tokenize Test Sentences:** same preprocessing as training.
3. **Generate Predictions:** sample model’s response token and map to label.
4. **Compile CSV:** two columns `ID,label` (Positive | Negative).

```bash
python submission.py \
  --model_name_or_path llama-3.1-8b-instruct \
  --peft_adapter best_lora_adapter/ \
  --test_file data/test.csv \
  --output submission.csv
```

## 🏅 Grading Policy

Final marks are computed based on your F1 Score relative to the best-performing score:

* **If F1 ≤ 0.5:**
  `Marks = (F1 / 0.5) × 50%`
* **If F1 > 0.5:**
  `Marks = 50% + ((F1 – 0.5) / (Max_F1 – 0.5)) × 50%`

## 🏃 How to Reproduce

1. Clone this repo to Kaggle.
2. Install dependencies in notebook.
3. Place `train.csv` & `test.csv` under `data/`.
4. Run EDA notebook: `notebooks/eda_multilingual_sa.ipynb`.
5. Fine-tune LoRA adapters: `notebooks/train_lora.ipynb`.
6. Run `submission.py` to generate `submission.csv`.

---

**Push the boundaries of multilingual AI!**
