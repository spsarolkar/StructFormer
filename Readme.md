# 🔀 StructFormer: A Transformer for Structured Data Transformation

StructFormer is a general-purpose Transformer model designed to transform structured input data (such as validation errors, business rule triggers, or CSV lines) into structured output actions (such as SQL statements, corrected records, or data mappings).

It achieves high accuracy (validated up to ~96%) on tasks like error-to-adjustment translation, multi-record transformations, and rule-based logic generation.

---

## 🔍 Use Cases

- ✅ Translating validation error logs into SQL adjustments
- ✅ Automating rule-based record updates or corrections
- ✅ Mapping flat CSV data to normalized multi-table outputs
- ✅ Generating templated JSON or configuration snippets
- ✅ Producing control files or actions from input triggers

---

## 📊 Example

**Input:**
```
ContID=CONT1001 ErrorType=Invalid Currency
```

**Output:**
```sql
UPDATE Trades SET Currency='EUR' WHERE TradeID=91029;
```

---

## 🚀 Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

Check mac os non x86 platform
```bash
python -c "import platform; print(platform.platform())"
```

Follow below instructions if above command returns x86

```bash
brew install --cask miniforge
conda create -n tf-arm python=3.10
conda activate tf-arm
python3 -c "import platform; print(platform.platform())"
# source ~/miniforge3/bin/activate
```

### 2. Prepare your training data
Ensure a CSV file with two columns:
- `input_text`  → validation error or structured trigger
- `output_text` → target SQL or structured action

Sample:
```csv
input_text,output_text
"TradeID=93297 AccountID=ACC1003 ErrorType=Missing Quantity","UPDATE Trades SET Quantity=46 WHERE TradeID=93297; INSERT INTO AdjustmentLog(ErrorID, AdjustedBy) VALUES('ERR3008', 'User1');"
```

### 3. Train the model
```bash
python train.py --data data/train.csv --output_dir model/ --vocab_size 4000
```

### 4. Run inference
```bash
python inference.py --model_dir model/ --text "ContID=CONT1002 ErrorType=Missing Amount"
```

---

## 💡 Tech Highlights
- Custom Transformer Encoder-Decoder (Keras 3)
- SentencePiece tokenizer for subword handling
- Dynamic padding and masking
- Supports resuming training, checkpointing

---

## 📈 Results
Achieved ~96% validation accuracy on internal structured-to-SQL adjustment task (see `notebooks/AdjustmentWithTransformer_Sentensepiece_TF.ipynb`).

---

## 💼 License
MIT License

