
## 📋 TO DO

### Step 1: Initial Setup (One-time only)
1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

2. **Activate the virtual environment**

     ```bash
     .venv\Scripts\Activate.ps1
     ```


### Step 2: Create Your Own Branch
Replace `{your-name}` with your actual name (no spaces, use hyphens)

```bash
git branch your-name
git checkout your-name
```



### Step 3: Run the Script
Choose ONE CSV file from the following:
- `01.csv`
- `02.csv`
- `03.csv`

Run the script with your chosen file:
```bash
python .\desc_generator.py your-chosen-file.csv
```
**Example:** 
`python .\desc_generator.py 01.csv`
`python .\desc_generator.py 02.csv`
`python .\desc_generator.py 03.csv`

### Step 4: Save and Upload Your Work
Run these commands one by one:

```bash
git add .
git commit -m "added data"
git push origin your-name
```

---

## 📝 Task Assignments

| Team Member | Assigned CSV File | Branch Name |
|------------|------------------|-------------|
| DAYA   | 01.csv          | member1     |
| SANKET   | 02.csv          | member2     |
| A.J   | 03.csv          | member3     |

---





---

## 📞 Need Help?

If you encounter any issues:
1. Take a screenshot of the error message
2. Note which step you're on
3. Send on whatsapp Group

---

## 🎯 Quick Reference Card

```bash
# Complete workflow for Member 1 (example)
python -m venv .venv
.venv\Scripts\Activate.ps1
git branch member1
git checkout member1
python .\desc_generator.py 01.csv
git add .
git commit -m "added data"
git push origin member1
```

Just copy, replace `member1` with your name, and run line by line! 
