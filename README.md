# Statistical_Inference_Project
Group 3 project for Statistical inference

## Setup
- Windows (Command Prompt):
Paste the following into terminal before executing code
```bash
git clone https://github.com/RuchithR/Statistical_Inference_Project
cd Statistical_Inference_Project
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

- macOS/Linux: Replace line 4 with
``` source venv/bin/activate ```

- Windows(PowerShell): Replace line 4 with
```.\venv\Scripts\Activate.ps1```

If terminal restarts, run line 4 once again.

## Running the code
All executable scripts are located in the src directory. To run any script, use:

```bash
python src/filename.py
```
replace ```filename``` with whatever script you want to run.

Note: If you are using Windows and facing issues with running the above command, try running ```python src\filename.py``` instead.

To run the R script, use the following command if you have R installed:
```bash
Rscript src/plot_results.R
```