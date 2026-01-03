How to run the file:

# With new log format (subdirectories):
python scripts/visualize_lives.py logs/gemini_base/20260102_123456_2_5_gemini_pro_Gemini_base_game0_1234.log

# Process all logs in a subdirectory:
python scripts/visualize_lives.py logs/bg_probabilistic --out output_directory/

# Process all logs across all subdirectories:
python scripts/visualize_lives.py logs --out output_directory/

# Specify custom output file:
python scripts/visualize_lives.py logs/gemini_base/some_log.log --out custom_plot.png

# Old format (still supported):
python scripts/visualize_lives.py logs/agent_BG_probabilistic_1785_20251227_165044.log