
with open("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/evaluate_and_optimize_test.py") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "from statsmodels" in line:
        new_lines.append("from scipy.stats import binomtest\n")
    elif "mcn = mcnemar(table, exact=True)" in line:
        new_lines.append("""
b = int(table[0, 1])
c = int(table[1, 0])
res_mcn = binomtest(b, n=b+c, p=0.5) if (b + c) > 0 else None
mcn_stat = float((abs(b - c) - 1)**2 / (b + c)) if (b + c) > 0 else 0.0
mcn_pval = float(res_mcn.pvalue) if res_mcn else 1.0
""")
    elif 'f"\\nMcNemar test GLADS-CT vs LungDxFormer:' in line:
        new_lines.append('print(f"\\nMcNemar test GLADS-CT vs LungDxFormer: b={b}, c={c}, p-value={mcn_pval:.4e}")\n')
    elif '"statistic": float(mcn.statistic)' in line:
        new_lines.append('        "statistic": mcn_stat,\n')
    elif '"pvalue": float(mcn.pvalue)' in line:
        new_lines.append('        "pvalue": mcn_pval,\n        "b_discordant": b,\n        "c_discordant": c\n')
    else:
        new_lines.append(line)

with open("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/evaluate_and_optimize_test.py", "w") as f:
    f.writelines(new_lines)
print("File patched successfully!")
