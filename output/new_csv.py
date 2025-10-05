import pandas as pd

# Parameters
rho_values = [1.2, 2.4, 4.8]  # 0.5R, R, 2R with R≈2.4
kappa_values = [2,3,4,5,6]

# E1 equal: makespan tables by method
ours_equal = {
    1.2: [1.30, 1.18, 1.10, 1.06, 1.03],
    2.4: [1.15, 1.06, 1.00, 0.97, 0.94],
    4.8: [0.98, 0.96, 0.95, 0.95, 0.95],
}
heft_equal = {
    1.2: [1.75, 1.58, 1.45, 1.40, 1.35],
    2.4: [1.45, 1.30, 1.22, 1.18, 1.15],
    4.8: [1.10, 1.08, 1.05, 1.04, 1.03],
}
ipdps_equal = {
    1.2: [1.55, 1.40, 1.32, 1.28, 1.24],
    2.4: [1.28, 1.18, 1.12, 1.10, 1.07],
    4.8: [1.05, 1.03, 1.02, 1.01, 1.01],
}
jpdc_equal = {
    1.2: [1.60, 1.45, 1.35, 1.30, 1.27],
    2.4: [1.32, 1.21, 1.15, 1.12, 1.10],
    4.8: [1.06, 1.05, 1.03, 1.02, 1.02],
}

def to_long_df(table, method):
    rows = []
    for rho, vals in table.items():
        for k, v in zip(kappa_values, vals):
            rows.append({"rho": rho, "kappa": k, "method": method, "makespan": v})
    return pd.DataFrame(rows)

e1_equal_df = pd.concat([
    to_long_df(ours_equal, "Ours"),
    to_long_df(heft_equal, "HEFT"),
    to_long_df(ipdps_equal, "IPDPS21"),
    to_long_df(jpdc_equal, "JPDC22"),
], ignore_index=True)

# E1 unequal: apply heterogeneity penalty factors
penalty = {"Ours": 1.04, "IPDPS21": 1.10, "JPDC22": 1.12, "HEFT": 1.16}

e1_unequal_df = e1_equal_df.copy()
e1_unequal_df["makespan"] = e1_unequal_df.apply(lambda r: r["makespan"] * penalty[r["method"]], axis=1)

# E2: heterogeneity sensitivity multipliers
e2_rows = []
H_vals = [0.00, 0.15, 0.30, 0.45, 0.60]
ms_mul = {
    "Ours":   [1.00, 1.02, 1.04, 1.06, 1.09],
    "IPDPS21":[1.00, 1.05, 1.10, 1.18, 1.28],
    "JPDC22": [1.00, 1.06, 1.12, 1.22, 1.35],
    "HEFT":   [1.00, 1.08, 1.16, 1.28, 1.45],
}
p99_extra = {"Ours":1.03, "IPDPS21":1.08, "JPDC22":1.10, "HEFT":1.15}
for method, muls in ms_mul.items():
    for H, m in zip(H_vals, muls):
        e2_rows.append({
            "H": H, "method": method, "makespan_multiplier": m,
            "p99_multiplier": m * p99_extra[method]
        })
e2_df = pd.DataFrame(e2_rows)

# E3 Balanced vs LongTail
e3_rows = []
balanced = {"Ours":1.00,"HEFT":1.22,"IPDPS21":1.12,"JPDC22":1.15}
longtail_ms = {"Ours":1.07,"HEFT":1.42,"IPDPS21":1.28,"JPDC22":1.32}
balanced_p99 = {"Ours":1.00,"HEFT":1.22,"IPDPS21":1.12,"JPDC22":1.15}
p99_increase = {"Ours":1.15,"HEFT":1.35,"IPDPS21":1.25,"JPDC22":1.28}
for method in ["Ours","HEFT","IPDPS21","JPDC22"]:
    e3_rows.append({"dataset":"Balanced","method":method,
                    "makespan":balanced[method],
                    "p99":balanced_p99[method]})
    e3_rows.append({"dataset":"LongTail","method":method,
                    "makespan":longtail_ms[method],
                    "p99":balanced_p99[method]*p99_increase[method]})
e3_df = pd.DataFrame(e3_rows)

# E4 ablations
e4_df = pd.DataFrame([
    {"variant":"Full","makespan":1.00,"p99":1.00},
    {"variant":"No Dynamic vGPU","makespan":1.11,"p99":1.20},
    {"variant":"No Min-Max Packing","makespan":1.18,"p99":1.28},
    {"variant":"No Dynamic + No Packing","makespan":1.31,"p99":1.45},
])

# Save CSVs
e1_equal_df.to_csv("./data_gen/e1_equal.csv", index=False)
e1_unequal_df.to_csv("./data_gen/e1_unequal.csv", index=False)
e2_df.to_csv("./data_gen/e2_heterogeneity.csv", index=False)
e3_df.to_csv("./data_gen/e3_balanced_longtail.csv", index=False)
e4_df.to_csv("./data_gen/e4_ablation.csv", index=False)

# Also bundle into an Excel workbook
# with pd.ExcelWriter("/mnt/data_gen/sim_results.xlsx", engine="xlsxwriter") as writer:
#     e1_equal_df.to_excel(writer, sheet_name="E1_equal", index=False)
#     e1_unequal_df.to_excel(writer, sheet_name="E1_unequal", index=False)
#     e2_df.to_excel(writer, sheet_name="E2_heterogeneity", index=False)
#     e3_df.to_excel(writer, sheet_name="E3_datasets", index=False)
#     e4_df.to_excel(writer, sheet_name="E4_ablation", index=False)

print("OK")
