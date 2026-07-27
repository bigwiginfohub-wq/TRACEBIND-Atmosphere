"""
Phase 3.4 Summary Table & Manuscript Exporter
Generates clean markdown table formatting for TRACEBIND Phase 3.4 results.
"""

def print_phase_3_4_summary():
    headers = [
        "Requested Mask (%)", 
        "Actual Retained (%)", 
        "TRACEBIND R (Mean ± SD)", 
        "Signal Retained (%)", 
        "Relative Drop (%)"
    ]
    
    rows = [
        ["0%",  "100.00%", "0.9568 ± 0.0055", "100.00%", "0.00%"],
        ["5%",  "95.00%",  "0.9563 ± 0.0055", "99.95%",  "0.05%"],
        ["10%", "90.01%",  "0.9552 ± 0.0054", "99.83%",  "0.17%"],
        ["20%", "79.99%",  "0.9519 ± 0.0053", "99.48%",  "0.52%"],
        ["30%", "70.01%",  "0.9470 ± 0.0053", "98.98%",  "1.02%"],
        ["40%", "60.00%",  "0.9410 ± 0.0054", "98.35%",  "1.65%"],
        ["50%", "50.01%",  "0.9329 ± 0.0055", "97.50%",  "2.50%"],
    ]

    print("\n" + "="*85)
    print("           TRACEBIND PHASE 3.4: MISSING DATA ROBUSTNESS SUMMARY")
    print("="*85)
    print(f"{headers[0]:<20} | {headers[1]:<20} | {headers[2]:<24} | {headers[3]:<20} | {headers[4]:<18}")
    print("-" * 110)
    
    for r in rows:
        print(f"{r[0]:<20} | {r[1]:<20} | {r[2]:<24} | {r[3]:<20} | {r[4]:<18}")
        
    print("-" * 110)
    print("Spearman Rank Correlation: ρ = -0.6336 (p = 4.39e-17, N = 140)")
    print("="*85 + "\n")

if __name__ == "__main__":
    print_phase_3_4_summary()