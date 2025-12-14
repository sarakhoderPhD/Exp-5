import argparse, numpy as np, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y", required=True, help="Path to labels .npy file")
    ap.add_argument("--a", required=True, help="Path to model A probabilities .npy (e.g. plain)")
    ap.add_argument("--b", required=True, help="Path to model B probabilities .npy (e.g. context)")
    ap.add_argument("--name-a", default="plain")
    ap.add_argument("--name-b", default="context")
    ap.add_argument("--out", required=True, help="Output path for the plot image")
    ap.add_argument("--bins", type=int, default=40)
    args = ap.parse_args()

    # Load data
    y = np.load(args.y).astype(int)
    pa = np.load(args.a).astype(float)
    pb = np.load(args.b).astype(float)

    # Function to calculate Selective Risk (Error Rate) vs Coverage
    def rc(p, y):
        # Abstention based on confidence (distance from 0.5)
        # Higher s = Higher confidence
        s = np.abs(p - 0.5)
        
        # Sort by confidence: lowest first (so we can drop them to lower coverage)
        order = np.argsort(s) 
        
        cov = []
        risks = []
        
        # Iterate from full coverage down to low coverage
        # k is the start index for the 'keep' set
        # We step by roughly 1% to speed up plotting if needed, or do all points
        # Doing all points:
        for k in range(0, len(y), max(1, len(y)//1000)): 
            keep = order[k:]              # keep the most confident tail
            if len(keep) == 0:
                break
            
            yy = y[keep]
            pp = p[keep] >= 0.5
            
            # Selective Risk = Error Rate on the accepted examples
            # Count where prediction (pp) does NOT match label (yy)
            n_errors = (pp != yy).sum()
            risk = n_errors / len(keep)
            
            cov.append(len(keep) / len(y))
            risks.append(risk)
            
        return np.array(cov), np.array(risks)

    # Calculate curves
    cov_a, risk_a = rc(pa, y)
    cov_b, risk_b = rc(pb, y)

    # Calculate AURC (Area Under Risk-Coverage). Lower is better.
    # We integrate Risk with respect to Coverage.
    aurc_a = np.trapz(risk_a, cov_a)
    aurc_b = np.trapz(risk_b, cov_b)

    # Plotting
    plt.figure(figsize=(6, 5), dpi=140)
    plt.plot(cov_a, risk_a, label=f"{args.name_a} (AURC={aurc_a:.4f})")
    plt.plot(cov_b, risk_b, label=f"{args.name_b} (AURC={aurc_b:.4f})")
    
    plt.xlabel("Coverage (kept %)")
    plt.ylabel("Selective Risk (Error Rate)")
    plt.title("Risk–Coverage (external)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out)
    
    print(f"Wrote {args.out}")
    print(f"AURC (Lower is better): {args.name_a}={aurc_a:.6f}, {args.name_b}={aurc_b:.6f}")
    
    # Validation check for your thesis table
    # Print the risk at specific coverage points to compare with your Table 4.4
    print("\n--- Validation Check (Compare with Table 4.4) ---")
    for name, c_arr, r_arr in [(args.name_a, cov_a, risk_a), (args.name_b, cov_b, risk_b)]:
        print(f"Model: {name}")
        for target in [1.0, 0.99, 0.98, 0.95, 0.90]:
            # Find closest real coverage
            idx = (np.abs(c_arr - target)).argmin()
            print(f"  Target Cov {target:.2f}: Actual {c_arr[idx]:.4f} -> Risk {r_arr[idx]:.4%}")

if __name__ == "__main__":
    main()