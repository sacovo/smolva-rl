import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, seaborn as sns
sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
W = [0,0.5,1,1.5,2]

# ---------- Fig 0: critic value trace (real critic output on a demo episode) ----------
demo_v = np.load("data/ep345_V_real.npy")
fig, ax = plt.subplots(figsize=(3.4, 1.9))
ax.plot(demo_v, color=sns.color_palette("deep")[0], lw=1.5)
ax.axhline(0, ls=':', c='gray', lw=.8)
ax.annotate("drops back", xy=(58, -0.60), xytext=(105, -0.72), fontsize=7,
            arrowprops=dict(arrowstyle="->", lw=.7, color="gray"))
ax.set_xlabel("step"); ax.set_ylabel("value $V$")
ax.set_ylim(-1.02, 0.08); ax.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig("figs/fig_critic.pdf")
fig.savefig("figs/fig_critic.png", dpi=200)  # web (project page)
plt.close(fig)

# ---------- Fig 1: CFG dose-response (spatial) ----------
curves = {  # spatial cfg curves
  r"$f^{+}$=0.3 (15% pos)":   [67.0,66.6,66.0,67.4,63.0],
  r"$f^{+}$=0.4 (21% pos)":   [72.8,70.8,70.4,67.4,65.4],
  r"$f^{+}$=0.8 (52% pos)":   [76.8,77.8,75.4,72.6,72.4],
}
fig, ax = plt.subplots(figsize=(3.4,2.5))
cols = sns.color_palette("rocket", 3)
for (lab,y),c in zip(curves.items(), cols):
    ax.plot(W,y,'-o',ms=4,lw=1.8,color=c,label=lab)
ax.axvline(0,ls=':',c='gray',lw=.8)
ax.set_xlabel("guidance weight $w$"); ax.set_ylabel("success rate (%)")
ax.set_title("Guidance-weight sweep (LIBERO-spatial)", fontsize=9)
ax.set_xticks(W); ax.legend(fontsize=7, loc="lower left", framealpha=.9)
fig.tight_layout(); fig.savefig("figs/fig_dose.pdf")
fig.savefig("figs/fig_dose.png", dpi=200)  # web (project page)
plt.close(fig)

print("wrote figs/fig_critic.{pdf,png}, figs/fig_dose.{pdf,png}")
