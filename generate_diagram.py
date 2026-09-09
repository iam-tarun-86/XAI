import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_multimodal_diagram():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_facecolor('#09090b')
    fig.patch.set_facecolor('#09090b')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Title
    ax.text(50, 93, "Multimodal Intermediate Neural Fusion XAI Framework", 
            ha='center', va='center', color='#f4f4f5', fontsize=16, fontweight='bold')
    ax.text(50, 89, "13 Clinical Tabular Features + 1D Lead-II ECG Signal | 1D CNN Grad-CAM + SHAP + LIME", 
            ha='center', va='center', color='#06b6d4', fontsize=10, fontfamily='monospace')

    # Box Helper Function
    def draw_box(x, y, w, h, title, subtitle, color, border_color):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", 
                                     linewidth=1.5, edgecolor=border_color, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 2, title, ha='center', va='center', color='#ffffff', fontsize=10, fontweight='bold')
        ax.text(x + w/2, y + h/2 - 3, subtitle, ha='center', va='center', color='#a1a1aa', fontsize=8)

    # Input Modalities
    draw_box(5, 60, 22, 16, "01. UCI Tabular Data", "920 Patient Records\n(13 Parameters)", "#18181b", "#3b82f6")
    draw_box(5, 25, 22, 16, "02. 1D ECG Signal", "1,000 Sampling Points\n(PhysioNet Lead-II)", "#18181b", "#06b6d4")

    # Feature Encoders
    draw_box(33, 60, 22, 16, "ClinicalEncoder", "Dense(128) -> BatchNorm\n64-d Embedding", "#18181b", "#3b82f6")
    draw_box(33, 25, 22, 16, "ECG1DCNNEncoder", "3-Stage Conv1D (16->32->64)\n1D Activation Maps", "#18181b", "#06b6d4")

    # Fusion Node
    draw_box(61, 42.5, 18, 18, "Intermediate Fusion", "Concatenated Vector\n[128-d Embedding]", "#18181b", "#8b5cf6")

    # Prediction Head
    draw_box(83, 42.5, 14, 18, "Prediction Head", "Dense(64) -> Sigmoid\nP(Heart Attack)", "#18181b", "#ef4444")

    # Connectors
    ax.annotate("", xy=(33, 68), xytext=(27, 68), arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=2))
    ax.annotate("", xy=(33, 33), xytext=(27, 33), arrowprops=dict(arrowstyle="->", color="#06b6d4", lw=2))

    ax.annotate("", xy=(61, 55), xytext=(55, 68), arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=2))
    ax.annotate("", xy=(61, 48), xytext=(55, 33), arrowprops=dict(arrowstyle="->", color="#06b6d4", lw=2))

    ax.annotate("", xy=(83, 51.5), xytext=(79, 51.5), arrowprops=dict(arrowstyle="->", color="#8b5cf6", lw=2))

    # XAI Attribution Section (Bottom)
    ax.text(50, 15, "Tri-Branch XAI Attribution: 1D ECG Grad-CAM (Temporal) | Tabular SHAP | Tabular LIME | Modality Ablation", 
            ha='center', va='center', color='#a1a1aa', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig('xai_framework_diagram.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.savefig('xai_framework_diagram.pdf', facecolor=fig.get_facecolor(), bbox_inches='tight')
    print("Architecture diagram saved as xai_framework_diagram.png & .pdf")

if __name__ == '__main__':
    generate_multimodal_diagram()
