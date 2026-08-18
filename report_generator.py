import os
import time
import base64
import io
from pathlib import Path
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# Import constants from predict.py
from predict import SR, N_MELS, N_FFT, HOP_LENGTH, CLASSES

def analyze_spectral_properties(spec: np.ndarray, salient: tuple) -> dict:
    """Analyze the spectral properties of the salient region to generate a clinical description.
    
    spec: 2D numpy array of mel spectrogram (db scaled, ref=max, range roughly -80 to 0)
    salient: (t0, t1) seconds of the salient region
    """
    try:
        t0, t1 = salient
        # Convert times to frame indices
        f0 = int((t0 * SR) / HOP_LENGTH)
        f1 = int((t1 * SR) / HOP_LENGTH)
        
        # Clamp frame indices to spectrogram boundaries
        n_frames = spec.shape[1]
        f0 = max(0, min(f0, n_frames - 1))
        f1 = max(0, min(f1, n_frames))
        if f1 <= f0:
            f1 = min(f0 + 1, n_frames)
            
        salient_spec_db = spec[:, f0:f1]
        
        # Convert dB back to power for linear analysis
        salient_power = librosa.db_to_power(salient_spec_db)
        
        # Find frequency bin with peak energy
        mean_freq_profile = np.mean(salient_power, axis=1)
        peak_bin = int(np.argmax(mean_freq_profile))
        peak_freq_hz = float(librosa.mel_to_hz(peak_bin))
        
        # Estimate spectral bandwidth / frequency spread (entropy of the power distribution)
        normalized_freq_profile = mean_freq_profile / (np.sum(mean_freq_profile) + 1e-9)
        spectral_entropy = -np.sum(normalized_freq_profile * np.log2(normalized_freq_profile + 1e-9))
        
        # Estimate temporal variance (transient vs sustained energy)
        frame_energies = np.sum(salient_power, axis=0)
        temporal_std = np.std(frame_energies)
        temporal_mean = np.mean(frame_energies) + 1e-9
        temporal_cv = temporal_std / temporal_mean # Coefficient of variation
        
        # Map findings to clinical description tags
        is_broadband = spectral_entropy > 4.2
        is_transient = temporal_cv > 0.4
        
        description = ""
        if is_broadband and is_transient:
            description = (
                f"short broadband energy bursts peaking around {peak_freq_hz:.0f} Hz. "
                "This spectral signature exhibits high transient fluctuation and wide frequency spread, "
                "consistent with discontinuous clicking sounds (such as crackles)."
            )
        elif not is_broadband and not is_transient:
            description = (
                f"sustained narrowband energy concentrated around {peak_freq_hz:.0f} Hz. "
                "This spectral signature exhibits continuous tonal/musical properties with low temporal variance, "
                "consistent with continuous musical sound (such as wheezes)."
            )
        elif is_broadband:
            description = (
                f"broadband sound spread across a wide frequency range (peaking at {peak_freq_hz:.0f} Hz). "
                "The energy is distributed across multiple frequency bands, indicating a mixed or noisy sound profile."
            )
        else:
            description = (
                f"narrowband energy peaks at {peak_freq_hz:.0f} Hz. "
                "The concentration of energy in this frequency band suggests a localized, tonal sound profile."
            )
            
        return {
            "peak_freq_hz": peak_freq_hz,
            "spectral_entropy": spectral_entropy,
            "temporal_cv": temporal_cv,
            "description": description
        }
    except Exception as e:
        return {
            "peak_freq_hz": 0.0,
            "spectral_entropy": 0.0,
            "temporal_cv": 0.0,
            "description": f"energy characteristics peaking around the salient frequencies. (Analysis error: {e})"
        }

def generate_report_spectrogram_image(spec: np.ndarray, salient: tuple, heatmap_timeline: list = None) -> str:
    """Renders the spectrogram with heatmap overlay and returns it as a base64 encoded PNG string."""
    duration = 5.0
    fig, ax = plt.subplots(figsize=(8, 2.5), dpi=150)
    ax.imshow(np.flipud(spec), aspect="auto", origin="upper", cmap="magma",
              extent=[0, duration, 0, N_MELS])
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mel Frequency Bin")
    ax.set_title("Mel Spectrogram with Acoustic Evidence Overlay")
    
    if heatmap_timeline:
        max_drop = max([drop for _, _, drop in heatmap_timeline] + [0.01])
        for t0, t1, drop in heatmap_timeline:
            if drop > 0:
                alpha_val = min(0.4, max(0.05, (drop / max_drop) * 0.35))
                color_val = "red" if drop > 0.5 * max_drop else "cyan"
                ax.axvspan(t0, t1, color=color_val, alpha=alpha_val)
                
    if salient:
        ax.axvline(salient[0], color="cyan", linestyle="--", linewidth=1.5)
        ax.axvline(salient[1], color="cyan", linestyle="--", linewidth=1.5, label="Peak salient segment")
        ax.legend(fontsize=8, loc="upper right")
        
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_report(result: dict, audio_path: str) -> str:
    """Generate a clean clinical summary report in HTML format.
    Returns the file path to the generated HTML report.
    """
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    filename = Path(audio_path).name
    
    # Calculate spectral properties of the salient region
    spectral_analysis = analyze_spectral_properties(result["spec"], result["salient"])
    
    # Generate spectrogram image
    img_b64 = generate_report_spectrogram_image(
        result["spec"], 
        result["salient"], 
        result.get("heatmap_timeline")
    )
    
    # Format probabilities
    probs_html = ""
    for cls_name, prob_val in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
        badge_color = "#4CAF50" if prob_val >= 0.70 else ("#FFC107" if prob_val >= 0.50 else "#F44336")
        probs_html += f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>{cls_name.capitalize()}</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">
                <span style="display:inline-block; width:100px; background:#eee; border-radius:4px; height:12px; margin-right:8px; text-align:left;">
                    <span style="display:inline-block; width:{prob_val*100}%; background:{badge_color}; height:100%; border-radius:4px;"></span>
                </span>
                <span style="font-weight: bold; color: {badge_color};">{prob_val:.1%}</span>
            </td>
        </tr>
        """
        
    # Format breath cycles table
    cycles_html = ""
    if result.get("cycles"):
        for i, (t0, t1, label) in enumerate(result["cycles"]):
            cycles_html += f"""
            <tr>
                <td style="padding: 6px; border-bottom: 1px solid #eee; text-align: center;">Cycle {i+1}</td>
                <td style="padding: 6px; border-bottom: 1px solid #eee; text-align: center;">{t0:.2f}s - {t1:.2f}s</td>
                <td style="padding: 6px; border-bottom: 1px solid #eee; text-align: center; font-weight: bold;">{label.upper()}</td>
            </tr>
            """
            
    # Formulate reasoning paragraph
    salient_t0, salient_t1 = result["salient"]
    peak_drop = max([drop for _, _, drop in result.get("heatmap_timeline", [])] + [0.0])
    reasoning_text = (
        f"The model's classification is primarily driven by the audio segment at "
        f"<strong>{salient_t0:.2f}s - {salient_t1:.2f}s</strong>, where temporarily muting this signal "
        f"caused the prediction confidence to drop by <strong>{peak_drop:.1%}</strong>. This salient region shows "
        f"<strong>{spectral_analysis['description']}</strong>"
    )
    
    # Quality status HTML
    quality_color = "#4CAF50" if result["quality"] == "good" else "#FFC107"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EchoAssist Summary Report - {filename}</title>
    <style>
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #333;
            line-height: 1.5;
            padding: 40px;
            max-width: 800px;
            margin: 0 auto;
            background-color: #f9f9f9;
        }}
        .report-card {{
            background: #fff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
        }}
        .header {{
            border-bottom: 2px solid #243B53;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }}
        .header-title {{
            font-size: 24px;
            font-weight: bold;
            color: #243B53;
            margin: 0;
        }}
        .meta-table {{
            width: 100%;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        .section-title {{
            font-size: 16px;
            font-weight: bold;
            color: #243B53;
            border-left: 4px solid #243B53;
            padding-left: 10px;
            margin-top: 30px;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .result-badge {{
            display: inline-block;
            padding: 8px 16px;
            background-color: #243B53;
            color: #fff;
            font-weight: bold;
            border-radius: 6px;
            font-size: 18px;
            text-transform: uppercase;
        }}
        .disclaimer-footer {{
            margin-top: 40px;
            border-top: 1px solid #ddd;
            padding-top: 15px;
            font-size: 11px;
            color: #777;
            text-align: center;
            line-height: 1.4;
        }}
    </style>
</head>
<body>

<div class="report-card">
    <div class="header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="header-title">EchoAssist Summary Report</div>
            <div style="font-size: 12px; color: #666;">Generated: {timestamp_str}</div>
        </div>
    </div>
    
    <table class="meta-table">
        <tr>
            <td style="width: 50%;"><strong>Source File:</strong> {filename}</td>
            <td style="width: 50%; text-align: right;"><strong>Analysis Version:</strong> 2.0 (ResNet18)</td>
        </tr>
    </table>
    
    <div class="section-title">Acoustic Classification Results</div>
    <div style="display: flex; justify-content: space-between; align-items: center; background: #f5f7fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <div>
            <div style="font-size: 12px; color: #666; margin-bottom: 4px;">Primary Acoustic Feature</div>
            <div class="result-badge">{result['label']}</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 12px; color: #666; margin-bottom: 4px;">Overall Confidence</div>
            <div style="font-size: 24px; font-weight: bold; color: #243B53;">{result['confidence']:.1%}</div>
        </div>
    </div>
    
    <div style="display: flex; gap: 20px; margin-bottom: 25px;">
        <div style="flex: 1;">
            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                <thead>
                    <tr style="background: #f5f7fa;">
                        <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Acoustic Class</th>
                        <th style="padding: 8px; text-align: right; border-bottom: 2px solid #ddd;">Probability</th>
                    </tr>
                </thead>
                <tbody>
                    {probs_html}
                </tbody>
            </table>
        </div>
        <div style="flex: 1; border: 1px solid #eee; border-radius: 8px; padding: 15px; background: #fafafa;">
            <div style="font-weight: bold; font-size: 14px; margin-bottom: 10px; color: #243B53;">Signal Quality Status</div>
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                <span style="width: 12px; height: 12px; border-radius: 50%; background: {quality_color}; display: inline-block;"></span>
                <strong style="text-transform: uppercase; color: {quality_color};">{result['quality']}</strong>
            </div>
            <div style="font-size: 13px; color: #555;">
                {result.get('quality_reason', 'Passed all standard signal quality gates.')}
            </div>
        </div>
    </div>
    
    <div class="section-title">Evidence-Based Reasoning</div>
    <div style="background: #f9fbfd; border: 1px solid #dceaf6; padding: 15px; border-radius: 8px; font-size: 14px; color: #243b53; text-align: justify; margin-bottom: 25px;">
        {reasoning_text}
    </div>
    
    <div class="section-title">Acoustic Evidence Heatmap Overlay</div>
    <div style="text-align: center; margin-bottom: 25px;">
        <img src="data:image/png;base64,{img_b64}" alt="Mel Spectrogram Evidence Heatmap" style="max-width: 100%; border-radius: 6px; border: 1px solid #ddd;" />
        <div style="font-size: 11px; color: #666; margin-top: 6px;">
            Red shaded areas mark segments that significantly drove classification confidence.
        </div>
    </div>
    
    {f'<div class="section-title">Segment Timeline (Breath Cycles)</div>' if cycles_html else ''}
    {f'''<table style="width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 20px;">
        <thead>
            <tr style="background: #f5f7fa; border-bottom: 2px solid #ddd;">
                <th style="padding: 6px;">Breath Segment</th>
                <th style="padding: 6px;">Time Interval</th>
                <th style="padding: 6px;">Classified Feature</th>
            </tr>
        </thead>
        <tbody>
            {cycles_html}
        </tbody>
    </table>''' if cycles_html else ''}
    
    <div class="disclaimer-footer">
        <strong>IMPORTANT CLINICAL DISCLAIMER:</strong><br/>
        EchoAssist is an acoustic decision support tool intended solely for clinical research and reference purposes. 
        It analyzes and describes acoustic lung sound characteristics only. This software does NOT output a medical diagnosis, 
        and should never replace professional medical judgment, physical examination, or diagnostic procedures.
    </div>
</div>

</body>
</html>
"""
    # Save the report as an HTML file in the project reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"echoassist_clinical_report_{filename.replace('.wav', '')}_{int(time.time())}.html"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    return str(report_path.absolute())
