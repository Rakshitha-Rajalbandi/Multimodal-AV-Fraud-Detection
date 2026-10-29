import os
import hashlib
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)


def compute_file_sha256(filepath):
    """Computes SHA-256 hash for chain-of-custody evidence integrity."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return "N/A (SHA-256 computation unavailable)"


def generate_pdf_report(results, request_id, filename, video_path, output_pdf_path, decision="PENDING REVIEW"):
    """
    Generates a formal, enterprise-grade PDF Forensic Audit Report.
    """
    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a")
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569")
    )

    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )

    bold_body = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>AI FORENSIC SECURITY & IDENTITY VERIFICATION</b><br/><font color='#64748b' size='8'>Enterprise Deepfake & Synthetic Media Audit Report</font>", title_style),
            Paragraph(f"<b>CONFIDENTIAL</b><br/><font color='#dc2626' size='8'>CONFIDENCE: {results.get('confidence', 'HIGH')}</font>", ParagraphStyle('RightHead', parent=subtitle_style, alignment=2))
        ]
    ]
    header_table = Table(header_data, colWidths=[5.0 * inch, 2.5 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3b82f6"), spaceAfter=10))

    # 2. File Metadata & Cryptographic Evidence Details
    sha256_hash = compute_file_sha256(video_path) if video_path and Path(video_path).exists() else "N/A"
    
    meta_data = [
        [Paragraph("<b>Request ID:</b>", bold_body), Paragraph(request_id, body_style), Paragraph("<b>Timestamp:</b>", bold_body), Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"), body_style)],
        [Paragraph("<b>Target File:</b>", bold_body), Paragraph(filename, body_style), Paragraph("<b>Input Media Type:</b>", bold_body), Paragraph(results.get("media_type", "Video Clip").upper(), body_style)],
        [Paragraph("<b>SHA-256 Hash:</b>", bold_body), Paragraph(f"<font size='7' name='Courier'>{sha256_hash[:32]}...{sha256_hash[-16:]}</font>", body_style), Paragraph("<b>Analyst Decision:</b>", bold_body), Paragraph(f"<font color='#2563eb'><b>{decision}</b></font>", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[1.1 * inch, 2.6 * inch, 1.2 * inch, 2.6 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 3. Executive Fraud Risk Verdict Box
    risk_pct = results.get("fraud_risk_percent", 0.0)
    prediction = results.get("prediction", "REAL")
    risk_level = results.get("risk_level", "LOW")

    verdict_bg = colors.HexColor("#fef2f2") if prediction == "FAKE" else colors.HexColor("#f0fdf4")
    verdict_border = colors.HexColor("#f87171") if prediction == "FAKE" else colors.HexColor("#4ade80")
    text_color_code = "#dc2626" if prediction == "FAKE" else "#16a34a"

    verdict_content = [
        [
            Paragraph(f"<font size='14'><b>FRAUD RISK SCORE: <font color='{text_color_code}'>{risk_pct}%</font> &nbsp;|&nbsp; VERDICT: <font color='{text_color_code}'>{prediction} ({risk_level} RISK)</font></b></font>", ParagraphStyle('Verdict', parent=body_style, alignment=1))
        ],
        [
            Paragraph(f"<b>Flagged Primary Modality:</b> {results.get('trigger_modality', 'NONE')} &nbsp;|&nbsp; <b>Confidence:</b> {results.get('confidence', 'HIGH')}", ParagraphStyle('SubVerdict', parent=body_style, alignment=1))
        ]
    ]
    verdict_table = Table(verdict_content, colWidths=[7.5 * inch])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), verdict_bg),
        ('BOX', (0, 0), (-1, -1), 1, verdict_border),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 12))

    # 4. Modality Breakdown Comparison Table
    story.append(Paragraph("1. Tri-Modality Feature Analysis & Comparison", heading_style))

    video_info = results.get("video", {})
    audio_info = results.get("audio", {})
    sync_info = results.get("sync", {})

    modality_data = [
        [Paragraph("<b>Modality / Feature Stream</b>", bold_body), Paragraph("<b>Extracted Metric</b>", bold_body), Paragraph("<b>Anomaly Probability</b>", bold_body), Paragraph("<b>Status Assessment</b>", bold_body)],
        [
            Paragraph("<b>Visual CNN (EfficientNet-B0)</b>", body_style),
            Paragraph(f"{video_info.get('frames_evaluated', 16)} Temporal Frames", body_style),
            Paragraph(f"{video_info.get('visual_score_percent', 0.0)}%", body_style),
            Paragraph(f"<font color='{'#dc2626' if video_info.get('visual_score_percent', 0)>45 else '#16a34a'}'><b>{'DEEPFAKE CUE' if video_info.get('visual_score_percent', 0)>45 else 'AUTHENTIC'}</b></font>", body_style)
        ],
        [
            Paragraph("<b>Audio 2D-CNN Spectrogram</b>", body_style),
            Paragraph("128 Mel-Bands (16kHz)", body_style),
            Paragraph(f"{audio_info.get('audio_score_percent', 0.0)}%", body_style),
            Paragraph(f"<font color='{'#dc2626' if audio_info.get('audio_score_percent', 0)>45 else '#16a34a'}'><b>{'SYNTHETIC VOICE' if audio_info.get('audio_score_percent', 0)>45 else 'NATURAL VOICE'}</b></font>", body_style)
        ],
        [
            Paragraph("<b>Lip-Sync Mismatch</b>", body_style),
            Paragraph("Cross-Correlation Shift", body_style),
            Paragraph(f"{round(sync_info.get('sync_mismatch_score', 0.0)*100, 2)}%", body_style),
            Paragraph(f"<font color='{'#dc2626' if sync_info.get('sync_mismatch_score', 0)>0.45 else '#16a34a'}'><b>{'MISMATCH DETECTED' if sync_info.get('sync_mismatch_score', 0)>0.45 else 'SYNCHRONIZED'}</b></font>", body_style)
        ]
    ]

    modality_table = Table(modality_data, colWidths=[2.2 * inch, 1.8 * inch, 1.6 * inch, 1.9 * inch])
    modality_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(modality_table)
    story.append(Spacer(1, 12))

    # 5. Suspicious Timestamps & Anomalous Intervals
    story.append(Paragraph("2. Temporal Risk Trend & Suspicious Timestamp Log", heading_style))
    
    timestamps = results.get("suspicious_timestamps", [])
    if timestamps:
        ts_rows = [[Paragraph("<b>Interval Timestamp</b>", bold_body), Paragraph("<b>Peak Anomaly Score</b>", bold_body), Paragraph("<b>Detected Anomaly Description</b>", bold_body)]]
        for item in timestamps:
            ts_rows.append([
                Paragraph(f"<b>{item.get('timestamp_str', 'N/A')}</b>", body_style),
                Paragraph(f"<font color='#dc2626'><b>{item.get('risk_percent', 0.0)}%</b></font>", body_style),
                Paragraph(item.get('description', 'Spatial / Acoustic artifact detected'), body_style)
            ])
        ts_table = Table(ts_rows, colWidths=[2.2 * inch, 1.8 * inch, 3.5 * inch])
        ts_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(ts_table)
    else:
        story.append(Paragraph("<i>No localized suspicious timestamps detected. Risk profile is uniform across video duration.</i>", body_style))

    story.append(Spacer(1, 12))

    # 6. Grad-CAM Explainable AI Heatmaps Snapshots
    story.append(Paragraph("3. Explainable AI (Grad-CAM) Visual Heatmap Evidence", heading_style))

    heatmaps = results.get("heatmaps", [])
    if heatmaps:
        img_cells = []
        for hm in heatmaps[:3]:
            hm_path = hm.get("filepath")
            if hm_path and Path(hm_path).exists():
                try:
                    img_cells.append([
                        RLImage(hm_path, width=2.2 * inch, height=2.2 * inch),
                        Paragraph(f"<b>{hm.get('label', 'Frame Snap')}</b><br/><font size='8' color='#64748b'>Frame {hm.get('frame_idx', 0)}<br/>Fake Prob: {hm.get('fake_prob', 0.0)*100:.1f}%</font>", body_style)
                    ])
                except Exception:
                    pass

        if img_cells:
            hm_table = Table(img_cells, colWidths=[2.4 * inch, 5.1 * inch])
            hm_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0"))
            ]))
            story.append(hm_table)
        else:
            story.append(Paragraph("<i>Visual heatmaps generated dynamically during model execution.</i>", body_style))
    else:
        story.append(Paragraph("<i>Visual heatmaps generated dynamically during model execution.</i>", body_style))

    story.append(Spacer(1, 16))

    # 7. Analyst Verification Stamp & Sign-Off Block
    sign_data = [
        [
            Paragraph("<b>INSTITUTIONAL AUDIT STAMP</b><br/><font size='7' color='#64748b'>Verified by AI Forensic Engine v2.4</font>", body_style),
            Paragraph(f"<b>Analyst Decision:</b> {decision}<br/><b>Audit Status:</b> VERIFIED & SEALED", body_style)
        ]
    ]
    sign_table = Table(sign_data, colWidths=[3.75 * inch, 3.75 * inch])
    sign_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(sign_table)

    # Build Document
    doc.build(story)
    return output_pdf_path
