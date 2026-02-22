from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import styles
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


def generate_pdf_report(
    filename,
    dataset_name,
    problem_type,
    leaderboard,
    best_model,
    best_score,
    feature_names
):
    doc = SimpleDocTemplate(filename)
    elements = []

    style_sheet = styles.getSampleStyleSheet()
    normal_style = style_sheet["Normal"]
    title_style = style_sheet["Heading1"]

    elements.append(Paragraph("AutoML AI Research Report", title_style))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Dataset: {dataset_name}", normal_style))
    elements.append(Paragraph(f"Problem Type: {problem_type}", normal_style))
    elements.append(Paragraph(f"Best Model: {best_model}", normal_style))
    elements.append(Paragraph(f"Best Score: {best_score}", normal_style))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("Model Leaderboard:", style_sheet["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    table_data = [["Rank", "Model", "Score"]]
    for rank, (model, score) in enumerate(leaderboard, start=1):
        table_data.append([rank, model, score])

    table = Table(table_data)
    table.setStyle(
        TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER')
        ])
    )

    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("Feature List:", style_sheet["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    for feature in feature_names:
        elements.append(Paragraph(feature, normal_style))

    doc.build(elements)
    print(f"\n📄 PDF Report Generated: {filename}")