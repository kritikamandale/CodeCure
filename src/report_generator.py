import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from pathlib import Path
import datetime

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 18)
        self.cell(0, 10, 'CodeCure Epidemic Intelligence Report', align='C', ln=1)
        self.set_font('Helvetica', 'I', 10)
        self.cell(0, 8, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', align='C', ln=1)
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C', ln=1)

def generate_interpretation(country, metrics, risk):
    score = risk.get("score", 0)
    category = risk.get("category", "Unknown")
    growth = float(risk.get("forecast_growth", 0)) * 100
    positivity = float(risk.get("positivity_rate", 0)) * 100
    
    interpretation = f"The automated assessment for {country} indicates a {category} risk level (Score: {score}/100). "
    if growth > 5:
        interpretation += f"Forecasts project a significant growth trajectory of ~{growth:.1f}% in active cases over the coming horizon. "
    elif growth < 0:
        interpretation += f"Forecasts indicate a potential decline of ~{abs(growth):.1f}% in active cases over the horizon. "
    else:
        interpretation += f"Forecasts show a relatively stable trajectory with ~{growth:.1f}% growth. "
        
    if positivity > 10:
        interpretation += f"Alarmingly, the test positivity rate stands at {positivity:.1f}%, suggesting widespread community transmission."
    else:
        interpretation += f"The current test positivity rate of {positivity:.1f}% suggests manageable overall transmission dynamics."
        
    return interpretation

def create_forecast_plot(country, historical_df, forecast_df):
    """Generates a small matplotlib snippet of the forecast for the PDF."""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot historical
    hist = historical_df.tail(60)
    ax.plot(hist['date'], hist['confirmed'], color='#3498db', label='Historical', linewidth=2)
    
    if forecast_df is not None and not forecast_df.empty:
        ax.plot(forecast_df['date'], forecast_df['forecast'], color='#e74c3c', linestyle='--', label='Forecast', linewidth=2)
        ax.fill_between(forecast_df['date'], forecast_df['lower_95'], forecast_df['upper_95'], color='#e74c3c', alpha=0.1)
    
    ax.set_title(f"Epidemiological Trajectory & Forecast - {country}", fontsize=14, pad=10)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Confirmed Cases", fontsize=11)
    ax.grid(color='gray', alpha=0.2)
    ax.legend()
    fig.autofmt_xdate()
    
    plot_path = OUTPUT_DIR / "temp_report_forecast.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return plot_path

def generate_pdf_report(country, historical_df, forecast_df, metrics, risk):
    pdf = PDFReport()
    pdf.add_page()
    
    # Title Section
    pdf.set_font('Helvetica', 'B', 22)
    pdf.cell(0, 15, f'Regional Profile: {country}', align='L', ln=1)
    pdf.ln(2)
    
    # Risk & Status summary
    pdf.set_font('Helvetica', 'B', 14)
    if risk.get("score", 0) >= 50:
        pdf.set_text_color(220, 50, 50)
    else:
        pdf.set_text_color(50, 150, 50)
        
    pdf.cell(0, 10, f'Overall Risk Level: {risk.get("category", "Unknown").upper()} ({risk.get("score", 0)}/100)', align='L', ln=1)
    
    # Key Metrics
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Current Key Metrics', align='L', ln=1)
    
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 8, f'> Total Confirmed Cases: {int(metrics.get("confirmed", 0)):,}', align='L', ln=1)
    pdf.cell(0, 8, f'> Total Deaths: {int(metrics.get("deaths", 0)):,}', align='L', ln=1)
    pdf.cell(0, 8, f'> 7-Day Average Cases: {float(metrics.get("cases_7day_avg", 0)):,.0f}', align='L', ln=1)
    pdf.cell(0, 8, f'> Positivity Rate: {float(metrics.get("positive_rate", 0)):.2%}', align='L', ln=1)
    
    pdf.ln(5)
    
    # Visuals
    plot_path = create_forecast_plot(country, historical_df, forecast_df)
    if plot_path.exists():
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, 'Automated Forecast Projection', align='L', ln=1)
        pdf.image(str(plot_path), w=170)
        pdf.ln(5)
        
    # AI Interpretation
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Executive Interpretation', align='L', ln=1)
    
    pdf.set_font('Helvetica', '', 12)
    pdf.multi_cell(0, 8, generate_interpretation(country, metrics, risk))
    
    output_tgt = OUTPUT_DIR / "country_report.pdf"
    pdf.output(str(output_tgt))
    return output_tgt
