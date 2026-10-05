"""Create small sample enterprise documents (fictional company) for demo and testing."""
from pathlib import Path
import random

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "sample_docs"
OUT.mkdir(exist_ok=True)

# ---------- PDF: HR policy ----------
styles = getSampleStyleSheet()
pages = [
    ("Nimbus Retail Pvt. Ltd. - Employee Handbook 2026", [
        "This handbook applies to all full-time employees of Nimbus Retail Pvt. Ltd., headquartered in Indore, Madhya Pradesh.",
        "It was approved by the HR Committee on 12 January 2026 and replaces the 2024 edition.",
        "Working hours are 9:30 AM to 6:00 PM, Monday to Friday. Store staff work in rotating shifts of 8 hours.",
    ]),
    ("Section 2: Leave Policy", [
        "Every full-time employee is entitled to 24 days of paid annual leave per calendar year.",
        "Up to 10 unused annual leave days can be carried forward to the next year. Days beyond 10 lapse on 31 March.",
        "Employees get 12 days of sick leave per year. A medical certificate is required for sick leave longer than 3 consecutive days.",
        "Maternity leave is 26 weeks of paid leave. Paternity leave is 10 working days, to be taken within 3 months of the child's birth.",
        "Leave requests must be submitted on the HR portal at least 7 days in advance, except for sick leave.",
    ]),
    ("Section 3: Travel and Expense Policy", [
        "Domestic travel by air is allowed only for journeys longer than 600 km. Otherwise employees travel by AC 2-tier train.",
        "The daily allowance is Rs. 2,500 in metro cities and Rs. 1,800 in other cities.",
        "Hotel stays are reimbursed up to Rs. 6,000 per night in metro cities and Rs. 4,000 per night elsewhere.",
        "Expense claims must be filed within 30 days of the trip with original bills. Claims above Rs. 50,000 need approval from the department head.",
    ]),
    ("Section 4: Remote Work and IT Security", [
        "Employees in corporate roles may work remotely up to 2 days per week with manager approval. Store staff are not eligible.",
        "Company laptops must use full-disk encryption and the corporate VPN when on public Wi-Fi.",
        "Passwords must be at least 12 characters and changed every 90 days. Multi-factor authentication is mandatory for email.",
        "Security incidents must be reported to it-security@nimbusretail.example within 24 hours.",
    ]),
]
doc = SimpleDocTemplate(str(OUT / "employee_handbook.pdf"), pagesize=A4)
story = []
for i, (title, paras) in enumerate(pages):
    story.append(Paragraph(title, styles["Heading1"]))
    for p in paras:
        story += [Paragraph(p, styles["BodyText"]), Spacer(1, 8)]
    if i < len(pages) - 1:
        story.append(PageBreak())
doc.build(story)

# ---------- TXT: product FAQ ----------
(OUT / "product_faq.txt").write_text("""Nimbus Retail - Customer FAQ (updated August 2026)

Q: What is the return policy?
A: Most products can be returned within 30 days of delivery if unused and in original packaging. Electronics have a 10-day return window. Innerwear and personalised items cannot be returned.

Q: How long does a refund take?
A: Refunds are processed within 5 to 7 business days after the returned item passes quality check. UPI refunds are usually faster, within 2 business days.

Q: Do you offer warranty on electronics?
A: All electronics carry a 1-year manufacturer warranty. The NimbusCare extended warranty adds 2 more years for 8% of the product price.

Q: What are the delivery charges?
A: Delivery is free on orders above Rs. 499. Orders below Rs. 499 have a delivery fee of Rs. 49. Express delivery in Indore, Bhopal and Ujjain costs Rs. 99 and arrives the next day.

Q: How do I contact customer support?
A: Call 1800-200-4455 (toll free, 9 AM to 9 PM, all days) or email support@nimbusretail.example.

Q: Which payment methods are accepted?
A: UPI, credit cards, debit cards, net banking, and cash on delivery for orders below Rs. 10,000.
""", encoding="utf-8")

# ---------- CSV: sales ----------
random.seed(7)
regions = ["North", "South", "East", "West", "Central"]
products = [("Smartphone X2", "Electronics", 18999), ("Bluetooth Earbuds", "Electronics", 2499),
            ("Cotton Kurta", "Apparel", 1299), ("Running Shoes", "Footwear", 3499),
            ("Pressure Cooker 5L", "Home & Kitchen", 2199), ("LED Desk Lamp", "Home & Kitchen", 899)]
rows = []
for i in range(1, 61):
    p, cat, price = random.choice(products)
    units = random.randint(5, 60)
    month = ["2026-04", "2026-05", "2026-06"][(i - 1) // 20]
    rows.append({"OrderID": f"ORD-{1000 + i}", "Month": month, "Region": random.choice(regions),
                 "Product": p, "Category": cat, "UnitPrice": price, "Units": units, "Revenue": price * units})
pd.DataFrame(rows).to_csv(OUT / "sales_q2_2026.csv", index=False)

# ---------- Excel: inventory (2 sheets) ----------
inv = pd.DataFrame([
    {"SKU": "SKU-101", "Product": "Smartphone X2", "Warehouse": "Indore", "Stock": 140, "ReorderLevel": 50, "Supplier": "TechSource India"},
    {"SKU": "SKU-102", "Product": "Bluetooth Earbuds", "Warehouse": "Indore", "Stock": 35, "ReorderLevel": 80, "Supplier": "SoundWave Pvt Ltd"},
    {"SKU": "SKU-103", "Product": "Cotton Kurta", "Warehouse": "Bhopal", "Stock": 420, "ReorderLevel": 100, "Supplier": "Malwa Textiles"},
    {"SKU": "SKU-104", "Product": "Running Shoes", "Warehouse": "Bhopal", "Stock": 60, "ReorderLevel": 75, "Supplier": "StrideWorks"},
    {"SKU": "SKU-105", "Product": "Pressure Cooker 5L", "Warehouse": "Ujjain", "Stock": 210, "ReorderLevel": 60, "Supplier": "HomeChef Appliances"},
    {"SKU": "SKU-106", "Product": "LED Desk Lamp", "Warehouse": "Ujjain", "Stock": 18, "ReorderLevel": 40, "Supplier": "BrightHome"},
])
suppliers = pd.DataFrame([
    {"Supplier": "TechSource India", "City": "Bengaluru", "LeadTimeDays": 7, "Rating": 4.6},
    {"Supplier": "SoundWave Pvt Ltd", "City": "Noida", "LeadTimeDays": 5, "Rating": 4.2},
    {"Supplier": "Malwa Textiles", "City": "Indore", "LeadTimeDays": 3, "Rating": 4.8},
    {"Supplier": "StrideWorks", "City": "Agra", "LeadTimeDays": 10, "Rating": 3.9},
    {"Supplier": "HomeChef Appliances", "City": "Pune", "LeadTimeDays": 6, "Rating": 4.4},
    {"Supplier": "BrightHome", "City": "Ahmedabad", "LeadTimeDays": 4, "Rating": 4.1},
])
with pd.ExcelWriter(OUT / "inventory.xlsx") as xw:
    inv.to_excel(xw, sheet_name="Stock", index=False)
    suppliers.to_excel(xw, sheet_name="Suppliers", index=False)

print("Sample documents written to", OUT)
