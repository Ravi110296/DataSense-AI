import os
import csv
from datetime import datetime, timedelta

def create_sample_data():
    os.makedirs("sample_data", exist_ok=True)
    
    # 1. Generate sales_data.csv
    # 52 realistic transactions across 4 categories, 4 regions, and 6 months (Jan-Jun 2024)
    transactions = [
        # (order_id, date, customer_id, region, product, category, quantity, unit_price)
        # Electronics
        ("ORD-1001", "2024-01-12", "CUST-501", "North", "ProBook Ultra 15", "Electronics", 3, 1200.0),
        ("ORD-1002", "2024-01-18", "CUST-502", "West", "SmartWatch Pro", "Electronics", 5, 250.0),
        ("ORD-1003", "2024-01-25", "CUST-503", "East", "AcousticMax Headphones", "Electronics", 4, 150.0),
        ("ORD-1004", "2024-02-03", "CUST-504", "South", "VisionView 4K Monitor", "Electronics", 2, 400.0),
        ("ORD-1005", "2024-02-14", "CUST-505", "North", "ProBook Ultra 15", "Electronics", 4, 1200.0),
        ("ORD-1006", "2024-02-22", "CUST-506", "West", "ProBook Ultra 15", "Electronics", 2, 1200.0),
        ("ORD-1007", "2024-03-05", "CUST-507", "East", "SmartWatch Pro", "Electronics", 6, 250.0),
        ("ORD-1008", "2024-03-19", "CUST-508", "North", "AcousticMax Headphones", "Electronics", 8, 150.0),
        ("ORD-1009", "2024-04-02", "CUST-509", "South", "ProBook Ultra 15", "Electronics", 3, 1200.0),
        ("ORD-1010", "2024-04-18", "CUST-510", "West", "VisionView 4K Monitor", "Electronics", 5, 400.0),
        ("ORD-1011", "2024-05-09", "CUST-511", "North", "SmartWatch Pro", "Electronics", 4, 250.0),
        ("ORD-1012", "2024-05-24", "CUST-512", "East", "ProBook Ultra 15", "Electronics", 5, 1200.0),
        ("ORD-1013", "2024-06-11", "CUST-513", "West", "ProBook Ultra 15", "Electronics", 3, 1200.0),
        ("ORD-1014", "2024-06-27", "CUST-514", "South", "AcousticMax Headphones", "Electronics", 5, 150.0),

        # Home & Kitchen
        ("ORD-1015", "2024-01-08", "CUST-515", "South", "BrewMaster Espresso Maker", "Home & Kitchen", 2, 300.0),
        ("ORD-1016", "2024-01-22", "CUST-516", "North", "AeroFry Digital Air Fryer", "Home & Kitchen", 4, 120.0),
        ("ORD-1017", "2024-02-07", "CUST-517", "West", "ChefElite Knife Set", "Home & Kitchen", 5, 80.0),
        ("ORD-1018", "2024-02-19", "CUST-518", "East", "RobotVac Smart Cleaner", "Home & Kitchen", 2, 350.0),
        ("ORD-1019", "2024-03-02", "CUST-519", "North", "BrewMaster Espresso Maker", "Home & Kitchen", 4, 300.0),
        ("ORD-1020", "2024-03-15", "CUST-520", "West", "AeroFry Digital Air Fryer", "Home & Kitchen", 6, 120.0),
        ("ORD-1021", "2024-03-29", "CUST-521", "South", "RobotVac Smart Cleaner", "Home & Kitchen", 3, 350.0),
        ("ORD-1022", "2024-04-10", "CUST-522", "East", "BrewMaster Espresso Maker", "Home & Kitchen", 3, 300.0),
        ("ORD-1023", "2024-04-26", "CUST-523", "North", "ChefElite Knife Set", "Home & Kitchen", 7, 80.0),
        ("ORD-1024", "2024-05-14", "CUST-524", "West", "RobotVac Smart Cleaner", "Home & Kitchen", 4, 350.0),
        ("ORD-1025", "2024-05-29", "CUST-525", "South", "BrewMaster Espresso Maker", "Home & Kitchen", 2, 300.0),
        ("ORD-1026", "2024-06-08", "CUST-526", "East", "AeroFry Digital Air Fryer", "Home & Kitchen", 5, 120.0),
        ("ORD-1027", "2024-06-20", "CUST-527", "North", "RobotVac Smart Cleaner", "Home & Kitchen", 3, 350.0),

        # Furniture
        ("ORD-1028", "2024-01-14", "CUST-528", "West", "ErgoComfort Office Chair", "Furniture", 4, 220.0),
        ("ORD-1029", "2024-01-29", "CUST-529", "North", "Nordic Wood Dining Table", "Furniture", 1, 650.0),
        ("ORD-1030", "2024-02-11", "CUST-530", "East", "LuxRest Memory Foam Mattress", "Furniture", 2, 800.0),
        ("ORD-1031", "2024-02-27", "CUST-531", "South", "Modular Bookshelf", "Furniture", 3, 180.0),
        ("ORD-1032", "2024-03-12", "CUST-532", "West", "ErgoComfort Office Chair", "Furniture", 5, 220.0),
        ("ORD-1033", "2024-03-24", "CUST-533", "North", "LuxRest Memory Foam Mattress", "Furniture", 2, 800.0),
        ("ORD-1034", "2024-04-05", "CUST-534", "East", "Nordic Wood Dining Table", "Furniture", 2, 650.0),
        ("ORD-1035", "2024-04-21", "CUST-535", "South", "ErgoComfort Office Chair", "Furniture", 3, 220.0),
        ("ORD-1036", "2024-05-04", "CUST-536", "West", "Modular Bookshelf", "Furniture", 4, 180.0),
        ("ORD-1037", "2024-05-18", "CUST-537", "North", "ErgoComfort Office Chair", "Furniture", 6, 220.0),
        ("ORD-1038", "2024-06-03", "CUST-538", "East", "LuxRest Memory Foam Mattress", "Furniture", 3, 800.0),
        ("ORD-1039", "2024-06-16", "CUST-539", "South", "Nordic Wood Dining Table", "Furniture", 1, 650.0),

        # Sports & Fitness
        ("ORD-1040", "2024-01-07", "CUST-540", "North", "ApexFold Treadmill", "Sports & Fitness", 1, 950.0),
        ("ORD-1041", "2024-01-20", "CUST-541", "South", "FlexiBell Adjustable Dumbbells", "Sports & Fitness", 3, 200.0),
        ("ORD-1042", "2024-02-05", "CUST-542", "East", "YogaZen Eco Mat", "Sports & Fitness", 10, 50.0),
        ("ORD-1043", "2024-02-18", "CUST-543", "West", "PulseFit Smart Bike", "Sports & Fitness", 2, 700.0),
        ("ORD-1044", "2024-03-08", "CUST-544", "North", "FlexiBell Adjustable Dumbbells", "Sports & Fitness", 4, 200.0),
        ("ORD-1045", "2024-03-22", "CUST-545", "South", "ApexFold Treadmill", "Sports & Fitness", 2, 950.0),
        ("ORD-1046", "2024-04-14", "CUST-546", "East", "PulseFit Smart Bike", "Sports & Fitness", 1, 700.0),
        ("ORD-1047", "2024-04-28", "CUST-547", "West", "YogaZen Eco Mat", "Sports & Fitness", 8, 50.0),
        ("ORD-1048", "2024-05-11", "CUST-548", "North", "ApexFold Treadmill", "Sports & Fitness", 2, 950.0),
        ("ORD-1049", "2024-05-26", "CUST-549", "East", "FlexiBell Adjustable Dumbbells", "Sports & Fitness", 5, 200.0),
        ("ORD-1050", "2024-06-05", "CUST-550", "South", "PulseFit Smart Bike", "Sports & Fitness", 2, 700.0),
        ("ORD-1051", "2024-06-22", "CUST-551", "West", "ApexFold Treadmill", "Sports & Fitness", 1, 950.0),
        ("ORD-1052", "2024-06-29", "CUST-552", "North", "YogaZen Eco Mat", "Sports & Fitness", 12, 50.0),
    ]

    csv_path = os.path.join("sample_data", "sales_data.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "order_date", "customer_id", "region", "product", "category", "quantity", "unit_price", "revenue"])
        for row in transactions:
            order_id, date, cust, region, product, category, qty, price = row
            revenue = round(qty * price, 2)
            writer.writerow([order_id, date, cust, region, product, category, qty, price, revenue])
            
    print(f"Created {csv_path} with {len(transactions)} rows.")

    # 2. Generate company_handbook.pdf using reportlab
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib import colors

    pdf_path = os.path.join("sample_data", "company_handbook.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=12
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1E40AF'),
        spaceBefore=14,
        spaceAfter=6
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#1F2937'),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#374151'),
        spaceAfter=6
    )

    story = []
    
    # Title & Disclaimer
    story.append(Paragraph("Apex Retail Solutions — Operations & Policy Handbook", title_style))
    story.append(Paragraph("<b>Notice:</b> This document contains fictional reference policies and catalog information for Apex Retail Solutions. Created exclusively for demonstration and testing of the Intelligent Data Analyst Agent.", body_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#93C5FD'), spaceBefore=8, spaceAfter=14))

    # Section 1: Product Catalog Overview
    story.append(Paragraph("1. Product Catalog Overview", h1_style))
    story.append(Paragraph("Apex Retail Solutions supplies premium products across four core retail categories: Electronics, Home & Kitchen, Furniture, and Sports & Fitness.", body_style))
    
    story.append(Paragraph("<b>Electronics:</b>", h2_style))
    story.append(Paragraph("• <b>ProBook Ultra 15:</b> High-performance 15.6-inch laptop with ultra-fast multi-core processor, 32GB RAM, and 1TB NVMe SSD. Designed for software engineering and creative professionals.<br/>"
                           "• <b>SmartWatch Pro:</b> Advanced health and fitness smartwatch featuring continuous ECG, optical heart rate sensor, GPS navigation, and 7-day battery life.<br/>"
                           "• <b>AcousticMax Headphones:</b> Wireless active noise-cancelling headphones featuring 40mm studio drivers and 35-hour playback time.<br/>"
                           "• <b>VisionView 4K Monitor:</b> 32-inch UHD HDR400 IPS professional monitor with 99% sRGB color gamut and USB-C power delivery.", body_style))

    story.append(Paragraph("<b>Home & Kitchen:</b>", h2_style))
    story.append(Paragraph("• <b>BrewMaster Espresso Maker:</b> 15-bar Italian pump espresso machine with integrated precision conical burr grinder and steam wand.<br/>"
                           "• <b>AeroFry Digital Air Fryer:</b> 6-quart digital air fryer with 8 one-touch cooking presets and rapid air vortex technology.<br/>"
                           "• <b>ChefElite Knife Set:</b> 8-piece Japanese high-carbon stainless steel chef cutlery set with ergonomic Pakkawood handles.<br/>"
                           "• <b>RobotVac Smart Cleaner:</b> Autonomous robotic vacuum cleaner with LiDAR mapping, 4000Pa suction power, and automatic dustbin emptying.", body_style))

    story.append(Paragraph("<b>Furniture:</b>", h2_style))
    story.append(Paragraph("• <b>ErgoComfort Office Chair:</b> Ergonomic mesh task chair with adaptive lumbar support, 4D adjustable armrests, and synchro-tilt mechanism.<br/>"
                           "• <b>Nordic Wood Dining Table:</b> Solid oak rectangular dining table seating 6-8 people with natural matte protective oil finish.<br/>"
                           "• <b>LuxRest Memory Foam Mattress:</b> 12-inch multi-layer cooling gel memory foam mattress with breathable bamboo cover.<br/>"
                           "• <b>Modular Bookshelf:</b> 5-tier customizable Scandinavian open shelving unit crafted from reinforced steel and engineered wood.", body_style))

    story.append(Paragraph("<b>Sports & Fitness:</b>", h2_style))
    story.append(Paragraph("• <b>ApexFold Treadmill:</b> Heavy-duty motorized folding treadmill with 3.5 CHP motor, interactive touchscreen, and hydraulic soft-drop system.<br/>"
                           "• <b>FlexiBell Adjustable Dumbbells:</b> Quick-dial adjustable dumbbell pair replacing 15 sets of weights (5 to 52.5 lbs per dumbbell).<br/>"
                           "• <b>YogaZen Eco Mat:</b> 6mm non-slip biodegradable natural tree rubber yoga mat with alignment guide lines.<br/>"
                           "• <b>PulseFit Smart Bike:</b> Stationary magnetic resistance studio cycling bike with Bluetooth cadence sensing and power meter.", body_style))

    story.append(Spacer(1, 10))

    # Section 2: Category Return and Refund Policies
    story.append(Paragraph("2. Return and Refund Policies", h1_style))
    story.append(Paragraph("Apex Retail Solutions provides customer-friendly return options tailored to the nature of each product line:", body_style))
    
    story.append(Paragraph("<b>Electronics Return Policy:</b>", h2_style))
    story.append(Paragraph("Electronics items, including the ProBook Ultra 15, SmartWatch Pro, AcousticMax Headphones, and VisionView 4K Monitor, carry a <b>30-day return window</b> starting from delivery date. Items must be in original condition with all packaging, manuals, power adapters, and accessories intact. If opened without technical defect, a <b>15% restocking fee</b> applies. Defective units are eligible for replacement or 100% refund without fee.", body_style))

    story.append(Paragraph("<b>Home & Kitchen Return Policy:</b>", h2_style))
    story.append(Paragraph("Home and kitchen appliances carry an extended <b>45-day return window</b>. Returned appliances (e.g. BrewMaster Espresso Maker, AeroFry Digital Air Fryer, RobotVac Smart Cleaner) must be thoroughly cleaned, drained of any liquid or food residue, and repacked in original boxing. Knife sets must remain complete without chipped blades.", body_style))

    story.append(Paragraph("<b>Furniture Return Policy (Special Conditions):</b>", h2_style))
    story.append(Paragraph("Furniture pieces feature a <b>30-day home trial period</b>. Special condition: Customers are responsible for scheduling freight carrier pickup. A <b>flat $50 return shipping fee</b> is deducted from the refund unless the furniture was damaged during transit. Mattresses (LuxRest) must have been used with a mattress protector and show no stains or tears.", body_style))

    story.append(Paragraph("<b>Sports & Fitness Return Policy:</b>", h2_style))
    story.append(Paragraph("Sports gear carries a <b>30-day return policy</b>. Equipment weighing over 50 lbs (such as the ApexFold Treadmill and PulseFit Smart Bike) requires a certified technician inspection during freight collection to confirm the motor and frame are intact.", body_style))

    story.append(Spacer(1, 10))

    # Section 3: Warranty Terms
    story.append(Paragraph("3. Warranty Terms and Coverage", h1_style))
    story.append(Paragraph("• <b>Electronics Warranty:</b> All electronic devices include a <b>2-year limited manufacturer warranty</b> covering internal hardware components, motherboard, and display panels against defects. Rechargeable internal lithium batteries are covered for <b>1 year</b>. Accidental drops and water ingress are excluded unless customer added ProCare Protection at checkout.<br/>"
                           "• <b>Home & Kitchen Warranty:</b> Includes a <b>1-year limited warranty</b> covering internal motor mechanisms, digital boards, and electrical heating elements.<br/>"
                           "• <b>Furniture Warranty:</b> Features an extended <b>5-year structural warranty</b> covering wooden frames, welding joints, and pneumatic cylinders on office chairs.<br/>"
                           "• <b>Sports & Fitness Warranty:</b> Includes a <b>3-year structural frame warranty</b> and a <b>1-year warranty on electronic consoles and moving drive belts</b>.", body_style))

    story.append(Spacer(1, 10))

    # Section 4: Discount & Pricing Guidelines
    story.append(Paragraph("4. Commercial Discounts and Special Pricing", h1_style))
    story.append(Paragraph("• <b>Home & Kitchen Bulk Discount:</b> Orders of Home & Kitchen products exceeding a total invoice value of <b>$1,000</b> automatically receive a <b>10% volume discount</b> applied at checkout.<br/>"
                           "• <b>West Region Furniture Commercial Incentive:</b> Commercial orders for Furniture items shipped to the <b>West region</b> qualify for an exclusive <b>5% regional business incentive discount</b>.<br/>"
                           "• <b>Electronics Trade-In Credit:</b> Customers purchasing the <b>ProBook Ultra 15</b> can trade in any qualifying used laptop for up to <b>$150 trade-in credit</b> toward the purchase price.<br/>"
                           "• <b>Seasonal Clearance:</b> End-of-season clearance items marked with an asterisk are final sale and non-refundable.", body_style))

    story.append(Spacer(1, 10))

    # Section 5: Regional Shipping & Support
    story.append(Paragraph("5. Regional Operations and Logistics", h1_style))
    story.append(Paragraph("Apex Retail Solutions operates regional distribution hubs across North (Chicago), South (Atlanta), East (Newark), and West (Reno). Standard ground transit requires 2 to 4 business days. White-glove indoor delivery and assembly is offered exclusively in the North and West regions for heavy Furniture and Fitness machinery.", body_style))

    doc.build(story)
    print(f"Created {pdf_path} successfully.")

if __name__ == "__main__":
    create_sample_data()
