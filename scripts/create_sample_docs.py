#!/usr/bin/env python3
"""
Script to generate sample legal PDF documents for testing the Document Research Agent.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.platypus.flowables import HRFlowable
import random

# Ensure the test_data directory exists
os.makedirs("test_data", exist_ok=True)

# Sample legal document templates
TEMPLATES = {
    "employment_contract": {
        "title": "Employment Agreement",
        "sections": [
            ("Parties", "This Employment Agreement (the \"Agreement\") is made and entered into as of {date}, by and between {employer_name} (the \"Employer\"), and {employee_name} (the \"Employee\")."),
            ("Terms of Employment", "The Employer hereby employs the Employee, and the Employee hereby accepts employment with the Employer, on the terms and conditions set forth in this Agreement."),
            ("Position and Duties", "During the Employment Term, the Employee shall serve as {position} of the Employer, with such duties, authority and responsibilities as are normally associated with and appropriate for such position."),
            ("Compensation", "During the Employment Term, the Employer shall pay the Employee a base salary at an annual rate of ${salary}, payable in accordance with the Employer's normal payroll practices."),
            ("Benefits", "During the Employment Term, the Employee shall be eligible to participate in all employee benefit plans, programs and arrangements that are generally made available to other employees of the Employer."),
            ("Term and Termination", "The initial term of the Employee's employment under this Agreement shall be for a period of {term} years, commencing on {start_date} (the \"Employment Term\")."),
            ("Non-Disclosure", "The Employee acknowledges that during the Employment Term, the Employee will have access to and become acquainted with various trade secrets and other confidential information which are owned by the Employer."),
            ("Governing Law", "This Agreement shall be governed by and construed in accordance with the laws of the State of {state}, without giving effect to any choice of law or conflict of law provisions."),
            ("Signatures", "IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the date first written above.")
        ]
    },
    "service_agreement": {
        "title": "Service Agreement",
        "sections": [
            ("Parties", "This Service Agreement (the \"Agreement\") is entered into as of {date} by and between {client_name} (the \"Client\") and {provider_name} (the \"Service Provider\")."),
            ("Services", "The Service Provider agrees to provide the following services to the Client: {services}."),
            ("Term", "The term of this Agreement shall commence on {start_date} and shall continue until {end_date}, unless earlier terminated in accordance with this Agreement."),
            ("Compensation", "In consideration for the Services provided by the Service Provider, the Client shall pay the Service Provider a fee of ${fee}, payable as follows: {payment_terms}."),
            ("Independent Contractor", "The Service Provider is an independent contractor and not an employee of the Client. The Service Provider shall be responsible for all taxes, insurance, and other obligations related to the Services provided."),
            ("Confidentiality", "During the term of this Agreement and thereafter, the Service Provider shall maintain the confidentiality of any proprietary or confidential information of the Client."),
            ("Termination", "Either party may terminate this Agreement upon {notice_period} days written notice to the other party. Upon termination, the Client shall pay the Service Provider for all Services performed up to the date of termination."),
            ("Limitation of Liability", "In no event shall either party be liable to the other for any indirect, special, incidental, or consequential damages arising out of or related to this Agreement."),
            ("Governing Law", "This Agreement shall be governed by and construed in accordance with the laws of the State of {state}, without giving effect to any choice of law or conflict of law provisions."),
            ("Signatures", "IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the date first written above.")
        ]
    },
    "non_disclosure_agreement": {
        "title": "Non-Disclosure Agreement",
        "sections": [
            ("Parties", "This Non-Disclosure Agreement (the \"Agreement\") is entered into as of {date} by and between {disclosing_party} (the \"Disclosing Party\") and {receiving_party} (the \"Receiving Party\")."),
            ("Purpose", "The parties wish to explore a business opportunity of mutual interest, and in connection with this opportunity, the Disclosing Party may disclose to the Receiving Party certain confidential information."),
            ("Definition of Confidential Information", "\"Confidential Information\" means any information disclosed by the Disclosing Party to the Receiving Party, either directly or indirectly, in writing, orally or by inspection of tangible objects, that is designated as \"Confidential\", \"Proprietary\" or some similar designation."),
            ("Non-Disclosure Obligations", "The Receiving Party shall hold the Confidential Information in strict confidence and shall not disclose such Confidential Information to any third party."),
            ("Term", "The obligations of the Receiving Party under this Agreement shall survive termination of any business relationship between the parties and shall continue for a period of {term} years from the date of disclosure of the Confidential Information."),
            ("Return of Materials", "All documents and other tangible objects containing or representing Confidential Information which have been disclosed by the Disclosing Party to the Receiving Party, shall be and remain the property of the Disclosing Party and shall be promptly returned to the Disclosing Party upon the Disclosing Party's written request."),
            ("No Rights Granted", "Nothing in this Agreement shall be construed as granting any rights to the Receiving Party under any patent, copyright, or other intellectual property right of the Disclosing Party."),
            ("Governing Law", "This Agreement shall be governed by and construed in accordance with the laws of the State of {state}, without giving effect to any choice of law or conflict of law provisions."),
            ("Signatures", "IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the date first written above.")
        ]
    },
    "lease_agreement": {
        "title": "Commercial Lease Agreement",
        "sections": [
            ("Parties", "This Commercial Lease Agreement (the \"Lease\") is made and entered into as of {date}, by and between {landlord_name} (the \"Landlord\"), and {tenant_name} (the \"Tenant\")."),
            ("Premises", "The Landlord hereby leases to the Tenant, and the Tenant hereby leases from the Landlord, the premises located at {property_address} (the \"Premises\")."),
            ("Term", "The term of this Lease shall be for a period of {term} years, commencing on {start_date} and ending on {end_date}, unless earlier terminated in accordance with this Lease."),
            ("Rent", "The Tenant shall pay to the Landlord as rent for the Premises the sum of ${rent} per month, payable in advance on the first day of each month during the term of this Lease."),
            ("Security Deposit", "Upon execution of this Lease, the Tenant shall deposit with the Landlord the sum of ${deposit} as a security deposit for the faithful performance by the Tenant of the terms of this Lease."),
            ("Use of Premises", "The Tenant shall use the Premises solely for the purpose of {use} and for no other purpose without the prior written consent of the Landlord."),
            ("Maintenance and Repairs", "The Tenant shall, at its own expense, maintain the Premises in good condition and repair during the term of this Lease. The Landlord shall be responsible for structural repairs to the building."),
            ("Insurance", "The Tenant shall, at its own expense, maintain a policy of commercial general liability insurance with respect to the Premises with minimum limits of liability of ${insurance_amount} per occurrence."),
            ("Default", "If the Tenant fails to pay rent when due or fails to perform any other obligation under this Lease, the Landlord may terminate this Lease and retake possession of the Premises."),
            ("Governing Law", "This Lease shall be governed by and construed in accordance with the laws of the State of {state}, without giving effect to any choice of law or conflict of law provisions."),
            ("Signatures", "IN WITNESS WHEREOF, the parties hereto have executed this Lease as of the date first written above.")
        ]
    },
    "purchase_agreement": {
        "title": "Purchase and Sale Agreement",
        "sections": [
            ("Parties", "This Purchase and Sale Agreement (the \"Agreement\") is made and entered into as of {date}, by and between {seller_name} (the \"Seller\"), and {buyer_name} (the \"Buyer\")."),
            ("Property", "The Seller agrees to sell to the Buyer, and the Buyer agrees to purchase from the Seller, the property located at {property_address} (the \"Property\")."),
            ("Purchase Price", "The purchase price for the Property shall be ${purchase_price} (the \"Purchase Price\")."),
            ("Deposit", "Upon execution of this Agreement, the Buyer shall deposit with the Escrow Agent the sum of ${deposit} (the \"Deposit\") as earnest money."),
            ("Closing", "The closing of the transaction contemplated by this Agreement (the \"Closing\") shall take place on {closing_date} at the offices of {closing_location}."),
            ("Title", "At the Closing, the Seller shall convey good and marketable title to the Property to the Buyer by warranty deed, free and clear of all liens and encumbrances, except for those permitted by this Agreement."),
            ("Inspections", "The Buyer shall have the right, at the Buyer's expense, to conduct inspections of the Property during the period from the date of this Agreement until {inspection_deadline}."),
            ("Representations and Warranties", "The Seller represents and warrants to the Buyer that the Seller is the sole owner of the Property and has the full right, power and authority to sell the Property to the Buyer."),
            ("Default", "If the Buyer defaults in the performance of this Agreement, the Seller may, as its sole and exclusive remedy, terminate this Agreement and retain the Deposit as liquidated damages."),
            ("Governing Law", "This Agreement shall be governed by and construed in accordance with the laws of the State of {state}, without giving effect to any choice of law or conflict of law provisions."),
            ("Signatures", "IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the date first written above.")
        ]
    }
}

# Sample data to fill in placeholders
SAMPLE_DATA = {
    "dates": ["January 1, 2023", "February 15, 2023", "March 10, 2023", "April 5, 2023", "May 20, 2023"],
    "employer_names": ["Acme Corporation", "Globex Inc.", "Stark Industries", "Wayne Enterprises", "Umbrella Corporation"],
    "employee_names": ["John Smith", "Jane Doe", "Robert Johnson", "Sarah Williams", "Michael Brown"],
    "positions": ["Chief Executive Officer", "Chief Financial Officer", "Chief Technology Officer", "Director of Marketing", "Vice President of Sales"],
    "salaries": ["120,000", "95,000", "110,000", "85,000", "100,000"],
    "terms": ["3", "2", "5", "1", "4"],
    "start_dates": ["January 15, 2023", "March 1, 2023", "April 15, 2023", "June 1, 2023", "July 15, 2023"],
    "states": ["California", "New York", "Texas", "Florida", "Illinois"],
    "client_names": ["XYZ Company", "ABC Corporation", "123 Industries", "Tech Solutions Inc.", "Global Enterprises"],
    "provider_names": ["Service Pro LLC", "ConsultCo Inc.", "Expert Services Group", "Professional Solutions", "Quality Providers Inc."],
    "services": ["IT consulting", "marketing services", "legal services", "accounting services", "design services"],
    "end_dates": ["December 31, 2023", "June 30, 2024", "September 30, 2023", "March 31, 2024", "August 15, 2023"],
    "fees": ["5,000", "10,000", "15,000", "8,000", "12,000"],
    "payment_terms": ["monthly installments", "upon completion", "50% upfront, 50% upon completion", "quarterly installments", "weekly installments"],
    "notice_periods": ["30", "60", "15", "45", "90"],
    "disclosing_parties": ["Acme Corporation", "Globex Inc.", "Stark Industries", "Wayne Enterprises", "Umbrella Corporation"],
    "receiving_parties": ["XYZ Company", "ABC Corporation", "123 Industries", "Tech Solutions Inc.", "Global Enterprises"],
    "landlord_names": ["Property Management Inc.", "Real Estate Holdings LLC", "Commercial Properties Group", "Office Space Solutions", "Urban Developments Inc."],
    "tenant_names": ["Retail Store Inc.", "Restaurant Group LLC", "Medical Practice Associates", "Law Firm Partners", "Tech Startup Co."],
    "property_addresses": ["123 Main Street, Suite 100, Anytown, CA 94567", "456 Business Avenue, Building B, Metropolis, NY 10001", "789 Commerce Parkway, Unit 200, Bigcity, TX 75001", "321 Corporate Drive, Office 300, Sunshine, FL 33000", "555 Industrial Boulevard, Suite 400, Windy City, IL 60000"],
    "rents": ["5,000", "8,000", "6,500", "4,000", "7,500"],
    "deposits": ["10,000", "16,000", "13,000", "8,000", "15,000"],
    "uses": ["retail store", "restaurant", "medical office", "law office", "technology office"],
    "insurance_amounts": ["1,000,000", "2,000,000", "1,500,000", "3,000,000", "2,500,000"],
    "seller_names": ["Property Seller LLC", "Estate of John Doe", "Commercial Holdings Inc.", "Real Estate Investments Group", "Property Owner Trust"],
    "buyer_names": ["Property Buyer Inc.", "Investor Group LLC", "Commercial Acquisitions Corp.", "Real Estate Development Co.", "Property Investment Trust"],
    "purchase_prices": ["1,000,000", "1,500,000", "2,000,000", "850,000", "1,250,000"],
    "closing_dates": ["July 1, 2023", "August 15, 2023", "September 30, 2023", "October 15, 2023", "November 30, 2023"],
    "closing_locations": ["First National Title Company", "Trusted Escrow Services", "Legal Title Corporation", "Security Title Agency", "Professional Escrow Group"],
    "inspection_deadlines": ["June 15, 2023", "July 31, 2023", "September 15, 2023", "October 1, 2023", "November 15, 2023"]
}

def random_sample(key):
    """Get a random sample from the sample data for the given key."""
    return random.choice(SAMPLE_DATA[key])

def fill_template(template_type):
    """Fill a template with random sample data."""
    template = TEMPLATES[template_type]
    title = template["title"]
    sections = []
    
    # Choose random values for consistency across the document
    random_date = random_sample("dates")
    random_state = random_sample("states")
    
    # Create a dictionary to store specific placeholder replacements
    placeholder_values = {
        "date": random_date,
        "state": random_state,
        "salary": random_sample("salaries"),
        "term": random_sample("terms"),
        "start_date": random_sample("start_dates"),
        "employer_name": random_sample("employer_names"),
        "employee_name": random_sample("employee_names"),
        "position": random_sample("positions"),
        "client_name": random_sample("client_names"),
        "provider_name": random_sample("provider_names"),
        "services": random_sample("services"),
        "end_date": random_sample("end_dates"),
        "fee": random_sample("fees"),
        "payment_terms": random_sample("payment_terms"),
        "notice_period": random_sample("notice_periods"),
        "disclosing_party": random_sample("disclosing_parties"),
        "receiving_party": random_sample("receiving_parties"),
        "landlord_name": random_sample("landlord_names"),
        "tenant_name": random_sample("tenant_names"),
        "property_address": random_sample("property_addresses"),
        "rent": random_sample("rents"),
        "deposit": random_sample("deposits"),
        "use": random_sample("uses"),
        "insurance_amount": random_sample("insurance_amounts"),
        "seller_name": random_sample("seller_names"),
        "buyer_name": random_sample("buyer_names"),
        "purchase_price": random_sample("purchase_prices"),
        "closing_date": random_sample("closing_dates"),
        "closing_location": random_sample("closing_locations"),
        "inspection_deadline": random_sample("inspection_deadlines"),
    }
    
    for section_title, section_content in template["sections"]:
        # Replace placeholders with values from our dictionary
        content = section_content
        
        for placeholder, value in placeholder_values.items():
            content = content.replace(f"{{{placeholder}}}", value)
        
        sections.append((section_title, content))
    
    return title, sections

def create_pdf(filename, title, sections):
    """Create a PDF document with the given title and sections."""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create a custom style for the title
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=1,  # Center alignment
        spaceAfter=20
    )
    
    # Create a custom style for section titles
    section_title_style = ParagraphStyle(
        name='SectionTitleStyle',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=10
    )
    
    # Create a custom style for section content
    section_content_style = ParagraphStyle(
        name='SectionContentStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=15
    )
    
    # Create document content
    content = []
    
    # Add title
    content.append(Paragraph(title, title_style))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=20))
    content.append(Spacer(1, 20))
    
    # Add sections
    for section_title, section_content in sections:
        content.append(Paragraph(section_title, section_title_style))
        content.append(Paragraph(section_content, section_content_style))
        content.append(Spacer(1, 10))
    
    # Build the PDF
    doc.build(content)

def main():
    """Generate sample legal documents."""
    # Template types to generate
    template_types = list(TEMPLATES.keys())
    
    # Generate 10 documents - 2 of each type
    for i in range(10):
        template_type = template_types[i % len(template_types)]
        title, sections = fill_template(template_type)
        
        # Create a unique filename
        filename = f"test_data/legal_document_{i+1:02d}_{template_type}.pdf"
        
        # Create the PDF
        create_pdf(filename, title, sections)
        print(f"Created {filename}")

if __name__ == "__main__":
    main() 