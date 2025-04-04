# Document Research Test Collection

This directory contains legal documents for testing the Document Research Agent.

## Generated Documents

1. **legal_document_01_employment_contract.pdf** - Employment agreement between an employer and employee, including compensation, duties, and confidentiality provisions.
2. **legal_document_02_service_agreement.pdf** - Service agreement between a client and service provider outlining services, compensation, and terms.
3. **legal_document_03_non_disclosure_agreement.pdf** - Non-disclosure agreement protecting confidential information shared between parties.
4. **legal_document_04_lease_agreement.pdf** - Commercial lease agreement for renting property, including rental terms and tenant obligations.
5. **legal_document_05_purchase_agreement.pdf** - Property purchase and sale agreement between a buyer and seller.
6. **legal_document_06_employment_contract.pdf** - Another employment agreement with different terms and parties.
7. **legal_document_07_service_agreement.pdf** - Another service agreement with different terms and parties.
8. **legal_document_08_non_disclosure_agreement.pdf** - Another non-disclosure agreement with different terms and parties.
9. **legal_document_09_lease_agreement.pdf** - Another commercial lease agreement with different terms and parties.
10. **legal_document_10_purchase_agreement.pdf** - Another property purchase agreement with different terms and parties.

## External Documents

11. **sample_contract_shuttle.pdf** - Sample shuttle service contract from Santa Cruz County Regional Transportation Commission.
12. **sample_nda.pdf** - Model confidentiality agreement from the Electronic Frontier Foundation.
13. **sample_employment_agreement.pdf** - HTML format employment agreement (may not be valid PDF).
14. **sample_lease_agreement.pdf** - HUD lease agreement (may be incomplete download).

## Usage with Document Research Agent

These documents can be used to test various query types with the Document Research Agent. For example:

- Employment terms and conditions
- Property lease requirements and obligations
- Confidentiality requirements in agreements
- Purchase and sale agreement stipulations
- Service agreement terms and conditions

Use the following format when querying specific documents:

```bash
python -m src.main "What are the confidentiality provisions in the agreements?" --filenames test_data/legal_document_03_non_disclosure_agreement.pdf test_data/sample_nda.pdf
``` 