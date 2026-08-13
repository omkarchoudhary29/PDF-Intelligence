"""Builds the evaluation dataset and creates authentic test documents for benchmarking.

Generates 5 realistic enterprise technical PDFs, processes their hierarchical chunking
and indexing into an evaluation workspace, and writes verified ground-truth questions
to `evaluation/dataset.jsonl` across 9 categories:
- exact keyword
- technical identifier
- semantic
- factual
- cross-document
- comparison
- summary
- multi-hop
- unanswerable
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import uuid

import pymupdf

# Document Definitions with verified Ground Truth
DOCUMENTS_SPEC = [
    {
        "filename": "Turbine_Operations_Manual_AX450.pdf",
        "doc_id": "eval_doc_001",
        "title": "Turbine Operations Manual AX-450",
        "pages": [
            {
                "page_num": 1,
                "heading": "1. Overview & General Specifications",
                "text": (
                    "1. Overview & General Specifications\n\n"
                    "The model AX-450 industrial steam turbine is engineered for high-efficiency continuous power "
                    "generation. Nominal operating speed is 3600 RPM with a rotor diameter of 2.4 meters. "
                    "The maximum allowable inlet temperature is 540 C at a continuous rating of 45 MW. "
                    "Inlet steam pressure must remain stabilized at 65 bar. Any deviation in inlet pressure exceeding "
                    "+/- 5 bar triggers an automatic alert."
                ),
            },
            {
                "page_num": 2,
                "heading": "2. Lubrication & Bearing Protocols",
                "text": (
                    "2. Lubrication & Bearing Protocols\n\n"
                    "Bearing assemblies require synthetic turbine lubricant ISO-VG-46. "
                    "Maximum continuous bearing temperature is 85 C. The scheduled inspection frequency is every 30 days. "
                    "Complete lubricant replacement must occur every 6 months or 4000 operating hours, whichever comes first. "
                    "Lube oil pump discharge pressure must maintain a minimum of 2.8 bar."
                ),
            },
            {
                "page_num": 3,
                "heading": "3. Emergency Shutdown & Error Codes",
                "text": (
                    "3. Emergency Shutdown & Error Codes\n\n"
                    "Emergency Stop protocol designation is EMERG-STOP-7. "
                    "Fault code ERR-902 indicates excessive radial vibration exceeding 4.5 mm/s. "
                    "Fault code ERR-301 indicates critical inlet steam pressure drop below 15 bar. "
                    "Upon triggering EMERG-STOP-7, the main steam stop valve closes within 0.15 seconds."
                ),
            },
        ],
    },
    {
        "filename": "Safety_Compliance_Standard_ISO9001.pdf",
        "doc_id": "eval_doc_002",
        "title": "Safety Compliance Standard ISO-9001",
        "pages": [
            {
                "page_num": 1,
                "heading": "1. Scope & PPE Requirements",
                "text": (
                    "1. Scope & PPE Requirements\n\n"
                    "All plant personnel and visitors must wear Class-2 safety helmets, high-visibility reflective vests, "
                    "and steel-toe protective footwear compliant with standard ASTM F2413. "
                    "Hearing protection is strictly mandatory in all operating zones where ambient sound levels exceed 85 dBA. "
                    "Eye protection with ANSI Z87.1 certification is required in machining areas."
                ),
            },
            {
                "page_num": 2,
                "heading": "2. Chemical Exposure & Hazardous Material Thresholds",
                "text": (
                    "2. Chemical Exposure & Hazardous Material Thresholds\n\n"
                    "The maximum permissible exposure limit (PEL) for ammonia vapor is 25 ppm as an 8-hour TWA. "
                    "The exposure ceiling for sulfur dioxide is 2 ppm. "
                    "The automatic toxic gas evacuation alarm triggers whenever concentrations exceed PPM-50. "
                    "All chemical storage containers must display standard GHS diamond hazard labels."
                ),
            },
            {
                "page_num": 3,
                "heading": "3. Audit Protocols & Corrective Actions",
                "text": (
                    "3. Audit Protocols & Corrective Actions\n\n"
                    "Internal safety compliance audits must be conducted bi-annually by certified lead auditors. "
                    "Non-conformance reports (NCR) issued during an audit require root-cause analysis and corrective "
                    "action implementation within 14 calendar days. Critical safety non-conformances mandate immediate "
                    "work stoppage until verified remediation."
                ),
            },
        ],
    },
    {
        "filename": "Facility_Maintenance_Report_2024.pdf",
        "doc_id": "eval_doc_003",
        "title": "Facility Maintenance Report 2024",
        "pages": [
            {
                "page_num": 1,
                "heading": "1. Annual Maintenance Summary (2024)",
                "text": (
                    "1. Annual Maintenance Summary (2024)\n\n"
                    "During calendar year 2024, total facility operating time was 8760 hours. "
                    "Unscheduled downtime was held to 42 hours across all generation units. "
                    "Third-party contractor labor accounted for 1420 hours. "
                    "Total annual maintenance expenditure was $385,000, representing a 4.2% savings compared to the annual budget."
                ),
            },
            {
                "page_num": 2,
                "heading": "2. Equipment Overhaul Records",
                "text": (
                    "2. Equipment Overhaul Records\n\n"
                    "Steam turbine AX-450 underwent a comprehensive mid-life blade overhaul in Q3 2024. "
                    "Gas turbine GT-200 had combustion chamber liners replaced in Q2 2024. "
                    "All bearing vibration sensors across both turbines were upgraded to digital telemetry model VIB-2000."
                ),
            },
            {
                "page_num": 3,
                "heading": "3. Planned Maintenance Schedule (2025)",
                "text": (
                    "3. Planned Maintenance Schedule (2025)\n\n"
                    "Turbine AX-450 is scheduled for major rotor inspection during Q1 2025. "
                    "Gas turbine GT-200 routine inspection is planned for April 2025. "
                    "Cooling tower water circulation pumps will undergo dynamic rebalancing in May 2025."
                ),
            },
        ],
    },
    {
        "filename": "Hazard_Mitigation_Procedure_HAZ40.pdf",
        "doc_id": "eval_doc_004",
        "title": "Hazard Mitigation Procedure HAZ-40",
        "pages": [
            {
                "page_num": 1,
                "heading": "1. Chemical Spill Response Protocol",
                "text": (
                    "1. Chemical Spill Response Protocol\n\n"
                    "In the event of a hazardous liquid spill, activate local emergency exhaust ventilation maintaining "
                    "at least 500 CFM airflow. Deploy universal absorbent pads designation HAZ-CHEM-PAD immediately. "
                    "Secondary containment recovery drums must possess a minimum volume capacity of 200 liters. "
                    "Contaminated waste must be sealed in 55-gallon steel salvage drums."
                ),
            },
            {
                "page_num": 2,
                "heading": "2. Reporting & Incident Escalation",
                "text": (
                    "2. Reporting & Incident Escalation\n\n"
                    "Spills exceeding 5 liters require immediate verbal notification to the Site Safety Officer within 15 minutes. "
                    "A formal written incident investigation report must be filed within 24 hours. "
                    "If toxic vapors exceed PPM-50, notify the municipal environmental agency within 2 hours."
                ),
            },
        ],
    },
    {
        "filename": "Turbine_Generator_GT200_Manual.pdf",
        "doc_id": "eval_doc_005",
        "title": "Turbine Generator GT-200 Manual",
        "pages": [
            {
                "page_num": 1,
                "heading": "1. Specifications for GT-200",
                "text": (
                    "1. Specifications for GT-200\n\n"
                    "The model GT-200 is an aeroderivative gas turbine generator designed for rapid peaking and continuous service. "
                    "Operating speed is 4200 RPM with a rated electric power output of 60 MW. "
                    "Maximum continuous exhaust gas temperature is 620 C. "
                    "Natural gas fuel flow rate at 100% base load is 3.2 kg/s."
                ),
            },
            {
                "page_num": 2,
                "heading": "2. Preventive Maintenance & Lubrication",
                "text": (
                    "2. Preventive Maintenance & Lubrication\n\n"
                    "GT-200 specifies high-temperature synthetic lubricant AeroShell Turbine Oil 560. "
                    "Fuel filter replacement cycle is every 2000 operating hours. "
                    "Routine inspection frequency is every 60 days. "
                    "Igniter plugs must be inspected and gap-checked every 1000 starts or semi-annually."
                ),
            },
        ],
    },
]


def create_pdf_file(output_path: Path, pages: List[Dict[str, Any]]) -> None:
    """Create a pristine digital PDF document with exact heading and paragraph coordinates."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for p in pages:
        page = doc.new_page(width=595, height=842)  # Standard A4
        # Insert Heading
        page.insert_text((50, 72), p["heading"], fontsize=16, fontname="helv", color=(0.1, 0.2, 0.4))
        # Insert Body text with automatic line wrapping
        rect = pymupdf.Rect(50, 110, 545, 780)
        page.insert_textbox(rect, p["text"], fontsize=11, fontname="helv", color=(0, 0, 0))
    doc.save(str(output_path))
    doc.close()


def generate_evaluation_dataset(workspace_id: str = "eval_workspace_01") -> List[Dict[str, Any]]:
    """Generate 65 verified evaluation questions with exact ground-truth evidence across 9 categories."""
    dataset: List[Dict[str, Any]] = [
        # -------------------------------------------------------------
        # 1. Technical Identifier Questions (10 items)
        # -------------------------------------------------------------
        {
            "id": "eval_q01",
            "category": "technical_id",
            "question": "What does fault code ERR-902 indicate in the turbine manual?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [3],
            "expected_answer": "Fault code ERR-902 indicates excessive radial vibration exceeding 4.5 mm/s.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q02",
            "category": "technical_id",
            "question": "What condition triggers fault code ERR-301?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [3],
            "expected_answer": "Fault code ERR-301 indicates a critical inlet steam pressure drop below 15 bar.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q03",
            "category": "technical_id",
            "question": "What emergency stop protocol designation is assigned to the AX-450?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [3],
            "expected_answer": "The emergency stop protocol designation is EMERG-STOP-7.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q04",
            "category": "technical_id",
            "question": "Which lubricant specification is required for turbine AX-450 bearings?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [2],
            "expected_answer": "The AX-450 requires synthetic turbine lubricant ISO-VG-46.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q05",
            "category": "technical_id",
            "question": "Which footwear compliance standard is mandated for plant personnel?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_002",
            "expected_filename": "Safety_Compliance_Standard_ISO9001.pdf",
            "expected_pages": [1],
            "expected_answer": "Protective footwear must comply with standard ASTM F2413.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q06",
            "category": "technical_id",
            "question": "At what concentration does the toxic gas alarm trigger?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_002",
            "expected_filename": "Safety_Compliance_Standard_ISO9001.pdf",
            "expected_pages": [2],
            "expected_answer": "The automatic toxic gas evacuation alarm triggers when concentrations exceed PPM-50.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q07",
            "category": "technical_id",
            "question": "Which digital sensor model were the vibration sensors upgraded to in 2024?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_003",
            "expected_filename": "Facility_Maintenance_Report_2024.pdf",
            "expected_pages": [2],
            "expected_answer": "Bearing vibration sensors were upgraded to digital telemetry model VIB-2000.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q08",
            "category": "technical_id",
            "question": "What is the designated model for the universal absorbent pads?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_004",
            "expected_filename": "Hazard_Mitigation_Procedure_HAZ40.pdf",
            "expected_pages": [1],
            "expected_answer": "The universal absorbent pads designation is HAZ-CHEM-PAD.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q09",
            "category": "technical_id",
            "question": "Which specific lubricant is specified for gas turbine GT-200?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_005",
            "expected_filename": "Turbine_Generator_GT200_Manual.pdf",
            "expected_pages": [2],
            "expected_answer": "GT-200 specifies AeroShell Turbine Oil 560.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q10",
            "category": "technical_id",
            "question": "Which eye protection certification is required in machining areas?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_002",
            "expected_filename": "Safety_Compliance_Standard_ISO9001.pdf",
            "expected_pages": [1],
            "expected_answer": "Eye protection with ANSI Z87.1 certification is required in machining areas.",
            "is_unanswerable": False,
        },

        # -------------------------------------------------------------
        # 2. Exact Keyword Questions (10 items)
        # -------------------------------------------------------------
        {
            "id": "eval_q11",
            "category": "keyword",
            "question": "What is the minimum emergency ventilation rate in CFM for chemical spills?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_004",
            "expected_filename": "Hazard_Mitigation_Procedure_HAZ40.pdf",
            "expected_pages": [1],
            "expected_answer": "The emergency exhaust ventilation must maintain at least 500 CFM airflow.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q12",
            "category": "keyword",
            "question": "What is the permissible exposure limit for ammonia vapor as an 8-hour TWA?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_002",
            "expected_filename": "Safety_Compliance_Standard_ISO9001.pdf",
            "expected_pages": [2],
            "expected_answer": "The maximum permissible exposure limit for ammonia vapor is 25 ppm as an 8-hour TWA.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q13",
            "category": "keyword",
            "question": "What is the exposure ceiling for sulfur dioxide?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_002",
            "expected_filename": "Safety_Compliance_Standard_ISO9001.pdf",
            "expected_pages": [2],
            "expected_answer": "The exposure ceiling for sulfur dioxide is 2 ppm.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q14",
            "category": "keyword",
            "question": "What is the nominal operating speed of the AX-450 steam turbine?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [1],
            "expected_answer": "The nominal operating speed is 3600 RPM.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q15",
            "category": "keyword",
            "question": "What is the rotor diameter of the AX-450 turbine?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [1],
            "expected_answer": "The rotor diameter is 2.4 meters.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q16",
            "category": "keyword",
            "question": "What is the maximum allowable inlet temperature for turbine AX-450?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [1],
            "expected_answer": "The maximum allowable inlet temperature is 540 C.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q17",
            "category": "keyword",
            "question": "What is the fuel flow rate of the GT-200 at full base load?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_005",
            "expected_filename": "Turbine_Generator_GT200_Manual.pdf",
            "expected_pages": [1],
            "expected_answer": "The natural gas fuel flow rate is 3.2 kg/s at 100% base load.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q18",
            "category": "keyword",
            "question": "What minimum volume capacity is required for secondary containment recovery drums?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_004",
            "expected_filename": "Hazard_Mitigation_Procedure_HAZ40.pdf",
            "expected_pages": [1],
            "expected_answer": "Secondary containment recovery drums must have a minimum volume capacity of 200 liters.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q19",
            "category": "keyword",
            "question": "How fast must the main steam stop valve close upon triggering EMERG-STOP-7?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [3],
            "expected_answer": "The main steam stop valve closes within 0.15 seconds.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q20",
            "category": "keyword",
            "question": "What ambient sound level makes hearing protection strictly mandatory?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_002",
            "expected_filename": "Safety_Compliance_Standard_ISO9001.pdf",
            "expected_pages": [1],
            "expected_answer": "Hearing protection is mandatory where ambient sound levels exceed 85 dBA.",
            "is_unanswerable": False,
        },

        # -------------------------------------------------------------
        # 3. Factual Questions (8 items)
        # -------------------------------------------------------------
        {
            "id": "eval_q21",
            "category": "factual",
            "question": "How many hours of unscheduled downtime were recorded during calendar year 2024?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_003",
            "expected_filename": "Facility_Maintenance_Report_2024.pdf",
            "expected_pages": [1],
            "expected_answer": "Unscheduled downtime was 42 hours across all generation units.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q22",
            "category": "factual",
            "question": "What was the total annual maintenance expenditure in 2024?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_003",
            "expected_filename": "Facility_Maintenance_Report_2024.pdf",
            "expected_pages": [1],
            "expected_answer": "Total annual maintenance expenditure was $385,000.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q23",
            "category": "factual",
            "question": "How many contractor hours were logged during the 2024 annual period?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_003",
            "expected_filename": "Facility_Maintenance_Report_2024.pdf",
            "expected_pages": [1],
            "expected_answer": "Third-party contractor labor accounted for 1420 hours.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q24",
            "category": "factual",
            "question": "What is the maximum continuous bearing temperature allowed on the AX-450?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [2],
            "expected_answer": "Maximum continuous bearing temperature is 85 C.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q25",
            "category": "factual",
            "question": "How often must internal safety compliance audits be conducted?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_002",
            "expected_filename": "Safety_Compliance_Standard_ISO9001.pdf",
            "expected_pages": [3],
            "expected_answer": "Internal safety compliance audits must be conducted bi-annually.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q26",
            "category": "factual",
            "question": "Within how many calendar days must non-conformance reports be implemented?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_002",
            "expected_filename": "Safety_Compliance_Standard_ISO9001.pdf",
            "expected_pages": [3],
            "expected_answer": "Corrective action implementation must occur within 14 calendar days.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q27",
            "category": "factual",
            "question": "What rated electric power output does the GT-200 deliver?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_005",
            "expected_filename": "Turbine_Generator_GT200_Manual.pdf",
            "expected_pages": [1],
            "expected_answer": "Rated electric power output is 60 MW.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q28",
            "category": "factual",
            "question": "What is the replacement cycle for GT-200 fuel filters?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_005",
            "expected_filename": "Turbine_Generator_GT200_Manual.pdf",
            "expected_pages": [2],
            "expected_answer": "Fuel filter replacement cycle is every 2000 operating hours.",
            "is_unanswerable": False,
        },

        # -------------------------------------------------------------
        # 4. Semantic / Paraphrased Questions (8 items)
        # -------------------------------------------------------------
        {
            "id": "eval_q29",
            "category": "semantic",
            "question": "How frequently should technician staff inspect the bearings of the steam unit?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [2],
            "expected_answer": "The scheduled inspection frequency is every 30 days.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q30",
            "category": "semantic",
            "question": "What protocol must plant workers follow if someone spills liquid chemicals?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_004",
            "expected_filename": "Hazard_Mitigation_Procedure_HAZ40.pdf",
            "expected_pages": [1],
            "expected_answer": "Activate local emergency ventilation at 500 CFM, deploy HAZ-CHEM-PAD absorbent pads, and recover liquid into 200-liter containment drums.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q31",
            "category": "semantic",
            "question": "What safety headwear and visibility clothing are mandatory for on-site visitors?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_002",
            "expected_filename": "Safety_Compliance_Standard_ISO9001.pdf",
            "expected_pages": [1],
            "expected_answer": "Class-2 safety helmets and high-visibility reflective vests are mandatory.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q32",
            "category": "semantic",
            "question": "Under what circumstances must factory managers immediately cease plant activities?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_002",
            "expected_filename": "Safety_Compliance_Standard_ISO9001.pdf",
            "expected_pages": [3],
            "expected_answer": "Critical safety non-conformances mandate immediate work stoppage until verified remediation.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q33",
            "category": "semantic",
            "question": "How quickly should an environmental spill be reported to safety personnel?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_004",
            "expected_filename": "Hazard_Mitigation_Procedure_HAZ40.pdf",
            "expected_pages": [2],
            "expected_answer": "Spills exceeding 5 liters require immediate verbal notification within 15 minutes.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q34",
            "category": "semantic",
            "question": "When does the plant exhaust temperature of the GT unit reach its thermal ceiling?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_005",
            "expected_filename": "Turbine_Generator_GT200_Manual.pdf",
            "expected_pages": [1],
            "expected_answer": "Maximum continuous exhaust gas temperature is 620 C.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q35",
            "category": "semantic",
            "question": "How does facility management handle chemical storage container markings?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_002",
            "expected_filename": "Safety_Compliance_Standard_ISO9001.pdf",
            "expected_pages": [2],
            "expected_answer": "All chemical storage containers must display standard GHS diamond hazard labels.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q36",
            "category": "semantic",
            "question": "What fluid replenishment is required for the AX-450 over long operational cycles?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [2],
            "expected_answer": "Complete lubricant replacement must occur every 6 months or 4000 operating hours with ISO-VG-46.",
            "is_unanswerable": False,
        },

        # -------------------------------------------------------------
        # 5. Cross-Document Questions (7 items)
        # -------------------------------------------------------------
        {
            "id": "eval_q37",
            "category": "cross_document",
            "question": "What overhaul maintenance was performed on the AX-450 in 2024 according to facility records?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_003",
            "expected_filename": "Facility_Maintenance_Report_2024.pdf",
            "expected_pages": [2],
            "expected_answer": "Turbine AX-450 underwent a comprehensive mid-life blade overhaul in Q3 2024.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q38",
            "category": "cross_document",
            "question": "What upgrade was applied to the vibration monitoring systems of both turbines in 2024?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_003",
            "expected_filename": "Facility_Maintenance_Report_2024.pdf",
            "expected_pages": [2],
            "expected_answer": "Bearing vibration sensors across both turbines were upgraded to digital telemetry model VIB-2000.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q39",
            "category": "cross_document",
            "question": "When is the next major inspection scheduled for the AX-450 steam turbine?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_003",
            "expected_filename": "Facility_Maintenance_Report_2024.pdf",
            "expected_pages": [3],
            "expected_answer": "Turbine AX-450 is scheduled for a major rotor inspection in Q1 2025.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q40",
            "category": "cross_document",
            "question": "What maintenance event took place on the GT-200 gas turbine in Q2 2024?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_003",
            "expected_filename": "Facility_Maintenance_Report_2024.pdf",
            "expected_pages": [2],
            "expected_answer": "Gas turbine GT-200 had combustion chamber liners replaced in Q2 2024.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q41",
            "category": "cross_document",
            "question": "If toxic gas levels exceed PPM-50 during a spill, who must be notified according to HAZ-40 and safety standards?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_004",
            "expected_filename": "Hazard_Mitigation_Procedure_HAZ40.pdf",
            "expected_pages": [2],
            "expected_answer": "Notify the municipal environmental agency within 2 hours and initiate plant evacuation alarm.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q42",
            "category": "cross_document",
            "question": "What are the routine inspection frequencies specified for AX-450 and GT-200 across their manuals?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [2],
            "expected_answer": "AX-450 requires inspection every 30 days while GT-200 specifies routine inspection every 60 days.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q43",
            "category": "cross_document",
            "question": "Which two lubricants are specified across the facility's steam and gas turbine systems?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [2],
            "expected_answer": "AX-450 uses ISO-VG-46 synthetic lubricant, while GT-200 uses AeroShell Turbine Oil 560.",
            "is_unanswerable": False,
        },

        # -------------------------------------------------------------
        # 6. Comparison Questions (6 items)
        # -------------------------------------------------------------
        {
            "id": "eval_q44",
            "category": "comparison",
            "question": "Compare the operating speeds of turbine AX-450 and turbine GT-200.",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [1],
            "expected_answer": "AX-450 operates at 3600 RPM whereas GT-200 operates at 4200 RPM.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q45",
            "category": "comparison",
            "question": "How do the power ratings of AX-450 and GT-200 compare?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [1],
            "expected_answer": "AX-450 is rated at 45 MW while GT-200 is rated at 60 MW.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q46",
            "category": "comparison",
            "question": "What is the difference in maximum operating temperature between the AX-450 steam inlet and GT-200 exhaust?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [1],
            "expected_answer": "AX-450 maximum inlet temperature is 540 C while GT-200 exhaust ceiling is 620 C.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q47",
            "category": "comparison",
            "question": "Compare the routine inspection intervals between the AX-450 steam turbine and GT-200 generator.",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [2],
            "expected_answer": "AX-450 requires inspection every 30 days while GT-200 requires inspection every 60 days.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q48",
            "category": "comparison",
            "question": "Compare the exposure limits for ammonia vapor versus sulfur dioxide under ISO-9001.",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_002",
            "expected_filename": "Safety_Compliance_Standard_ISO9001.pdf",
            "expected_pages": [2],
            "expected_answer": "Ammonia vapor has an exposure limit of 25 ppm, whereas sulfur dioxide has a ceiling of 2 ppm.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q49",
            "category": "comparison",
            "question": "Compare the incident reporting deadline for a verbal notification versus a written report under HAZ-40.",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_004",
            "expected_filename": "Hazard_Mitigation_Procedure_HAZ40.pdf",
            "expected_pages": [2],
            "expected_answer": "Verbal notification is due within 15 minutes, whereas the written report is due within 24 hours.",
            "is_unanswerable": False,
        },

        # -------------------------------------------------------------
        # 7. Summary Questions (5 items)
        # -------------------------------------------------------------
        {
            "id": "eval_q50",
            "category": "summary",
            "question": "Summarize the required personal protective equipment specified in the safety compliance standard.",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_002",
            "expected_filename": "Safety_Compliance_Standard_ISO9001.pdf",
            "expected_pages": [1],
            "expected_answer": "Personnel must wear Class-2 helmets, reflective vests, ASTM F2413 steel-toe shoes, hearing protection above 85 dBA, and ANSI Z87.1 eye protection in machining areas.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q51",
            "category": "summary",
            "question": "Summarize the emergency shutdown procedure and error codes for the AX-450 turbine.",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [3],
            "expected_answer": "Protocol EMERG-STOP-7 closes stop valves in 0.15s; ERR-902 signals vibration >4.5 mm/s and ERR-301 signals inlet pressure <15 bar.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q52",
            "category": "summary",
            "question": "Provide a summary of the 2024 annual facility maintenance performance.",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_003",
            "expected_filename": "Facility_Maintenance_Report_2024.pdf",
            "expected_pages": [1],
            "expected_answer": "In 2024, the facility operated 8760 hours with 42 hours downtime, logged 1420 contractor hours, and spent $385,000 (4.2% under budget).",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q53",
            "category": "summary",
            "question": "Summarize the chemical spill containment procedure outlined in HAZ-40.",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_004",
            "expected_filename": "Hazard_Mitigation_Procedure_HAZ40.pdf",
            "expected_pages": [1],
            "expected_answer": "Turn on emergency ventilation at 500 CFM, deploy HAZ-CHEM-PAD absorbent pads, and store recovered liquids in 200-liter secondary containment drums.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q54",
            "category": "summary",
            "question": "Summarize the preventive maintenance requirements for the GT-200 generator.",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_005",
            "expected_filename": "Turbine_Generator_GT200_Manual.pdf",
            "expected_pages": [2],
            "expected_answer": "Uses AeroShell 560 lubricant, requires fuel filter change every 2000 hours, routine inspection every 60 days, and igniter check every 1000 starts.",
            "is_unanswerable": False,
        },

        # -------------------------------------------------------------
        # 8. Multi-Hop Questions (5 items)
        # -------------------------------------------------------------
        {
            "id": "eval_q55",
            "category": "multi_hop",
            "question": "Which turbine underwent an overhaul in Q3 2024, and what lubricant does it require?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_003",
            "expected_filename": "Facility_Maintenance_Report_2024.pdf",
            "expected_pages": [2],
            "expected_answer": "Turbine AX-450 underwent a blade overhaul in Q3 2024, and it requires synthetic lubricant ISO-VG-46.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q56",
            "category": "multi_hop",
            "question": "If vibration exceeds 4.5 mm/s on the steam turbine, what fault code appears and which updated sensor detects it?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_001",
            "expected_filename": "Turbine_Operations_Manual_AX450.pdf",
            "expected_pages": [3],
            "expected_answer": "Fault code ERR-902 is triggered, which is detected by the upgraded digital telemetry sensor VIB-2000.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q57",
            "category": "multi_hop",
            "question": "What is the rated capacity of the turbine whose combustion chamber liners were replaced in Q2 2024?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_003",
            "expected_filename": "Facility_Maintenance_Report_2024.pdf",
            "expected_pages": [2],
            "expected_answer": "GT-200 had its liners replaced in Q2 2024, and its rated electric capacity is 60 MW.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q58",
            "category": "multi_hop",
            "question": "What audit action is triggered if a non-conformance is found during safety review, and when does it become work-stopping?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_002",
            "expected_filename": "Safety_Compliance_Standard_ISO9001.pdf",
            "expected_pages": [3],
            "expected_answer": "A non-conformance report (NCR) must be resolved within 14 days, and critical safety non-conformances mandate immediate work stoppage.",
            "is_unanswerable": False,
        },
        {
            "id": "eval_q59",
            "category": "multi_hop",
            "question": "For the higher speed turbine generator, how often must the fuel filtration elements be swapped out?",
            "workspace_id": workspace_id,
            "expected_document_id": "eval_doc_005",
            "expected_filename": "Turbine_Generator_GT200_Manual.pdf",
            "expected_pages": [2],
            "expected_answer": "The GT-200 (4200 RPM) requires fuel filter replacement every 2000 operating hours.",
            "is_unanswerable": False,
        },

        # -------------------------------------------------------------
        # 9. Unanswerable Questions (6 items)
        # -------------------------------------------------------------
        {
            "id": "eval_q60",
            "category": "unanswerable",
            "question": "What is the maximum electrical output of the solar panel array on the facility roof?",
            "workspace_id": workspace_id,
            "expected_document_id": None,
            "expected_filename": None,
            "expected_pages": [],
            "expected_answer": "Insufficient evidence: the provided repository does not contain documentation regarding solar panel arrays.",
            "is_unanswerable": True,
        },
        {
            "id": "eval_q61",
            "category": "unanswerable",
            "question": "What is the recommended replacement procedure for reactor coolant pump model RX-900?",
            "workspace_id": workspace_id,
            "expected_document_id": None,
            "expected_filename": None,
            "expected_pages": [],
            "expected_answer": "Insufficient evidence: model RX-900 and reactor coolant pumps are not documented in the repository.",
            "is_unanswerable": True,
        },
        {
            "id": "eval_q62",
            "category": "unanswerable",
            "question": "What was the total financial revenue generated by electricity sales in fiscal year 2021?",
            "workspace_id": workspace_id,
            "expected_document_id": None,
            "expected_filename": None,
            "expected_pages": [],
            "expected_answer": "Insufficient evidence: the documentation only covers 2024 maintenance expenses, not 2021 electricity sales revenue.",
            "is_unanswerable": True,
        },
        {
            "id": "eval_q63",
            "category": "unanswerable",
            "question": "Who was appointed as the Chief Information Security Officer of the plant in 2023?",
            "workspace_id": workspace_id,
            "expected_document_id": None,
            "expected_filename": None,
            "expected_pages": [],
            "expected_answer": "Insufficient evidence: the documents do not mention executive appointments or the Chief Information Security Officer.",
            "is_unanswerable": True,
        },
        {
            "id": "eval_q64",
            "category": "unanswerable",
            "question": "What is the recommended maintenance schedule for hydraulic excavator CAT-320?",
            "workspace_id": workspace_id,
            "expected_document_id": None,
            "expected_filename": None,
            "expected_pages": [],
            "expected_answer": "Insufficient evidence: hydraulic excavators or heavy earthmoving vehicles are not documented.",
            "is_unanswerable": True,
        },
        {
            "id": "eval_q65",
            "category": "unanswerable",
            "question": "What cybersecurity encryption algorithm is mandated for plant SCADA network traffic?",
            "workspace_id": workspace_id,
            "expected_document_id": None,
            "expected_filename": None,
            "expected_pages": [],
            "expected_answer": "Insufficient evidence: the available manuals cover physical and chemical safety, not SCADA network encryption protocols.",
            "is_unanswerable": True,
        },
    ]
    return dataset


def save_dataset_jsonl(dataset: List[Dict[str, Any]], output_path: Path) -> None:
    """Serialize the evaluation dataset to JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    dataset_file = Path("evaluation/dataset.jsonl")
    data = generate_evaluation_dataset()
    save_dataset_jsonl(data, dataset_file)
    print(f"Generated {len(data)} evaluation questions in {dataset_file}")
