"""
Configuration file - PUT YOUR API KEYS HERE
"""

# ========== GROQ API KEYS (REQUIRED) ==========
# Get free keys from: https://console.groq.com/keys
GROQ_API_KEYS = [
        "",   # Replace with your first key
        ""   # Replace with your second key
    ]

# ========== MODEL SETTINGS ==========
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality
LLM_MODEL = "openai/gpt-oss-120b"  # Best Groq model

# ========== RAG SETTINGS ==========
CHUNK_SIZE = 800           # Characters per chunk
CHUNK_OVERLAP = 150        # Overlap between chunks
TOP_K_CHUNKS = 3           # How many chunks to retrieve per indicator

# ========== INDICATORS TO EXTRACT ==========
INDICATORS = [
    {
        "id": 1,
        "name": "Total Scope 1 GHG Emissions",
        "unit": "tCO₂e",
        "keywords": ["scope 1", "direct emissions", "combustion", "stationary"],
        "context": "environmental GHG emissions climate"
    },
    {
        "id": 2,
        "name": "Total Scope 2 GHG Emissions",
        "unit": "tCO₂e",
        "keywords": ["scope 2", "indirect emissions", "purchased electricity", "location-based"],
        "context": "environmental GHG emissions energy"
    },
    {
        "id": 3,
        "name": "Total Scope 3 GHG Emissions",
        "unit": "tCO₂e",
        "keywords": ["scope 3", "value chain", "financed emissions", "category 15"],
        "context": "environmental GHG emissions value chain"
    },
    {
        "id": 4,
        "name": "GHG Emissions Intensity",
        "unit": "tCO₂e per €M revenue",
        "keywords": ["emissions intensity", "carbon intensity", "emissions per revenue"],
        "context": "environmental intensity metrics"
    },
    {
        "id": 5,
        "name": "Total Energy Consumption",
        "unit": "MWh",
        "keywords": ["energy consumption", "total energy", "energy use", "electricity consumption"],
        "context": "environmental energy"
    },
    {
        "id": 6,
        "name": "Renewable Energy Percentage",
        "unit": "%",
        "keywords": ["renewable energy", "green energy", "renewable electricity", "clean energy percentage"],
        "context": "environmental energy renewable"
    },
    {
        "id": 7,
        "name": "Net Zero Target Year",
        "unit": "year",
        "keywords": ["net zero", "carbon neutral", "net-zero target", "2050", "climate target"],
        "context": "environmental climate commitment target"
    },
    {
        "id": 8,
        "name": "Green Financing Volume",
        "unit": "€ millions",
        "keywords": ["green financing", "sustainable finance", "green loans", "green bonds"],
        "context": "environmental finance sustainable"
    },
    {
        "id": 9,
        "name": "Total Employees",
        "unit": "FTE",
        "keywords": ["total employees", "workforce", "headcount", "FTE", "staff numbers"],
        "context": "social workforce employees"
    },
    {
        "id": 10,
        "name": "Female Employees",
        "unit": "%",
        "keywords": ["female employees", "women workforce", "gender diversity", "female representation"],
        "context": "social diversity gender"
    },
    {
        "id": 11,
        "name": "Gender Pay Gap",
        "unit": "%",
        "keywords": ["gender pay gap", "pay equity", "wage gap", "pay disparity"],
        "context": "social diversity compensation"
    },
    {
        "id": 12,
        "name": "Training Hours per Employee",
        "unit": "hours",
        "keywords": ["training hours", "average training", "learning hours", "development hours"],
        "context": "social learning development"
    },
    {
        "id": 13,
        "name": "Employee Turnover Rate",
        "unit": "%",
        "keywords": ["turnover rate", "attrition", "employee retention", "staff turnover"],
        "context": "social workforce retention"
    },
    {
        "id": 14,
        "name": "Work-Related Accidents",
        "unit": "count",
        "keywords": ["work accidents", "workplace injuries", "occupational accidents", "safety incidents"],
        "context": "social health safety"
    },
    {
        "id": 15,
        "name": "Collective Bargaining Coverage",
        "unit": "%",
        "keywords": ["collective bargaining", "union coverage", "labor rights", "collective agreements"],
        "context": "social labor rights"
    },
    {
        "id": 16,
        "name": "Board Female Representation",
        "unit": "%",
        "keywords": ["board diversity", "female board", "women directors", "board gender"],
        "context": "governance board diversity"
    },
    {
        "id": 17,
        "name": "Board Meetings",
        "unit": "count/year",
        "keywords": ["board meetings", "board sessions", "board convened"],
        "context": "governance board meetings"
    },
    {
        "id": 18,
        "name": "Corruption Incidents",
        "unit": "count",
        "keywords": ["corruption incidents", "bribery cases", "fraud incidents", "ethics violations"],
        "context": "governance ethics compliance"
    },
    {
        "id": 19,
        "name": "Avg Payment Period to Suppliers",
        "unit": "days",
        "keywords": ["payment period", "payment terms", "supplier payment", "days payable"],
        "context": "governance supply chain"
    },
    {
        "id": 20,
        "name": "Suppliers Screened for ESG",
        "unit": "%",
        "keywords": ["supplier screening", "ESG assessment", "supplier evaluation", "vendor screening"],
        "context": "governance supply chain ESG"
    }
]
