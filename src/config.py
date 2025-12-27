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
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"  # Changed from gpt-oss (more reliable)

# ========== RAG SETTINGS (OPTIMIZED FOR 90%+) ==========
CHUNK_SIZE = 700           # Slightly smaller for precision
CHUNK_OVERLAP = 200        # More overlap to avoid missing data
TOP_K_CHUNKS = 10          # Retrieve more chunks (was 3)

# ========== ENHANCED INDICATORS ==========
INDICATORS = [
    {
        "id": 1,
        "name": "Total Scope 1 GHG Emissions",
        "unit": "tCO₂e",
        "keywords": [
            "scope 1", "direct emissions", "scope 1 ghg", "scope 1 emissions",
            "stationary combustion", "mobile combustion", "direct greenhouse gas",
            "scope one", "scope i emissions"
        ],
        "context": "environmental climate GHG emissions scope 1 direct table"
    },
    {
        "id": 2,
        "name": "Total Scope 2 GHG Emissions",
        "unit": "tCO₂e",
        "keywords": [
            "scope 2", "indirect emissions", "scope 2 ghg", "scope 2 emissions",
            "purchased electricity", "location-based", "market-based",
            "scope two", "scope ii emissions", "electricity emissions"
        ],
        "context": "environmental GHG emissions energy scope 2 indirect"
    },
    {
        "id": 3,
        "name": "Total Scope 3 GHG Emissions",
        "unit": "tCO₂e",
        "keywords": [
            "scope 3", "scope 3 emissions", "scope 3 ghg", "value chain emissions",
            "financed emissions", "category 15", "indirect scope 3",
            "scope three", "scope iii", "supply chain emissions"
        ],
        "context": "environmental GHG emissions value chain scope 3 financed"
    },
    {
        "id": 4,
        "name": "GHG Emissions Intensity",
        "unit": "tCO₂e per €M revenue",
        "keywords": [
            "emissions intensity", "carbon intensity", "ghg intensity",
            "emissions per revenue", "emissions per million", "intensity ratio",
            "emission intensity", "carbon footprint per revenue"
        ],
        "context": "environmental intensity metrics revenue carbon per million"
    },
    {
        "id": 5,
        "name": "Total Energy Consumption",
        "unit": "MWh",
        "keywords": [
            "total energy consumption", "energy consumption", "energy use",
            "electricity consumption", "total energy", "energy usage",
            "mwh consumption", "megawatt hours", "energy total"
        ],
        "context": "environmental energy consumption total mwh electricity"
    },
    {
        "id": 6,
        "name": "Renewable Energy Percentage",
        "unit": "%",
        "keywords": [
            "renewable energy", "renewable electricity", "green energy",
            "renewable percentage", "renewable share", "clean energy",
            "renewable sources", "green power", "renewable energy share"
        ],
        "context": "environmental energy renewable percentage share green"
    },
    {
        "id": 7,
        "name": "Net Zero Target Year",
        "unit": "year",
        "keywords": [
            "net zero", "net-zero", "carbon neutral", "net zero target",
            "2050 target", "climate target", "decarbonization target",
            "carbon neutrality", "net zero year", "net zero commitment"
        ],
        "context": "environmental climate commitment target 2050 net zero carbon"
    },
    {
        "id": 8,
        "name": "Green Financing Volume",
        "unit": "€ millions",
        "keywords": [
            "green financing", "sustainable finance", "green loans",
            "green bonds", "sustainable lending", "green portfolio",
            "climate finance", "esg financing", "green funding",
            "billion", "bn", "sustainable investment"
        ],
        "context": "environmental finance sustainable lending climate investment green"
    },
    {
        "id": 9,
        "name": "Total Employees",
        "unit": "FTE",
        "keywords": [
            "total employees", "workforce", "headcount", "fte",
            "staff numbers", "total workforce", "employee count",
            "full-time equivalent", "total staff", "employees total"
        ],
        "context": "social workforce employees headcount fte staff total"
    },
    {
        "id": 10,
        "name": "Female Employees",
        "unit": "%",
        "keywords": [
            "female employees", "women workforce", "female percentage",
            "gender diversity", "female representation", "women employees",
            "female staff", "gender ratio", "women percentage"
        ],
        "context": "social diversity gender female workforce women percentage"
    },
    {
        "id": 11,
        "name": "Gender Pay Gap",
        "unit": "%",
        "keywords": [
            "gender pay gap", "pay gap", "pay equity", "wage gap",
            "pay disparity", "gender wage gap", "compensation gap",
            "salary gap", "gender pay", "pay difference"
        ],
        "context": "social diversity compensation pay equity gender gap"
    },
    {
        "id": 12,
        "name": "Training Hours per Employee",
        "unit": "hours",
        "keywords": [
            "training hours", "average training", "training per employee",
            "learning hours", "development hours", "hours per employee",
            "training time", "average training hours", "hours training"
        ],
        "context": "social learning development training employee hours average"
    },
    {
        "id": 13,
        "name": "Employee Turnover Rate",
        "unit": "%",
        "keywords": [
            "turnover rate", "attrition", "staff turnover",
            "employee turnover", "voluntary turnover", "turnover percentage",
            "resignation rate", "exit rate", "retention rate", "leavers"
        ],
        "context": "social workforce stability retention turnover attrition"
    },
    {
        "id": 14,
        "name": "Work-Related Accidents",
        "unit": "count",
        "keywords": [
            "work accidents", "workplace injuries", "occupational accidents",
            "safety incidents", "work-related injuries", "accidents with leave",
            "workplace accidents", "injury count", "work incidents"
        ],
        "context": "social health safety workplace accidents injuries"
    },
    {
        "id": 15,
        "name": "Collective Bargaining Coverage",
        "unit": "%",
        "keywords": [
            "collective bargaining", "union coverage", "bargaining coverage",
            "labor rights", "collective agreements", "union membership",
            "collective agreement", "bargaining agreements", "union percentage"
        ],
        "context": "social labor rights union collective bargaining coverage"
    },
    {
        "id": 16,
        "name": "Board Female Representation",
        "unit": "%",
        "keywords": [
            "board diversity", "female board", "women directors",
            "board gender", "female directors", "women on board",
            "board representation", "female board members", "board women"
        ],
        "context": "governance board diversity gender female directors women"
    },
    {
        "id": 17,
        "name": "Board Meetings",
        "unit": "count/year",
        "keywords": [
            "board meetings", "board sessions", "board met",
            "meetings held", "board convened", "sessions per year",
            "board meeting count", "number of meetings", "meetings year"
        ],
        "context": "governance board meetings sessions count annual"
    },
    {
        "id": 18,
        "name": "Corruption Incidents",
        "unit": "count",
        "keywords": [
            "corruption incidents", "bribery cases", "corruption cases",
            "ethics violations", "fraud incidents", "bribery incidents",
            "no corruption", "zero corruption", "corruption count"
        ],
        "context": "governance ethics compliance corruption bribery fraud"
    },
    {
        "id": 19,
        "name": "Avg Payment Period to Suppliers",
        "unit": "days",
        "keywords": [
            "payment period", "payment terms", "supplier payment",
            "days payable", "payment days", "average payment",
            "supplier terms", "payment time", "days to pay"
        ],
        "context": "governance supply chain payment suppliers days average"
    },
    {
        "id": 20,
        "name": "Suppliers Screened for ESG",
        "unit": "%",
        "keywords": [
            "supplier screening", "esg assessment", "supplier evaluation",
            "vendor screening", "supplier esg", "sustainable procurement",
            "supplier assessment", "screened suppliers", "esg screening"
        ],
        "context": "governance supply chain esg screening sustainability assessment"
    }
]
