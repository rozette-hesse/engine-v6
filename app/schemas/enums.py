"""Input enum definitions for all symptom and lifestyle signals."""

from enum import Enum


class Phase(str, Enum):
    menstrual   = "MEN"
    follicular  = "FOL"
    ovulatory   = "OVU"
    luteal      = "LUT"
    late_luteal = "LL"


class EnergyLevel(str, Enum):
    very_low  = "Very Low"
    low       = "Low"
    medium    = "Medium"
    high      = "High"
    very_high = "Very High"


class StressLevel(str, Enum):
    low      = "Low"
    moderate = "Moderate"
    high     = "High"
    severe   = "Severe"


class MoodLevel(str, Enum):
    very_low  = "Very Low"
    low       = "Low"
    medium    = "Medium"
    high      = "High"
    very_high = "Very High"


class CrampLevel(str, Enum):
    none        = "None"
    mild        = "Mild"
    moderate    = "Moderate"
    severe      = "Severe"
    very_severe = "Very Severe"


class MoodSwingLevel(str, Enum):
    none     = "None"
    mild     = "Mild"
    moderate = "Moderate"
    severe   = "Severe"


class CravingsLevel(str, Enum):
    none     = "None"
    mild     = "Mild"
    moderate = "Moderate"
    strong   = "Strong"


class FlowLevel(str, Enum):
    none       = "None"
    light      = "Light"
    moderate   = "Moderate"
    heavy      = "Heavy"
    very_heavy = "Very Heavy"


class TendernessLevel(str, Enum):
    none     = "None"
    mild     = "Mild"
    moderate = "Moderate"
    severe   = "Severe"


class IndigestionLevel(str, Enum):
    none     = "None"
    mild     = "Mild"
    moderate = "Moderate"
    severe   = "Severe"


class BrainFogLevel(str, Enum):
    none     = "None"
    mild     = "Mild"
    moderate = "Moderate"
    severe   = "Severe"


class BloatingLevel(str, Enum):
    none     = "None"
    mild     = "Mild"
    moderate = "Moderate"
    severe   = "Severe"


class SleepQuality(str, Enum):
    poor      = "Poor"
    fair      = "Fair"
    good      = "Good"
    very_good = "Very Good"


class IrritabilityLevel(str, Enum):
    none     = "None"
    mild     = "Mild"
    moderate = "Moderate"
    severe   = "Severe"


class CervicalMucus(str, Enum):
    unknown  = "unknown"
    dry      = "dry"
    sticky   = "sticky"
    creamy   = "creamy"
    watery   = "watery"
    eggwhite = "eggwhite"
