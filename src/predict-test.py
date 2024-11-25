import requests

url = "http://localhost:9696/predict"

respondent_id = "abc-123"
respondent = {
    "h1n1_concern": "2.0",
    "h1n1_knowledge": "2.0",
    "behavioral_antiviral_meds": "0.0",
    "behavioral_avoidance": "0.0",
    "behavioral_face_mask": "0.0",
    "behavioral_wash_hands": "1.0",
    "behavioral_large_gatherings": "0.0",
    "behavioral_outside_home": "0.0",
    "behavioral_touch_face": "0.0",
    "doctor_recc_h1n1": "1.0",
    "doctor_recc_seasonal": "1.0",
    "chronic_med_condition": "0.0",
    "child_under_6_months": "0.0",
    "health_worker": "0.0",
    "opinion_h1n1_vacc_effective": "4.0",
    "opinion_h1n1_risk": "2.0",
    "opinion_h1n1_sick_from_vacc": "1.0",
    "opinion_seas_vacc_effective": "5.0",
    "opinion_seas_risk": "2.0",
    "opinion_seas_sick_from_vacc": "1.0",
    "age_group": "55_to_64",
    "education": "some_college",
    "race": "white",
    "sex": "female",
    "income_poverty": "above_poverty_lte_75k",
    "marital_status": "married",
    "rent_or_own": "own",
    "employment_status": "employed",
    "hhs_geo_region": "fpwskwrf",
    "census_msa": "msa_not_principle__city",
    "household_adults": "2.0",
    "household_children": "0.0"
 }

targets = {
    "h1n1_vaccine": 0,
    "seasonal_vaccine": 1
}

response = requests.post(url, json=respondent).json()
print(response)

if response["h1n1"] == True:
    print(f"Incorrect. {respondent_id} did not get h1n1 vaccine.")
else:
    print(f"Correct. {respondent_id} did not get h1n1 vaccine.")

if response["seasonal"] == True:
    print(f"Correct. {respondent_id} did get seasonal flu vaccine.")
else:
    print(f"Incorrect. {respondent_id} did get seasonal flu vaccine.")
