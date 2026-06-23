import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

n = 500

skills_pool = ["Python","Java","JavaScript","React","Node.js","SQL","Machine Learning",
               "Deep Learning","AWS","Docker","Kubernetes","Data Analysis","TensorFlow",
               "PyTorch","NLP","Computer Vision","FastAPI","Django","MongoDB","Redis",
               "C++","Go","Rust","Flutter","Swift","Kotlin","GraphQL","TypeScript",
               "Spark","Hadoop","Tableau","Power BI","Selenium","Jenkins","Git"]

educations = ["B.Tech","M.Tech","BCA","MCA","B.Sc CS","MBA","PhD"]
edu_scores  = {"B.Tech":70,"M.Tech":85,"BCA":60,"MCA":75,"B.Sc CS":65,"MBA":72,"PhD":92}

departments = ["Engineering","Data Science","Product","DevOps","Marketing","HR","Finance"]
job_titles  = ["Software Engineer","Data Scientist","ML Engineer","Backend Developer",
               "Frontend Developer","DevOps Engineer","Product Manager","Data Analyst"]

first_names = ["Aarav","Priya","Rohit","Sneha","Vikram","Ananya","Arjun","Kavya","Rahul",
               "Pooja","Amit","Neha","Siddharth","Riya","Karan","Divya","Manish","Swati",
               "Aditya","Megha","Suresh","Nisha","Rajesh","Preeti","Vivek","Anjali","Nikhil",
               "Shruti","Gaurav","Sakshi","Tarun","Pallavi","Varun","Kritika","Harsh","Simran"]
last_names  = ["Sharma","Verma","Singh","Gupta","Patel","Kumar","Mehta","Shah","Joshi",
               "Yadav","Mishra","Agarwal","Tiwari","Pandey","Chauhan","Saxena","Srivastava"]

rows = []
for i in range(n):
    fname   = random.choice(first_names)
    lname   = random.choice(last_names)
    name    = f"{fname} {lname}"
    email   = f"{fname.lower()}.{lname.lower()}{i}@email.com"
    exp     = round(random.uniform(0, 12), 1)
    edu     = random.choice(educations)
    edu_sc  = edu_scores[edu] + random.randint(-5, 5)
    n_skills= random.randint(3, 12)
    skills  = random.sample(skills_pool, min(n_skills, len(skills_pool)))
    skill_sc= min(100, n_skills * 8 + random.randint(-5, 10))
    certs   = random.randint(0, 7)
    cert_sc = min(100, certs * 13 + random.randint(-5, 5))
    projects= random.randint(1, 12)
    proj_sc = min(100, projects * 8 + random.randint(-5, 5))
    gpa     = round(random.uniform(5.0, 10.0), 2)
    comm_sc = random.randint(45, 100)
    prob_sc = random.randint(40, 100)
    ats_sc  = int(0.25*skill_sc + 0.20*edu_sc + 0.15*min(100,exp*8) +
                  0.15*cert_sc  + 0.10*proj_sc + 0.10*comm_sc + 0.05*prob_sc
                  + random.randint(-4, 4))
    ats_sc  = max(0, min(100, ats_sc))
    shortlisted = int(ats_sc >= 65 and exp >= 1 and n_skills >= 4)

    rows.append({
        "candidate_id"          : f"C{i+1:04d}",
        "name"                  : name,
        "email"                 : email,
        "education"             : edu,
        "gpa"                   : gpa,
        "experience_years"      : exp,
        "skills"                : ", ".join(skills),
        "num_skills"            : n_skills,
        "certifications"        : certs,
        "projects_count"        : projects,
        "department"            : random.choice(departments),
        "job_title_applied"     : random.choice(job_titles),
        "skill_score"           : skill_sc,
        "education_score"       : edu_sc,
        "certification_score"   : cert_sc,
        "project_score"         : proj_sc,
        "communication_score"   : comm_sc,
        "problem_solving_score" : prob_sc,
        "ats_score"             : ats_sc,
        "shortlisted"           : shortlisted,
    })

df = pd.DataFrame(rows)
df.to_csv("/home/claude/ai-hiring-assistant/backend/data/candidates_dataset.csv", index=False)
print(f"Dataset: {len(df)} rows | Shortlisted: {df['shortlisted'].sum()} | Not: {(df['shortlisted']==0).sum()}")
