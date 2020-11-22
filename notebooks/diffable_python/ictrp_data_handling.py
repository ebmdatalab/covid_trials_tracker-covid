# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Data cleaning and handling

# +
#ICTRP Search: "covid-19" or "novel coronavirus" or "2019-ncov" or "covid19" or "sars-cov-2"
import xmltodict
import pandas as pd
import numpy as np
from datetime import date
import re
import unicodedata
import text_unidecode as unidecode
from collections import Counter

#POINT THIS TO THE UPDATED XML - deprecated as ICTRP is down.

#with open('ICTRP-Results_18Mar2020.xml', 'rb') as f:
#    xml = xmltodict.parse(f, dict_constructor=dict)

#df = pd.DataFrame(xml['Trials_downloaded_from_ICTRP']['Trial'])

#This now takes the CSV posted by the ICTRP as an input from here: https://www.who.int/ictrp/en/
#One bit of pre-data management to do before loading: I take the "Date enrollment" field and in
#Excel I format it as a date, otherwise they are a pain to import due to differeing formats
#I then save it as an excel spreadsheet from the original CSV.

#df = pd.read_excel('this_weeks_data/COVID19-web_12aug2020.xlsx', dtype={'Phase': str})
df = pd.read_csv('this_weeks_data/COVID19-web_11nov2020.csv', dtype={'Phase': str})

#UPDATE THESE WITH EACH RUN
prior_extract_date = date(2020,9,4)
this_extract_date = date(2020,11,11)

def enrollment_dates(x):
    format_1 = re.compile(r"\d{4}/\d{2}/\d{2}")
    format_2 = re.compile(r"\d{2}/\d{2}/\d{4}")
    format_3 = re.compile(r"\d{4}-\d{2}-\d{2}")
    format_4 = re.compile(r"\d{2}-\d{2}-\d{4}")
    if isinstance(x, str) and x[0].isalpha():
        return pd.to_datetime(x)
    elif isinstance(x, str) and bool(re.match(format_1, x)):
        return pd.to_datetime(x, format='%Y/%m/%d')
    elif isinstance(x, str) and bool(re.match(format_2, x)):
        return pd.to_datetime(x, format='%d/%m/%Y')
    elif isinstance(x, str) and bool(re.match(format_3, x)):
        return pd.to_datetime(x, format='%Y-%m-%d')
    elif isinstance(x, str) and bool(re.match(format_4, x)):
        return pd.to_datetime(x, format='%d-%m-%Y')
    else:
        return pd.to_datetime(x, errors='coerce')

def fix_date(x):
    if isinstance(x,str):
        return x
    else:
        x = pd.to_datetime(x).date()
        return x

#This is fixes for known broken enrollement dates    
known_errors= {
    'IRCT20200310046736N1': ['2641-06-14', '2020-04-01'],
    'EUCTR2020-001909-22-FR': ['nan', '2020-04-29']
}

def fix_errors(fix_dict, df):
    for a, b in fix_dict.items():
        ind = df[df.TrialID == a].index.values[0]
        if str(df.at[ind, 'Date enrollement']) == str(b[0]):
            df.at[ind, 'Date enrollement'] = b[1]
        else:
            print(f'Original Value Did not Match for {a}')
    return df
    
def d_c(x):
    return x[x.TrialID.duplicated()]


# -

df = fix_errors(known_errors, df)

# +
df['Date enrollement'] = df['Date enrollement'].apply(enrollment_dates)

df['Date registration'] = pd.to_datetime(df['Date registration3'], format='%Y%m%d')

#df['Date registration'] = pd.to_datetime(df['Date registration'], format='%d/%m/%Y')
#df['Date enrollement'] = pd.to_datetime(df['Date enrollement'], errors='coerce')

# +
#Extracting target enrollment
size = df['Target size'].tolist()

extracted_size = []
for s in size:
    try:
        s = int(s)
        extracted_size.append(s)
    except (ValueError, TypeError):
        if not s or pd.isnull(s):
            extracted_size.append('Not Available')
        elif isinstance(s,str):
            digits = []
            nums = re.findall(r':\d{1,10};',s)
            for n in nums:
                digits.append(int(n.replace(':','').replace(';','')))
            extracted_size.append(sum(digits))
        else:
            print(type(s))

df['target_enrollment'] = extracted_size

#Creating retrospective registration
df['retrospective_registration'] = np.where(df['Date registration'] > df['Date enrollement'], True, False)

# +
#Taking only what we need right now

cols = ['TrialID', 'Source Register', 'Date registration', 'Date enrollement', 'retrospective_registration', 
        'Primary sponsor', 'Recruitment Status', 'Phase', 'Study type', 'Countries', 'Public title', 'Acronym',
        'Intervention', 'target_enrollment', 'web address', 'results yes no', 'results url link', 
        'Last Refreshed on']

df_cond = df[cols].reset_index(drop=True)

#renaming columns to match old format so I don't have to change them throughout
df_cond.columns = ['TrialID', 'Source_Register', 'Date_registration', 'Date_enrollement', 
                   'retrospective_registration', 'Primary_sponsor', 'Recruitment_Status', 'Phase', 'Study_type', 
                   'Countries', 'Public_title', 'Acronym', 'Intervention', 'target_enrollment', 'web_address', 
                   'has_results', 'results_url_link', 'Last_Refreshed_on']

print(f'The ICTRP shows {len(df_cond)} trials as of {this_extract_date}')
# -

#POINT THIS TO LAST WEEK'S PROCESSED DATA 
last_weeks_trials = pd.read_csv('last_weeks_data/trial_list_2020-09-04.csv').drop_duplicates()

#Check for which registries we are dealing with:
df_cond.Source_Register.value_counts()

# When working with data straight from XML we need to do some tedious tidying up of dates because of different formats. They do not parse correctly by default in Pandas. They are standardized, however, in the ICTRP spreadsheet so I have removed this code for now. It is archived in old commits to the GitHub repo for future refrence if needed.

# +
#lets get rid of trials from before 2020 for now

pre_2020 = len(df_cond[df_cond['Date_registration'] < pd.Timestamp(2020,1,1)])

print(f'Excluded {pre_2020} trials from before 2020')

df_cond_rec = df_cond[df_cond['Date_registration'] >= pd.Timestamp(2020,1,1)].reset_index(drop=True)

print(f'{len(df_cond_rec)} trials remain')
# -

# This code removes trials that indicate that they never started. This is done on the Chinese registry through specific language in the Title. Trials from ClinicalTrials.gov are indicated by the `Withdrawn` trial status. 
#
# This will be expanded moving forward to account for the unique terminology used by other registries as necessary moving forward.

# +
#Removing cancelled/withdrawn trials for what registries we have to date

cancelled_trials = len(df_cond_rec[(df_cond_rec['Public_title'].str.contains('Cancelled')) | 
                                   (df_cond_rec['Public_title'].str.contains('Retracted due to'))])

print(f'Excluded {cancelled_trials} cancelled trials with no enrollment via Title')

#Now lets get rid of registrations we know don't belong from manual review, cross-referenced with 
#ClinicalTrials.gov list of "Withdrawn" trials. See 'manual_data.xlsx' for reasons for exclusion

exclusions = pd.read_excel('manual_data.xlsx', sheet_name = 'manual removals')

exclude = exclusions.trial_id.to_list()

#This gets all the COVID trials currently listed as "Withdrawn" on ClinicalTrials.gov

ct_gov_withdrawn = pd.read_csv('https://clinicaltrials.gov/ct2/results/download_fields??cond=COVID-19&term=&type=&rslt=&recrs=i&age_v=&gndr=&intr=&titles=&outc=&spons=&lead=&id=&cntry=&state=&city=&dist=&locn=&rsub=&strd_s=&strd_e=&prcd_s=&prcd_e=&sfpd_s=&sfpd_e=&rfpd_s=&rfpd_e=&lupd_s=&lupd_e=&sort=&down_count=1000&down_flds=all&down_fmt=csv')

ct_w = ct_gov_withdrawn['NCT Number'].to_list()

if set(ct_w) - set(exclude):
    print(f'Add new Withdrawn Trials from ClincialTrials.gov: {set(ct_w) - set(exclude)}')
else:
    print('All Withdrawn trials accounted for')

# +
print(f'Excluded {len(exclude)} non-COVID trials screened through manual review')

df_cond_nc = df_cond_rec[~((df_cond_rec['Public_title'].str.contains('Cancelled')) | 
                           (df_cond_rec['Public_title'].str.contains('Retracted due to'))) & 
                         ~(df_cond_rec['Recruitment_Status'] == "Withdrawn") &
                         ~(df_cond_rec['TrialID'].isin(exclude))].reset_index(drop=True)

print(f'{len(df_cond_nc)} trials remain')
# -

# As a general rule, simply for quality and ease of use, we will usually default to a ClinicalTrials.gov record over another type of registration in instances of cross-registration. The ICTRP alerts users to trials with known cross-registrations in the "Bridge" field of their dataset and only lists 1 registration per trial (but does not tell you the cross-registered trial IDs). These aren't comprehensive but are a good start. We can manually check and catalogue these. However we will want to replace some of these when the "Parent" registry is another registry. The first step is to remove the duplicate or replaced entries, then we will add the ClinicalTrials.gov (or another) version of the registry entry  back into the dataset when we append known trials. We will then join in a new column listing the known cross-registered trial ids.

# +
c_reg = pd.read_excel('manual_data.xlsx', sheet_name = 'cross registrations')
replace_ids = c_reg.id_to_replace.tolist()

replaced = df_cond_nc[df_cond_nc.TrialID.isin(replace_ids)]
print(f'{len(replaced)} known cross registrations will be replaced')

df_cond_nc = df_cond_nc[~(df_cond_nc.TrialID.isin(replace_ids))].reset_index(drop=True)


# +
#Function for checking changes

def trial_diffs(new=True):
    df = df_cond_nc.merge(last_weeks_trials['trialid'], left_on = 'TrialID', right_on = 'trialid', how='outer', indicator=True)
    if new:
        new = df[(df['_merge'] == 'left_only')]
        return new['TrialID'].tolist()
    else:
        dropped = df[(df['_merge'] == 'right_only')]
        return dropped['trialid'].tolist()


# +
additions = pd.read_excel('manual_data.xlsx', sheet_name = 'additional_trials').drop('from', 
                                                                                     axis=1).reset_index(drop=True)

print(f'There are approximately {len(trial_diffs(new=True))} new trials')

added = additions.TrialID.tolist()

print(f'The following trials were removed since the last time and were manually checked to confirm:')
print(list(set(trial_diffs(new=False)) - set(added) - set(replace_ids)))
# -

# Now we just need to take a quick look at trials that came and went since the last update. We can add in any additional trials that we know about that are not accounted for in the ICTRP database.

# +
#Here we check to see if any of our manual additions have been added to the dataset

for t in added:
    if t in df_cond_nc.TrialID.tolist():
        print(f'{t} is already in the data')
    else:
        continue

# +
#These are trials we know about that are not showing up in the ICTRP data pull 
#or are re-added as cross-registrations

print(f'An additional {len(additions)} known trials, or preferred cross registrations were added to the data')

print(added)

df_cond_all = df_cond_nc.append(additions)
df_cond_all['Date_enrollement'] = df_cond_all['Date_enrollement'].apply(enrollment_dates)

print(f'The final dataset is {len(df_cond_all)} trials')

# +
#This ensures our check for retrospective registration is accurate w/r/t cross-registrations

c_r_comp_dates = c_reg[['trial_id_keep', 'cross_reg_date']].groupby('trial_id_keep', as_index=False).min()
c_r_merged = c_r_comp_dates.merge(df_cond_nc[['TrialID', 'Date_registration', 'Date_enrollement']], 
                                 left_on='trial_id_keep', right_on='TrialID', how='left')
c_r_merged['earliest_reg'] = c_r_merged[['cross_reg_date', 'Date_registration']].min(axis=1)
pre_reg = c_r_merged[c_r_merged.TrialID.notnull() & (c_r_merged.earliest_reg <= c_r_merged.Date_enrollement)].trial_id_keep.to_list()

ret_reg = c_r_merged[c_r_merged.TrialID.notnull() & ~(c_r_merged.earliest_reg <= c_r_merged.Date_enrollement)].trial_id_keep.to_list()
ret_reg

for index, row in df_cond_all.iterrows():
    if row.TrialID in pre_reg:
        df_cond_all.at[index, 'retrospective_registration'] = True
    elif row.TrialID in ret_reg:
        df_cond_all.at[index, 'retrospective_registration'] = False

# +
#finally, add cross-registration field

df_cond_all = df_cond_all.merge(c_reg[['trial_id_keep', 'additional_ids']].drop_duplicates(), 
                              left_on='TrialID', 
                              right_on='trial_id_keep', 
                              how='left').drop('trial_id_keep', axis=1).rename(columns=
                                                                               {'additional_ids':
                                                                                'cross_registrations'}
                                                                              ).reset_index(drop=True)

df_cond_all['cross_registrations'] = df_cond_all['cross_registrations'].fillna('None')


# -

# Normalisation and data cleaning of all fields. This will be expanded each update as more trials get added and more registries start to add trials with their own idiosyncratic data categories. 

# +
#A small function to help quickly check the contents of various columns

def check_fields(field):
    return df_cond_all[field].unique()

#check_fields('Phase')

#Check fields for new unique values that require normalisation
#for x in check_fields('Countries'):
#    print(x)


# +
#Data cleaning various fields. 
#One important thing we have to do is make sure there are no nulls or else the data won't properly load onto the website

#semi-colons in the intervention field mess with CSV
df_cond_all['Intervention'] = df_cond_all['Intervention'].str.replace(';', '')

#Study Type
obv_replace = ['Observational [Patient Registry]', 'observational', 'Observational Study']
int_replace = ['interventional', 'Interventional clinical trial of medicinal product', 'Treatment', 
               'INTERVENTIONAL', 'Intervention', 'Interventional Study', 'PMS']
hs_replace = ['Health services reaserch', 'Health Services reaserch', 'Health Services Research']

df_cond_all['Study_type'] = (df_cond_all['Study_type'].str.replace(' study', '')
                             .replace(obv_replace, 'Observational').replace(int_replace, 'Interventional')
                             .replace('Epidemilogical research', 'Epidemiological research')
                             .replace(hs_replace, 'Health services research')
                             .replace('Others,meta-analysis etc', 'Other'))

#phase
df_cond_all['Phase'] = df_cond_all['Phase'].fillna('Not Applicable')
na = ['0', 'Retrospective study', 'Not applicable', 'New Treatment Measure Clinical Study', 'Not selected', 
      'Phase 0', 'Diagnostic New Technique Clincal Study', '0 (exploratory trials)', 'Not Specified']
p1 = ['1', 'Early Phase 1', 'I', 'Phase-1', 'Phase I']
p12 = ['1-2', '2020-02-01 00:00:00', 'Phase I/II', 'Phase 1 / Phase 2', 'Phase 1/ Phase 2',
       'Human pharmacology (Phase I): yes\nTherapeutic exploratory (Phase II): yes\nTherapeutic confirmatory - (Phase III): no\nTherapeutic use (Phase IV): no\n']
p2 = ['2', 'II', 'Phase II', 'IIb', 'Phase-2', 'Phase2',
      'Human pharmacology (Phase I): no\nTherapeutic exploratory (Phase II): yes\nTherapeutic confirmatory - (Phase III): no\nTherapeutic use (Phase IV): no\n']
p23 = ['Phase II/III', '2020-03-02 00:00:00', 'II-III', 'Phase 2 / Phase 3', 'Phase 2/ Phase 3', '2-3',
       'Human pharmacology (Phase I): no\nTherapeutic exploratory (Phase II): yes\nTherapeutic confirmatory - (Phase III): yes\nTherapeutic use (Phase IV): no\n']
p3 = ['3', 'Phase III', 'Phase-3', 'III',
      'Human pharmacology (Phase I): no\nTherapeutic exploratory (Phase II): no\nTherapeutic confirmatory - (Phase III): yes\nTherapeutic use (Phase IV): no\n']
p34 = ['Phase 3/ Phase 4', 'Phase III/IV',
       'Human pharmacology (Phase I): no\nTherapeutic exploratory (Phase II): no\nTherapeutic confirmatory - (Phase III): yes\nTherapeutic use (Phase IV): yes\n']
p4 = ['4', 'IV', 'Post Marketing Surveillance', 'Phase IV', 'PMS',
      'Human pharmacology (Phase I): no\nTherapeutic exploratory (Phase II): no\nTherapeutic confirmatory - (Phase III): no\nTherapeutic use (Phase IV): yes\n']

df_cond_all['Phase'] = (df_cond_all['Phase'].replace(na, 'Not Applicable').replace(p1, 'Phase 1')
                        .replace(p12, 'Phase 1/Phase 2').replace(p2, 'Phase 2')
                        .replace(p23, 'Phase 2/Phase 3').replace(p3, 'Phase 3').replace(p34, 'Phase 3/Phase 4')
                        .replace(p4, 'Phase 4'))

#Fixing Observational studies incorrectly given a Phase in ICTRP data
m = ((df_cond_all.Phase.str.contains('Phase')) & (df_cond_all.Study_type == 'Observational'))
df_cond_all['Phase'] = df_cond_all.Phase.where(~m, 'Not Applicable')

#Recruitment Status
df_cond_all['Recruitment_Status'] = df_cond_all['Recruitment_Status'].replace('Not recruiting', 'Not Recruiting')
df_cond_all['Recruitment_Status'] = df_cond_all['Recruitment_Status'].fillna('No Status Given')

#Get rid of messy accents
def norm_names(x):
    if isinstance(x,float):
        return x
    else:
        text = unidecode.unidecode(x)
        normed = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode()
        return normed 
    
df_cond_all['Primary_sponsor'] = df_cond_all.Primary_sponsor.apply(norm_names)
df_cond_all['Primary_sponsor'] = df_cond_all['Primary_sponsor'].replace('NA', 'No Sponsor Name Given')
df_cond_all['Primary_sponsor'] = df_cond_all['Primary_sponsor'].replace('nan', 'No Sponsor Name Given')

# +
#Countries
df_cond_all['Countries'] = df_cond_all['Countries'].fillna('No Country Given').replace('??', 'No Country Given')

china_corr = ['Chian', 'China?', 'Chinese', 'Wuhan', 'Chinaese', 'china', 'Taiwan, Province Of China', 
              "The People's Republic of China"]

country_values = df_cond_all['Countries'].tolist()

new_list = []

for c in country_values:
    country_list = []
    if isinstance(c, float):
        country_list.append('No Sponsor Name Given')
    elif c == 'No Sponsor Name Given':
        country_list.append('No Sponsor Name Given')
    elif c in china_corr:
        country_list.append('China')
    elif c in ['Iran (Islamic Republic of)', 'Iran, Islamic Republic of']:
        country_list.append('Iran')
    elif c in ['Viet nam', 'Viet Nam']:
        country_list.append('Vietnam')
    elif c in ['Korea, Republic of', 'Korea, Republic Of', 'KOREA'] :
        country_list.append('South Korea')
    elif c in ['USA', 'United States of America', 'U.S.']:
        country_list.append('United States')
    elif c == 'Japan,Asia(except Japan),Australia,Europe':
        country_list = ['Japan', 'Australia', 'Asia', 'Europe']
    elif c == 'Japan,Asia(except Japan),North America,South America,Australia,Europe,Africa':
        country_list = ['Japan, Asia(except Japan), North America, South America, Australia, Europe, Africa']
    elif c == 'The Netherlands':
        country_list.append('Netherlands')
    elif c == 'England':
        country_list.append('United Kingdom')
    elif c == 'Japan,North America':
        country_list = ['Japan', 'North America']
    elif c == 'Czechia':
        country_list.append('Czech Republic')
    elif c == 'ASIA':
        country_list.append('Asia')
    elif c == 'EUROPE':
        country_list.append('Europe')
    elif c == 'MALAYSIA':
        country_list.append('Malaysia')
    elif c in ['Congo', 'Congo, Democratic Republic', 'Congo, The Democratic Republic of the']:
        country_list.append('Democratic Republic of Congo')
    elif c in ["C√¥te D'Ivoire", 'Cote Divoire']:
        country_list.append("Cote d'Ivoire")
    elif c in ['Türkiye', 'TÃ¼rkiye']:
        country_list.append('Turkey')
    elif c == 'SOUTH AMERICA':
        country_list.append('South America')
    elif c == 'AFRICA':
        country_list.append('Africa')
    elif ';' in c:
        c_list = c.split(';')
        unique_values = list(set(c_list))
        for v in unique_values:
            if v in china_corr:
                country_list.append('China')
            elif v in ['Iran (Islamic Republic of)', 'Iran, Islamic Republic of']:
                country_list.append('Iran')
            elif v in ['Korea, Republic of', 'Korea, Republic Of', 'KOREA']:
                country_list.append('South Korea')
            elif v in ['Viet nam', 'Viet Nam']:
                country_list.append('Vietnam')
            elif v in ['USA', 'United States of America']:
                country_list.append('United States')
            elif v == 'The Netherlands':
                country_list.append('Netherlands')
            elif v == 'England':
                country_list.append('United Kingdom')
            elif v == 'Czechia':
                country_list.append('Czech Republic')
            elif v == 'ASIA':
                country_list.append('Asia')
            elif v == 'EUROPE':
                country_list.append('Europe')
            elif v == 'MALAYSIA':
                country_list.append('Malaysia')
            elif v in ['Congo', 'Congo, Democratic Republic', 'Congo, The Democratic Republic of the']:
                country_list.append('Democratic Republic of Congo')
            elif v in ["C√¥te D'Ivoire", 'Cote Divoire']:
                country_list.append("Cote d'Ivoire")
            elif v in ['Türkiye', 'TÃ¼rkiye']:
                country_list.append('Turkey')
            elif v == 'SOUTH AMERICA':
                country_list.append('South America')
            elif v == 'AFRICA':
                country_list.append('Africa')
            else:
                country_list.append(v)
    else:
        country_list.append(c.strip())
    new_list.append(', '.join(country_list))

df_cond_all['Countries'] = new_list
# -

# Last space for manual intervention. This will include manual normalisation of new names, any updates to the normalisation schedule from the last update, and updating manually-coded intervention type data.

# +
#Normalizing sponsor names
#Run this cell, updating the spon_norm csv you are loading after manual adjusting
#until you get the 'All sponsor names normalized' to print

spon_norm = pd.read_excel('manual_data.xlsx', sheet_name = 'sponsor')

df_cond_norm = df_cond_all.merge(spon_norm, left_on = 'Primary_sponsor', right_on ='unique_spon_names', how='left')
df_cond_norm = df_cond_norm.drop('unique_spon_names', axis=1)

new_unique_spon_names = (df_cond_norm[df_cond_norm['normed_spon_names'].isna()][['Primary_sponsor', 'TrialID']]
                        .groupby('Primary_sponsor').count())

if len(new_unique_spon_names) > 0:
    new_unique_spon_names.to_csv('to_norm.csv')
    print('Update the normalisation schedule and rerun')
else:
    print('All sponsor names normalized')

# +
#Integrating intervention type data
#Once again, run to bring in the old int-type data, islolate the new ones, update, and rerun until
#producing the all-clear message

int_type = pd.read_excel('manual_data.xlsx', sheet_name = 'intervention')
df_cond_int = df_cond_norm.merge(int_type[['trial_id', 'study_category',
                                           'intervention', 'intervention_list']], 
                                 left_on = 'TrialID', right_on = 'trial_id', how='left')

df_cond_int = df_cond_int.drop('trial_id', axis=1)

new_int_trials = df_cond_int[(df_cond_int['study_category'].isna()) | (df_cond_int['intervention'].isna())]

if len(new_int_trials) > 0:
    new_int_trials[['TrialID', 'Public_title', 'Intervention', 'study_category', 
                    'intervention', 'intervention_list']].to_csv('int_to_assess.csv')
    print('Update the intervention type assessments and rerun')
else:
    print('All intervention types matched')
    df_cond_int = df_cond_int.drop('Intervention', axis=1).reset_index(drop=True)

# +
#Can use this cell to output counts of values from columns

treatments = df_cond_int[df_cond_int.study_category == 'Drug']['study_category'].tolist()
countries = df_cond_int.Countries.to_list()

def var_counts(var_list, split_char, lower=False):
    final_list = []
    for v in var_list:
        t_list = v.split(split_char)
        for l in t_list:
            if lower:
                final_list.append(l.lower().strip())
            else:
                final_list.append(l.strip())
    return Counter(final_list)


# +
comp_dates = pd.read_excel('manual_data.xlsx', sheet_name = 'Completion Dates')
df_comp_dates = df_cond_int.merge(comp_dates, 
                                  left_on='TrialID', right_on='trialid', 
                                  how='left', indicator=True).drop('trialid', axis=1)

print('These trials are missing completion date data:')
print(df_comp_dates[df_comp_dates['_merge'] == 'left_only'].TrialID.to_list())

df_comp_dates = df_comp_dates.drop('_merge', axis=1).reset_index(drop=True)
    
df_comp_dates['primary_completion_date'] = (pd.to_datetime(df_comp_dates['primary_completion_date'], 
                                                          errors='coerce', 
                                                          format='%Y-%m-%d')
                                            .fillna('Not Available').apply(fix_date))

df_comp_dates['full_completion_date'] = (pd.to_datetime(df_comp_dates['full_completion_date'], 
                                                          errors='coerce', 
                                                          format='%Y-%m-%d')
                                            .fillna('Not Available').apply(fix_date))

# +
#check for any results on ICTRP

results_checked = ['NCT04323592', 'JPRN-UMIN000040520', 'NCT04410159', 'NCT04422561', 'NCT04491994', 
                   'ACTRN12620000869976', 'KCT0005226', 'NCT04280705', 'NCT04343261', 'NCT04523831', 
                   'NCT04491240', 'NCT04343092', 'NCT04425850', 'NCT04542694', 'JPRN-UMIN000040405', 
                   'JPRN-jRCTs041190120']

ictrp_results = df_comp_dates[(df_comp_dates.has_results.notnull()) | (df_comp_dates.has_results.notnull())]

if len(ictrp_results) > 0:
    print(f'There are {len(ictrp_results) - len(results_checked)} results to check for: {list(set(ictrp_results.TrialID.tolist()) - set(results_checked))}')
else:
    print('There are no results to check')

#If results cross-check with results already collected in 'manual_data' excel file and add any new trial results.

# +
results = pd.read_excel('manual_data.xlsx', sheet_name = 'Results')
df_results = df_comp_dates.merge(results, 
                                 left_on='TrialID', 
                                 right_on='trialid', 
                                 how='left').drop('trialid', axis=1)

df_results['results_link'] = df_results['results_link'].fillna('No Results')
df_results['results_type'] = df_results['results_type'].fillna('No Results')

df_results['results_publication_date'] = (pd.to_datetime(df_results['results_publication_date'], 
                                                          errors='coerce', 
                                                          format='%Y-%m-%d')
                                            .fillna('No Results').apply(fix_date))

# +
#Final organising

col_names = []

for col in list(df_results.columns):
    col_names.append(col.lower())
    
df_results.columns = col_names

reorder = ['trialid', 'source_register', 'date_registration', 'date_enrollement', 'retrospective_registration', 
           'normed_spon_names', 'recruitment_status', 'phase', 'study_type', 'countries', 'public_title', 
           'acronym', 'study_category', 'intervention', 'intervention_list', 'target_enrollment', 
           'primary_completion_date', 'full_completion_date', 'web_address', 'results_type', 
           'results_publication_date', 'results_link', 'last_refreshed_on', 'cross_registrations']

df_final = df_results[reorder].reset_index(drop=True).drop_duplicates().reset_index()
df_final['acronym'] = df_final.acronym.fillna('')
df_final['last_refreshed_on'] = pd.to_datetime(df_final['last_refreshed_on'])
# -

#Checking for any null values
df_final[df_final.isna().any(axis=1)]

#Quick look at the data
df_final.head(10)

# +
#Export final dataset
df_final.to_csv(f'processed_data_sets/trial_list_{this_extract_date}.csv', index=False)

df_final.to_csv(f'tableau_data/current_data.csv', index=False)

# +
just_results = df_final[df_final.results_type != 'No Results']

results_total = len(just_results)

print(f'There are {results_total} trials with results')

# +
#Export json for website
import json
with open("website_data/trials_latest.json", "w") as f:
    json.dump({"data": df_final.astype(str).values.tolist()}, f, indent=2)

with open("website_data/results_latest.json", "w") as f:
    json.dump({"data": just_results.astype(str).values.tolist()}, f, indent=2)    
# -
# # Overall Trend in Registered Trials Graph

# +
just_reg = df_final[['trialid', 'date_registration']].reset_index(drop=True)
#just_reg = mar18[['trialid', 'date_registration']].reset_index(drop=True)
#just_reg['date_registration'] = pd.to_datetime(just_reg['date_registration'], format='%d/%m/%Y')

#catch old registrations that were expanded to include COVID, we can get rid of these for now
just_reg = just_reg[just_reg['date_registration'] >= pd.Timestamp(2020,1,1)].reset_index(drop=True)


# +
just_reg.index = just_reg['date_registration']


grouped = just_reg.resample('W').count()
cumsum = grouped.cumsum()

# +
import matplotlib.pyplot as plt

labels = []

for x in list(grouped.index):
    labels.append(str(x.date()))

x_pos = [i for i, _ in enumerate(labels)]

#fig, ax = plt.subplots(figsize=(10,5), dpi = 300)

#l1 = plt.plot(x_pos, grouped['trialid'], marker = 'o')
#l2 = plt.plot(x_pos, cumsum['trialid'], marker = 'o')

#for i, j in zip(x_pos[1:], grouped['trialid'].tolist()[1:]):
#    ax.annotate(str(j), (i,j), xytext = (i-.1, j-50))

#for i, j in zip(x_pos, cumsum['trialid']):
#    ax.annotate(str(j), (i,j), xytext = (i-.2, j+25))
    

gr = grouped['trialid'].to_list()
cs = cumsum['trialid'].to_list()

#plt.xticks(x_pos, labels, rotation=45, fontsize=8)
#plt.ylim(-50, 2500)
#plt.xlabel('Week Ending Date')
#plt.ylabel('Registered Trials')
#plt.title('Registered COVID-19 Trials by Week on the ICTRP')
#plt.legend(('New Trials', 'Cumulative Trials'), loc=2)
#plt.savefig(f'trial_count_{last_extract_date}.png')
#plt.show()
# +
import plotly.graph_objects as go

labels = []

for x in list(grouped.index):
    labels.append(str(x.date()))

fig = go.Figure()

fig.add_trace(go.Scatter(x=labels[:-1], y=grouped['trialid'][:-1], fill=None, name='New Trials'))

fig.add_trace(go.Scatter(x=labels[:-1], y=cumsum['trialid'][:-1], fill=None, name='Cumulative Trials'))

fig.update_layout(title={'text': 'Registered COVID-19 Trials by Week', 'xanchor': 'center', 'x': 0.5}, 
                  xaxis_title='Week Ending Date',
                  yaxis_title='Registered Trials',
                  legend = dict(x=0, y=1, traceorder='normal', bgcolor='rgba(0,0,0,0)'))



fig.show()
fig.write_html("html_figures/registered trials.html")
# -


int_types = df_final.study_category.value_counts()
int_types

# +
treatment_dict = dict(drugs = int_types['Drug'] + int_types['Drug (Chemoprophylaxis)'], 
                      atmp = int_types['ATMP'], 
                      clinical_char = (int_types['Clinical Presentation'] + int_types['Diagnostics'] + 
                                       int_types['Prognosis'] + int_types['Clinical Presentation (Epidemiology)']),
                      drug_other_combo = (int_types['Drug (+ ATMP + Other (renal))'] + int_types['Drug (+ ATMP)'] + 
                                          int_types['ATMP (+ Drug)'] + int_types['Drug (+ Chemoprophylaxis)']),
                      supp = int_types['Supplement'],
                      geno = int_types['Genomics'],
                      th = int_types['Telehealth'],
                      pro = int_types['Procedure'],
                      tm = int_types[[s.startswith('Traditional Medicine') for s in int_types.index]].sum(),
                      other = (int_types[[s.startswith('Other') for s in int_types.index]].sum() 
                               + int_types['Health System'])
                     )

fig = go.Figure(go.Bar(
            x=list(treatment_dict.values()),
            y=['Drugs', 'ATMP', 'Clinical Characteristics', 'Multiple Therapies', 'Supplement', 'Genomics', 
               'Telehealth', 'Procedure', 'Traditional Medicine', 'Other'],
            orientation='h'))

fig.update_layout(title={'text': 'Intervention Type of Registered Trials', 'xanchor': 'center', 'x': 0.5}, 
                  xaxis_title='Number of Trials')

fig.show()
fig.write_html('html_figures/int_bar.html')
# +
fig = go.Figure(go.Bar(
            x=df_final.source_register.value_counts().values,
            y=df_final.source_register.value_counts().index,
            orientation='h'))

fig.update_layout(title={'text': 'Registered Studies by Trial Registry', 'xanchor': 'center', 'x': 0.55}, 
                  xaxis_title='Number of Studies Registered',
                  yaxis=dict(autorange="reversed"))

fig.show()
fig.write_html('html_figures/registries_bar.html')

# +
treatments = df_final[((df_final.study_category.str.contains('Drug')) | (df_final.study_category.str.contains('ATMP'))) & ~(df_final.study_category.str.contains('Traditional Medicine'))]['intervention_list'].tolist()
common_treatments = pd.DataFrame(var_counts(treatments, ';', lower=True).most_common())
common_treatments.columns = ['treatment', 'trial_count']

fig = go.Figure(go.Bar(
            x=common_treatments[common_treatments.trial_count >= 15]['trial_count'],
            y=common_treatments[common_treatments.trial_count >= 15]['treatment'],
            orientation='h'))

fig.update_layout(title={'text': 'Most Commonly Studied Drugs & ATMPs (n>=15)', 'xanchor': 'center', 'x': 0.5}, 
                  xaxis_title='Number of Studies Registered',
                  yaxis=dict(autorange="reversed", dtick=1))

fig.show()
fig.write_html('html_figures/treatment_bar.html')

# +
countries = df_final.countries.to_list()
most_studies = pd.DataFrame(var_counts(countries, ',', lower=False).most_common())
most_studies.columns = ['country', 'trial_count']

fig = go.Figure(go.Bar(
            x=most_studies[(most_studies.trial_count >= 50) & (most_studies.country != 'No Country Given')]['trial_count'],
            y=most_studies[(most_studies.trial_count >= 50) & (most_studies.country != 'No Country Given')]['country'],
            orientation='h'))

fig.update_layout(title={'text': 'Most Common Study Locations (n>=50)', 'xanchor': 'center', 'x': 0.5}, 
                  xaxis_title='Number of Studies Registered',
                  yaxis=dict(autorange="reversed", dtick=1))

fig.show()
fig.write_html('html_figures/location_bar.html')
# -

