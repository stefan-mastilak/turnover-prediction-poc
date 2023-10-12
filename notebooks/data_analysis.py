# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# -------
# # Turnover prediction for Multi
# ### Dataset analysis - SR-Bank data
# -------
#
# <h2>Columns description</h2> 
# <table>
#     <tr>
#         <td>Kolonne</td>
#         <td>Beskrivelse</td>
#         <td>Name</td>
#         <td>Description</td>
#         <td>Historic value</td>
#         <td>Comment</td>
#     </tr>
#     <tr>
#         <td>Customer and period</td>
#         <td>CUST</td>
#         <td></td>
#         <td></td>
#         <td></td>
#         <td></td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>År-Mnd</td>
#         <td>Data hentet fra siste dag i angitt måned</td>
#         <td>Year-month</td>
#         <td></td>
#         <td></td>
#         <td></td>
#     </tr>
#     <tr>
#         <td>Person</td>
#         <td>Arbeidsgiver nr</td>
#         <td></td>
#         <td>Employer nr</td>
#         <td></td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>PERSON ID</td>
#         <td>Unik ID for person</td>
#         <td>Person ID</td>
#         <td>Uniqe ID for person</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Kjønn</td>
#         <td></td>
#         <td>Gender</td>
#         <td></td>
#         <td>N</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Fødselsdato</td>
#         <td></td>
#         <td>Birth date</td>
#         <td></td>
#         <td>N</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Alder</td>
#         <td></td>
#         <td>Age</td>
#         <td></td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Nasjonalitet</td>
#         <td></td>
#         <td>Nationality</td>
#         <td></td>
#         <td>N</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Sivil status</td>
#         <td></td>
#         <td>Marital status</td>
#         <td></td>
#         <td>N</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Første gang ansatt dato</td>
#         <td>Første gang person ble ansatt i arbeidsgiver</td>
#         <td>First time employed</td>
#         <td>First time employed in employer</td>
#         <td>N</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Sluttdato person</td>
#         <td>Dato for når personen sluttet hos arbeidsgiver</td>
#         <td>Stop date person</td>
#         <td>Stop date for person in the employer</td>
#         <td>N</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Sluttårsak kode</td>
#         <td>Årsak til at personen sluttet</td>
#         <td>Termination reason code</td>
#         <td></td>
#         <td>N</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Sluttårsak kode navn</td>
#         <td>Årsak til at personen sluttet</td>
#         <td>Termination reason description</td>
#         <td></td>
#         <td>N</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Adresse 1</td>
#         <td>Bostedsadresse i angitt periode</td>
#         <td>Address 1</td>
#         <td>Home address in given period</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Adresse 2</td>
#         <td>Bostedsadresse i angitt periode</td>
#         <td>Address 2</td>
#         <td>Home address in given period</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Adresse 3</td>
#         <td>Bostedsadresse i angitt periode</td>
#         <td>Address 3</td>
#         <td>Home address in given period</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Postnr ved lønnskjøring</td>
#         <td>Bostedsadresse i angitt periode</td>
#         <td>Postal code</td>
#         <td>Home address in given period</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Poststed ved lønnskjøring</td>
#         <td>Bostedsadresse i angitt periode</td>
#         <td>Post office</td>
#         <td>Home address in given period</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td>Employment</td>
#         <td>ARBEIDSFORHOLD ID</td>
#         <td>Unik ID for arbeidsforhold</td>
#         <td>Employment ID</td>
#         <td>Uniqe ID for employment</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Arbfnr</td>
#         <td>Arbeidsforholdsnummer</td>
#         <td>Employment number</td>
#         <td></td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Enhet nr</td>
#         <td>Org.enhet</td>
#         <td>Org. unit</td>
#         <td></td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Lokasjon Post nr</td>
#         <td>Lokalisering av Org.enhet</td>
#         <td>Org. unit postal code</td>
#         <td>Org. unit address in given periode</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Lokasjon Post navn</td>
#         <td>Lokalisering av Org.enhet</td>
#         <td>Org. unit post office</td>
#         <td>Org. unit address in given periode</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Lokasjon Kommune nr</td>
#         <td>Lokalisering av Org.enhet</td>
#         <td>Org. unit municipality nr</td>
#         <td>Org. unit address in given periode</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Lokasjon Kommune navn</td>
#         <td>Lokalisering av Org.enhet</td>
#         <td>Org. unit municipality name</td>
#         <td>Org. unit address in given periode</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Startdato</td>
#         <td>Startdato for arbeidsforholdet</td>
#         <td>Start date employment</td>
#         <td></td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Gjelder fra</td>
#         <td>Startdato for gjeldene verdier</td>
#         <td>Vailid from</td>
#         <td>Start date for the given values on the employment record</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Gjelder til</td>
#         <td>Stoppdato for gjeldene verdier</td>
#         <td>Vailid to</td>
#         <td>Stop date for the given values on the employment record</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Sluttdato</td>
#         <td>Sluttdato for arbeidsforholdet</td>
#         <td>End date employment</td>
#         <td>Actual end of employment</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Ansettelsesform (A-mel.)</td>
#         <td>Innrapportering a-melding (samme kodeverk for alle kunder)</td>
#         <td>Form of employment (A-mel)</td>
#         <td>General codes across all customers - permanent / temporary</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Arbeidsforhold type (A-mel.)</td>
#         <td>Innrapportering a-melding (samme kodeverk for alle kunder)</td>
#         <td>Type of employment (A-mel)</td>
#         <td>General codes across all customers - Ordinary / freelance</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Arbeidstidsordning (A-mel.)</td>
#         <td>Innrapportering a-melding (samme kodeverk for alle kunder)</td>
#         <td>Working time arrangement (A-mel)</td>
#         <td>General codes across all customers</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Sluttårsak (A-mel.)</td>
#         <td>Innrapportering a-melding (samme kodeverk for alle kunder)</td>
#         <td>Termination reason (A-mel)</td>
#         <td>General codes across all customers -</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Ansattform</td>
#         <td>Kodeverk for kunde</td>
#         <td>Employment type code</td>
#         <td>Customer spesific codes</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Ansattform tekst</td>
#         <td>Kodeverk for kunde</td>
#         <td>Employment type name</td>
#         <td>Customer spesific codes</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Ansattandel</td>
#         <td>Stillingsandel, tilsattprosent etc.</td>
#         <td>Employed percentage</td>
#         <td></td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Tilstedeprosent</td>
#         <td>Andel tilsted av ansattandel (reduseres f.eks ved permisjon)</td>
#         <td>Present percentage</td>
#         <td></td>
#         <td>J</td>
#         <td>Indikerer permisjon</td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Utbetalingsprosent</td>
#         <td>Andel av lønn som blir utbetalt (reduseres f.eks til 80 ved fødselspermisjon med 80% lønn)</td>
#         <td>Payout percentage</td>
#         <td></td>
#         <td>J</td>
#         <td>Indikerer permisjon</td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Timer per uke</td>
#         <td>Avtalt arbeidstid pr. uke</td>
#         <td>Hours per week</td>
#         <td></td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Stillingsgruppe kode</td>
#         <td>Kategorisering av stilling</td>
#         <td>Position group code</td>
#         <td></td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Stillingsgruppe navn</td>
#         <td>Kategorisering av stilling</td>
#         <td>Position group name</td>
#         <td></td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Stillingskode</td>
#         <td>Kategorisering av stilling</td>
#         <td>Position code</td>
#         <td></td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Stillingskode tekst</td>
#         <td>Kategorisering av stilling</td>
#         <td>Position name</td>
#         <td></td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Stillingsbetegnelse</td>
#         <td>Kategorisering av stilling</td>
#         <td>Position description</td>
#         <td></td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Årsakskode kode</td>
#         <td>Årsak til evt. Endring av arbeidsforholdet</td>
#         <td>Reason for change code</td>
#         <td>Reason for changes on employment record</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Årsakskode beskrivelse</td>
#         <td>Årsak til evt. Endring av arbeidsforholdet</td>
#         <td>Reason for change desc</td>
#         <td>Reason for changes on employment record</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td></td>
#         <td>Årslønn</td>
#         <td>Årslønn i 100% stilling</td>
#         <td>Salary</td>
#         <td>Salary in 100% position (Ansattandel / Employed percentage)</td>
#         <td>J</td>
#         <td></td>
#     </tr>
#     <tr>
#         <td>Other</td>
#         <td>Antall</td>
#         <td>Column just needed for tecnical reasons in export</td>
#     </tr>
# </table>
#

# ----
# ### Imports:

# +
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Option to display all columns of dataframe:
pd.set_option("display.max_columns", None)


# +
def plot_val_counts_bar(df, col, x_lbl, y_lbl, title):
    """
    Plot value counts of specific column of the dataframe
    :param df: dataframe 
    :param col: column name
    :param x_lbl: x label name
    :param y_lbl: y label name
    :param title: figure title
    """
    sns.set(rc = {'figure.figsize':(8,4)})
    sns.set_theme(style="whitegrid")
    ax = sns.countplot(data=df, x=col, order=df[f'{col}'].value_counts().index)
    ax.set_title(title, fontsize=18, pad=20, fontweight='bold')
    ax.set_xlabel(x_lbl, fontsize=15, color='black', labelpad=20)
    ax.set_ylabel(y_lbl, fontsize=15, color='black', labelpad=20)
    plt.tight_layout()
    plt.xticks(rotation=90)
    plt.show()

    
def plot_pie(items, labels, legend, title):
    """
    param items: items to be displayed
    param labels: labels for items to be displayed
    param title: Figure title text 
    """
    plt.figure(figsize=(6,6))
    plt.pie(x=items,
            labels=labels,
            autopct='%1.2f%%',
            textprops={'fontsize':14},
            explode = [0.01 for i in items],
            colors=sns.color_palette('Set2'))
    plt.title(label=title, 
              fontdict={"fontsize":16},
              pad=20)
    plt.legend(legend, loc="upper right")
    plt.show()


# -

# ------------
# ## 1) Data grouping/splitting

# +
# load data from csv:
df = pd.read_csv("../data/poc_data.csv", sep=";", low_memory=False)

active = df[df['Sluttdato'].isnull()]
resigned = df[df['Sluttdato'].notnull()]
# -

active['PERSON ID'].nunique()

resigned['PERSON ID'].nunique()

# #### Data example

df.tail()

# #### Dataframe info

df.info()

# # 2) Columns analysis
#

# --------
# ## CUST
# Customer - constant for SR bank - not relevant for the prediction

# show number of unique fields
df['CUST'].nunique()

df['CUST'].unique()

# ---------------
# ## År-Mnd
# Entry date - not relevant for the prediction

df['År-Mnd'].value_counts()

# -------------
# ## Arbeidsgiver nr
# Employer number - unique for employer - mandatory field

# check if null values
df['Arbeidsgiver nr'].isna().value_counts()

df['Arbeidsgiver nr'].value_counts(dropna=False)

plot_val_counts_bar(df=df, 
                    col='Arbeidsgiver nr', 
                    x_lbl='employer nr', 
                    y_lbl='count', 
                    title='Employer distribution')

# ---------
# ## PERSON ID
# Personal number - mandatory field

# number of unique person ids
df['PERSON ID'].nunique()

# dataframe length in total:
len(df)

# Check if null values
df['PERSON ID'].isna().value_counts()

df['PERSON ID'].value_counts()

# ## Kjønn
# Gender - mandatory field

# check if null values
df['Kjønn'].isna().value_counts()

df['Kjønn'].value_counts(dropna=False)

plot_pie(items=[len(df[df['Kjønn'] == 'Mann']), len(df[df['Kjønn'] == 'Kvinne'])], 
         labels=['Mann', 'Kvinne'], 
         legend=['Man','Woman'],
         title='Gender distribution')

# -----
# ## Fødselsdato
# Birthdate - mandatory field - this field could be used to calculate person's age

# check if null values
df['Fødselsdato'].isna().value_counts()

# birth date format
df['Fødselsdato'].head()

# ---
# ## Alder
# Age - for some reason all values here are 0. But we can derive age from birth date 

# Check uniqueness of values - all zeroes - why? (nevermind, because we have birthdate)
df['Alder'].unique(), df['Alder'].nunique()

# ----
# ## Nationalitet
# Nationality - mandatory field

# check if null values
df['Nationalitet'].isna().value_counts()

df['Nationalitet'].value_counts()

plot_val_counts_bar(df, 'Nationalitet', 'nationality', 'count', 'Nationality distribution')

# ----
# ## Sivilstatus
# Marital status:
# * Gift/registrert partner - Married/registered partner
# * Samboer - Cohabitant
# * Ugift - Not engaged
# * Enke/enkemann - Widow/widower

df['Sivilstatus'].value_counts(dropna=False)

plot_val_counts_bar(df, col='Sivilstatus', x_lbl='status', y_lbl='count', 
                    title='Marital status distribution')

# ----
# ## Første gang ansatt dato
# First time employed - mandatory field - date field -first time employed in employer

df['Første gang ansatt dato'].isna().value_counts()

df['Første gang ansatt dato'].head()

# ----
# ## Sluttdato person
# Stop date - stop date of the person in employeer

df['Sluttdato person'].value_counts(dropna=False)

# ---
# ## Sluttårsak kode
# Termination reason code - codes are related to the 'Sluttårsak kode navn' field

df['Sluttårsak kode'].value_counts(dropna=False)

# ---
# ## Sluttårsak kode navn
# Termination reason description:
# * Sluttet etter eget ønske - Ended at will   
# * Alderspensjon - Retirement pension               
# * Annen sluttårsak - Other end reason               
# * Dødsfall - Death                         
# * Annet selskap i konsernet - Other company in the group        
# * Uførepensjon - Disability pension                    
# * Avskj/Oppsig. arb.t.forh. - Dismissal/Termination working hours          

df['Sluttårsak kode navn'].value_counts(dropna=False)

plot_val_counts_bar(df, 'Sluttårsak kode navn', 'Termination reason', 'count', 'Termination reason distribution')

# ---
# ## Adresse 1
# Address 1 - Home address in given period

df['Adresse 1'].head()

# ---
# ## Adresse 2
# Address 2 - Home address in given period - Not used in majority of cases - probably not relevant for us

# check number of unique vaues
df['Adresse 2'].nunique()

# show unique values
df['Adresse 2'].unique()

# ---
# ## Adresse 3
# Address 3 - Almost not uset at all - not relevant in our case

# check number of unique vaues
df['Adresse 3'].nunique()

# show unique values
df['Adresse 3'].unique()

# ---
# ## Postnummer ved lønnskjøring
# Postal code - Home address in given period

# check number of unique vaues
df['Postnummer ved lønnskjøring'].nunique()

# example:
df['Postnummer ved lønnskjøring'].unique()[:15]

# ------
# ## Poststed ved lønnskjøring
# Post office - Home address in given period

# check datatype
df['Poststed ved lønnskjøring'].dtypes

# check number of unique vaues
df['Poststed ved lønnskjøring'].nunique()

# example
df['Poststed ved lønnskjøring'].unique()[:15]

# ---
# ## ARBEIDSFORHOLD ID
# Employment ID - Uniqe ID for employment

# check number of unique vaues
df['ARBEIDSFORHOLD ID'].nunique()

# example
df['ARBEIDSFORHOLD ID'].unique()

# ---
# ## Arbfnr - Arbeidsforholdsnummer
# Employment number

# check number of unique vaues
df['Arbfnr'].nunique()

# example
df['Arbfnr'].unique()

# ---
# ## Enhet nr
# Organisation Unit number

# example
df['Enhet nr'].unique()[:15]

# ---
# ## Lokasjon Post nr
# Organisation Unit postal code - Org. unit address in given period

# check number of unique vaues:
df['Lokasjon Post nr'].nunique()

# example
df['Lokasjon Post nr'].unique()

# ---
# ## Lokasjon Post navn
# Org. unit post office - Org. unit address in given period

# check number of unique vaues
df['Lokasjon Post navn'].nunique()

# example
df['Lokasjon Post navn'].unique()

# ---
# ## Lokasjon Kommune nr
# Org. unit municipality number - Org. unit address in given period

# example
df['Lokasjon Kommune nr'].unique()

# ---
# ## Lokasjon Kommune navn
# Org. unit municipality name - Org. unit address in given period

df['Lokasjon Kommune navn'].unique()

df['Lokasjon Kommune navn'].value_counts()

# ---
# ## Startdato
# Start date of employment - mandatory field - dtype object (string)

df['Startdato'].isna().value_counts()

# example
df['Startdato'].head()

# Check if null values
df['Startdato'].isna().value_counts()

# ## Gjelder fra
# Valid from - Start date for the given values on the employment record - dtype object (string)

# example
df['Gjelder fra'].head()

# Check if null values
df['Gjelder fra'].isna().value_counts()

# ---
# ## Gjelder til
# Valid to - Stop date for the given values on the employment record - dtype object (string)

df['Gjelder til'].head()

# Check if null values
df['Gjelder til'].isna().value_counts()

# ---
# ## Sluttdato
# End date of employment - Actual end of employment - dtype object (string)

df['Sluttdato'].head()

# Check if null values
df['Sluttdato'].isna().value_counts()

# ---
# ## 	Ansettelsesform (A-mel.)
# Form of employment (A-mel) - General codes across all customers - permanent / temporary

df['Ansettelsesform (A-mel.)'].value_counts(dropna=False)

plot_val_counts_bar(df, 'Ansettelsesform (A-mel.)', '', '', 'Form of employment distribution')

# ---
# ## Arbeidsforhold type (A-mel.)
# Type of employment (A-mel) - General codes across all customers - Ordinary / freelance

df['Arbeidsforhold type (A-mel.)'].value_counts(dropna=False)

plot_val_counts_bar(df, 'Arbeidsforhold type (A-mel.)', '', '', 'Type of employment distribution')

# ---
# ## Arbeidstidsordning (A-mel.)
# Working time arrangement (A-mel) - General codes across all customers

df['Arbeidstidsordning (A-mel.)'].value_counts(dropna=False)

plot_val_counts_bar(df, 'Arbeidstidsordning (A-mel.)', '', '', 'Working time distribution')

# ---
# ## Sluttårsak (A-mel.)
# Termination reason (A-mel) - General codes across all customers
# * Arbeidstaker har sagt opp selv - Employee has resigned himself
# * Endring i organisasjon eller byttet jobb internt - Change in organization or changed job internally
# * Byttet lønnssystem eller regnskapfører - Changed payroll system or accountant
# * Kontrakt, engasjement eller vikariat er utløpt - The contract, engagement or temporary position has expired
# * Arbeidsforhold skulle aldri vært rapportert - Employment conditions should never have been reported

df['Sluttårsak (A-mel.)'].value_counts(dropna=False)

plot_val_counts_bar(df, 'Sluttårsak (A-mel.)', '', '', 'Termination reason distribution')

# ---
# ## Ansattform
# Employment type code - Customer specific codes

df['Ansattform'].value_counts(dropna=False)

# ---
# ## Ansattform tekst
# Employment type name - related to employment type code field

df['Ansattform tekst'].value_counts(dropna=False)

plot_val_counts_bar(df, 'Ansattform tekst', '', '', 'Employment type distribution')

# ---
# ## Ansattandel
# Employed percentage

df['Ansattandel'].value_counts(dropna=False)

plot_val_counts_bar(df, 'Ansattandel', '', '', 'Employed percentage distribution')

# ---
# ## Tilstedeprosent
# Present percentage

df['Tilstedeprosent'].value_counts(dropna=False)

plot_val_counts_bar(df, 'Tilstedeprosent', '', '', 'Present percentage distribution')

# ---
# ## Utbetalingsprosent
# Payout percentage

df['Utbetalingsprosent'].value_counts(dropna=False)

plot_val_counts_bar(df, 'Utbetalingsprosent', '', '', 'Payout percentage distribution')

# ---
# ## Timer per uke
# Hours per week

df['Timer per uke'].value_counts(dropna=False)

plot_val_counts_bar(df, 'Timer per uke', '', '', 'Hours per week distribution')

# ---
# ## Stillingsgruppe kode
# Position group code

df['Stillingsgruppe kode'].value_counts(dropna=False)

plot_val_counts_bar(df, 'Stillingsgruppe kode', '', '', 'Position group code distribution')

# ---
# ## Stillingsgruppe navn
# Position group name - Related to Position group code (seems like duplicate value to it)

df['Stillingsgruppe navn'].value_counts(dropna=False)

plot_val_counts_bar(df, 'Stillingsgruppe navn', '', '', 'Position group name distribution')

# ---
# ## Stillingskode
# Position code

df['Stillingskode'].value_counts(dropna=False)

df['Stillingskode'].nunique()

df['Stillingskode'].unique()

# ---
# ## Stillingskode tekst
# Position name

df['Stillingskode tekst'].value_counts(dropna=False)

df['Stillingskode tekst'].nunique()

df['Stillingskode tekst'].unique()[:15]

# ---
# ## Stillingsbetegnelse
# Position description

df['Stillingsbetegnelse'].value_counts(dropna=False)

df['Stillingsbetegnelse'].nunique()

df['Stillingsbetegnelse'].unique()[:15]

# ---
# ## Årsakskode kode
# Reason for change code

df['Årsakskode kode'].value_counts(dropna=False)

plot_val_counts_bar(df, 'Årsakskode kode', '', '', 'Reason for change code distribution')

# ---
# ## Årsakskode beskrivelse
# Reason for change description

df['Årsakskode beskrivelse'].value_counts(dropna=False)

plot_val_counts_bar(df, 'Årsakskode beskrivelse', '', '', 'Reason for change description distribution')

# ---
# ## Årslønn
# Salary

df['Årslønn']

# ---
# ## Antall

df['Antall']

# -----------
# ### Active vs non-active employees distribution

resigned_group = df.loc[df["Sluttdato person"].notna()]
active_group = df.loc[df["Sluttdato person"].isna()]

plot_pie(items=[len(active_group), len(resigned_group)],
         labels=['Active employees','Resigned employees'],
         legend=[f'Active: {len(active_group)}', f'Resigned: {len(resigned_group)}'],
         title='Employees distribution')


