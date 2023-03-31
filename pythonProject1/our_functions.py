import numpy
import pandas
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle

from itertools import cycle

tfidf_model = pickle.load(open('models/tf_idf_model.pkl', 'rb'))
pca = pickle.load(open('models/pca_model.pkl', 'rb'))

components_ =  800
def transfrom_gender_dummy(df: pandas.DataFrame):
    """ A function to dummify the column gender

    :param df: our dataframe
    :return: pandas.DataFrame
    """
    df.gender.fillna("No info", inplace=True)
    try:
        df["gender"] = df.gender.str.replace("Female", "0")
        df["gender"] = df.gender.str.replace("Male", "1")
        df["gender"] = df.gender.str.replace("No info", "2")
        df["gender"] = df.gender.astype(int)
    except:
        pass

    return df


def transform_creator_dummy(df: pandas.DataFrame):
    """Create a function that dummify the column creator

    :param df:
    :return: None
    """
    conditions = [
        df.creator.str.contains("Marvel"),
        df.creator.str.contains("DC")
    ]
    choices = ["0", "1"]
    df["creator"] = numpy.select(conditions, choices, default="2")
    return df


def clean_height(df: pandas.DataFrame):
    """Clean the columns height and weight

    :param df:
    :return: None
    """
    try:
        df["height"] = df.height.str.extract(r'(\d+)\s*cm', expand=True)
        df["height"] = pandas.to_numeric(df['height'])
        df['weight'] = df.weight.str.extract(r'(\d+)\s*kg', expand=True)
        df["weight"] = pandas.to_numeric(df['weight'])

    except:
        pass

    return df


def lemmatize_text(text):
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


def process_text(df):
    df = clean_height(df)
    df = transfrom_gender_dummy(df)
    df['powers_text_mots'] = df['powers_text'].apply(lambda x: len(x.split()))
    df['powers_text_nb_phrase'] = df['powers_text'].apply(lambda x: len(x.split('. ')))

    df['powers_text_clean'] = df['powers_text'].copy()
    df['powers_text_clean'] = df['powers_text_clean'].map(lambda x: re.sub('[,\.!?()"]', '', x))
    df['powers_text_clean'] = df['powers_text_clean'].map(lambda x: re.sub('\d', '', x))
    df['lemmatized_text'] = df['powers_text_clean'].apply(lambda x: lemmatize_text(x))
    return df


def create_tfidf_merge(df):
    df = process_text(df)
    df_tfidf = process_text(df)

    tfidf_matrix = tfidf_model.transform([" ".join(words) for words in df_tfidf["lemmatized_text"]])
    labels_tfidf = tfidf_model.get_feature_names_out()
    df_tfidf_ = pandas.DataFrame(data=tfidf_matrix.toarray(), columns=labels_tfidf)

    cols = ['overall_score', 'intelligence_score', 'strength_score', 'speed_score', 'durability_score',
            'power_score', 'combat_score', 'gender', 'height', 'weight', 'has_electrokinesis',
            'has_energy_constructs', 'has_mind_control_resistance', 'has_matter_manipulation',
            'has_telepathy_resistance',
            'has_mind_control', 'has_enhanced_hearing', 'has_dimensional_travel', 'has_element_control',
            'has_size_changing',
            'has_fire_resistance', 'has_fire_control', 'has_dexterity', 'has_reality_warping', 'has_illusions',
            'has_energy_beams', 'has_peak_human_condition', 'has_shapeshifting', 'has_heat_resistance', 'has_jump',
            'has_self-sustenance', 'has_energy_absorption', 'has_cold_resistance', 'has_magic', 'has_telekinesis',
            'has_toxin_and_disease_resistance', 'has_telepathy', 'has_regeneration', 'has_immortality',
            'has_teleportation',
            'has_force_fields', 'has_energy_manipulation', 'has_endurance', 'has_longevity', 'has_weapon-based_powers',
            'has_energy_blasts', 'has_enhanced_senses', 'has_invulnerability', 'has_stealth', 'has_marksmanship',
            'has_flight',
            'has_accelerated_healing', 'has_weapons_master', 'has_intelligence', 'has_reflexes', 'has_super_speed',
            'has_durability', 'has_stamina', 'has_agility', 'has_super_strength']

    df_tfidf_pca = pca.transform(df_tfidf_)
    df = df[cols]
    df_tfidf_pca = pandas.DataFrame(df_tfidf_pca)
    df_tfidf_pca.columns = ["component_{}".format(i) for i in range(1, components_ + 1)]
    data_merged = pandas.concat([df, df_tfidf_pca], axis=1)
    return data_merged

#### from Dora
