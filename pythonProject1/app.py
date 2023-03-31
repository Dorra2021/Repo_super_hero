import pandas
from flask import Flask, request, jsonify, render_template
import pickle
from our_functions import *
app = Flask(__name__)
# Load the model
model = pickle.load(open('models/xgb_model.pkl','rb'))
saved_dtypes = pickle.load(open('models/dtypes.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


#def get_data_from_post():
#    """
#    get raw data from user
#    :return: pandas.DataFrame
#    """
#    data = request.get_json(force=True)
#    #return data #
#
@app.route('/predict', methods=['POST'])
def predict():
    data = [x for x in request.form.values()]
    app.logger.debug(data)
    cols = ["powers_text",'overall_score', 'intelligence_score', 'strength_score',
            'speed_score', 'durability_score', 'power_score',
            'combat_score', 'gender', 'height', 'weight','has_electrokinesis', 'has_energy_constructs',
            'has_mind_control_resistance', 'has_matter_manipulation', 'has_telepathy_resistance',
            'has_mind_control', 'has_enhanced_hearing', 'has_dimensional_travel', 'has_element_control',
            'has_size_changing', 'has_fire_resistance', 'has_fire_control', 'has_dexterity', 'has_reality_warping',
            'has_illusions', 'has_energy_beams', 'has_peak_human_condition', 'has_shapeshifting',
            'has_heat_resistance', 'has_jump', 'has_self-sustenance', 'has_energy_absorption',
            'has_cold_resistance', 'has_magic', 'has_telekinesis', 'has_toxin_and_disease_resistance',
            'has_telepathy', 'has_regeneration', 'has_immortality', 'has_teleportation', 'has_force_fields',
            'has_energy_manipulation', 'has_endurance', 'has_longevity', 'has_weapon-based_powers',
            'has_energy_blasts', 'has_enhanced_senses', 'has_invulnerability', 'has_stealth', 'has_marksmanship',
            'has_flight', 'has_accelerated_healing', 'has_weapons_master', 'has_intelligence', 'has_reflexes',
            'has_super_speed', 'has_durability', 'has_stamina', 'has_agility', 'has_super_strength']

    data_dict = dict()
    for col, elem in zip(cols, data):
        data_dict[col] = elem

    data = pandas.DataFrame(data_dict, index=[0])
    cols = ['overall_score', 'powers_text', 'intelligence_score', 'strength_score',
            'speed_score', 'durability_score', 'power_score',
            'combat_score', 'gender', 'height', 'weight','has_electrokinesis', 'has_energy_constructs',
            'has_mind_control_resistance', 'has_matter_manipulation', 'has_telepathy_resistance',
            'has_mind_control', 'has_enhanced_hearing', 'has_dimensional_travel', 'has_element_control',
            'has_size_changing', 'has_fire_resistance', 'has_fire_control', 'has_dexterity', 'has_reality_warping',
            'has_illusions', 'has_energy_beams', 'has_peak_human_condition', 'has_shapeshifting',
            'has_heat_resistance', 'has_jump', 'has_self-sustenance', 'has_energy_absorption',
            'has_cold_resistance', 'has_magic', 'has_telekinesis', 'has_toxin_and_disease_resistance',
            'has_telepathy', 'has_regeneration', 'has_immortality', 'has_teleportation', 'has_force_fields',
            'has_energy_manipulation', 'has_endurance', 'has_longevity', 'has_weapon-based_powers',
            'has_energy_blasts', 'has_enhanced_senses', 'has_invulnerability', 'has_stealth', 'has_marksmanship',
            'has_flight', 'has_accelerated_healing', 'has_weapons_master', 'has_intelligence', 'has_reflexes',
            'has_super_speed', 'has_durability', 'has_stamina', 'has_agility', 'has_super_strength']
    data = data[cols]
    data = create_tfidf_merge(data)

    data = data.astype(saved_dtypes)
    prediction = model.predict(data)
    # Take the first value of prediction
    print(data)
    output = prediction[0]
    if output == 0:
        creator = 'Marvel'
    elif output == 1:
        creator = 'DC'
    else:
        creator = 'Other Creator'
    return render_template('index.html', prediction_text='La maison d\'édition du super-héros est {}'.format(creator))

if __name__ == '__main__':
    app.run(port=5000, debug=True)



#make a function pour prendre en considération les étapes du préprocessing 