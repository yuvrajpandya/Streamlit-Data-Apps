import streamlit as st 
import joblib
import os
import numpy as np
import pandas as pd

attrib_info = """
#### Attribute Information:
    - Age 1.20-65
    - Sex 1. Male, 2.Female
    - Polyuria 1.No, 2.Yes.
    - Polydipsia 1.No, 2.Yes.
    - sudden weight loss 1.No, 2.Yes.
    - weakness 1.No, 2.Yes.
    - Polyphagia 1.No, 2.Yes.
    - Genital thrush 1.No, 2.Yes.
    - visual blurring 1.No, 2.Yes.
    - Itching 1.No, 2.Yes.
    - Irritability 1.No, 2.Yes.
    - delayed healing 1.No, 2.Yes.
    - partial paresis 1.No, 2.Yes.
    - muscle stiffness 1.No, 2.Yes.
    - Alopecia 1.No, 2.Yes.
    - Obesity 1.No, 2.Yes.
    - Class 1.Negative, 2.Positive.

"""
label_dict = {"No":1,"Yes":2}
gender_map = {"Male":1,"Female":2}
target_label_map = {"Negative":0,"Positive":1}

# features = ['age', 'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss',
#        'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring',
#        'itching', 'irritability', 'delayed_healing', 'partial_paresis',
#        'muscle_stiffness', 'alopecia', 'obesity']

features = ['age', 'gender', 'polyuria', 'polydipsia', 's', 'w', 'p', 'g', 'v', 'i', 'i', 'd', 'p', 'm', 'a', 'b']

def get_fvalue(val):
	feature_dict = {"No":1,"Yes":2}
	for key,value in feature_dict.items():
		if val == key:
			return value 

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 



# Load ML Models
@st.cache
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


def run_ml_app():
	st.subheader("Machine Learning Section")
	loaded_model = load_model("models/diabetes_logistics_reg_predictor.pkl")
	loaded_preprocessor = load_model("models/diabetes_data_preprocessor.pkl") 

	with st.beta_expander("Attributes Info"):
		st.markdown(attrib_info,unsafe_allow_html=True)

	# Layout
	col1,col2 = st.beta_columns(2)

	with col1:
		age = st.number_input("Age",10,100)
		gender = st.radio("Gender",("Female","Male"))
		polyuria = st.radio("Polyuria",["No","Yes"])
		polydipsia = st.radio("Polydipsia",["No","Yes"]) 
		sudden_weight_loss = st.selectbox("Sudden_weight_loss",["No","Yes"])
		weakness = st.radio("weakness",["No","Yes"]) 
		polyphagia = st.radio("polyphagia",["No","Yes"]) 
		genital_thrush = st.selectbox("Genital_thrush",["No","Yes"]) 
		
	
	with col2:
		visual_blurring = st.selectbox("Visual_blurring",["No","Yes"])
		itching = st.radio("itching",["No","Yes"]) 
		irritability = st.radio("irritability",["No","Yes"]) 
		delayed_healing = st.radio("delayed_healing",["No","Yes"]) 
		partial_paresis = st.selectbox("Partial_paresis",["No","Yes"])
		muscle_stiffness = st.radio("muscle_stiffness",["No","Yes"]) 
		alopecia = st.radio("alopecia",["No","Yes"]) 
		obesity = st.select_slider("obesity",["No","Yes"]) 

	with st.beta_expander("Your Selected Options"):
		result = {
		'Age':[age],
		'Gender':[gender],
		'Polyuria':[polyuria],
		'Polydipsia':[polydipsia],
		'Sudden_weight_loss':[sudden_weight_loss],
		'Weakness':[weakness],
		'Polyphagia':[polyphagia],
		'Genital_thrush':[genital_thrush],
		'Visual_blurring':[visual_blurring],
		'Itching':[itching],
		'Irritability':[irritability],
		'Delayed_healing':[delayed_healing],
		'Partial_paresis':[partial_paresis],
		'Muscle_stiffness':[muscle_stiffness],
		'Alopecia':[alopecia],
		'Obesity':[obesity]}
		st.write(result)


	with st.beta_expander("Prediction Results"):
		# receive & transform the input
		st.write('Transformed input')
		df_result = pd.DataFrame(result)
		single_sample = loaded_preprocessor.transform(df_result) #load the data preprocessor to transform the user input
		st.write(single_sample)

		# predict the outcome
		prediction = loaded_model.predict(single_sample)
		pred_prob = loaded_model.predict_proba(single_sample)
		# st.write(prediction)

		# display the results with probabilities of each outcome
		prediction_description = "Diabetes Risk - " + format(prediction[0]) + " | See Prediction Probability Score for the likelihood"
		if prediction == "Positive":
			# st.warning("Patient's prediction{}".format(prediction[0]))
			st.warning(prediction_description)
		else:
			st.success(prediction_description)
			
		pred_probability_score = {"Negative DM":pred_prob[0][0]*100,"Positive DM":pred_prob[0][1]*100}
		st.subheader("Prediction Probability Score")
		st.json(pred_probability_score)

