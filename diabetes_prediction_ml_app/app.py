import streamlit as st 
import streamlit.components.v1 as stc 
from eda_app import run_eda_app
from ml_app import run_ml_app

html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Early Stage Diabetes Risk Prediction App </h1>
		</div>
		"""

def main():
	# st.title("ML Web App with Streamlit")
	stc.html(html_temp)

	menu = ["Home","EDA","ML","About"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")
		st.write("""
			### Early Stage Diabetes Risk Predictor App
			The app's main objective is to showcase fast prototyping of a data app to present the
			explorative analysis & how model prediction works in a real time. 

			#### Datasource
				- This dataset contains the sign and symptoms data of newly diabetic or would be diabetic patient.
				- https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.
			#### App Content
				- EDA Section: Exploratory Data Analysis of Data
				- ML Section: ML Predictor App

			#### Machine Learning model
				- Since it is a two-class classification problem, I have used a simple LogisticRegression model
				- Multiple models such as RandomForestClassifier & SVM could be used which might give better scores
			""")

		st.subheader("Findings")
		st.write("""
				During the EDA and testing phase of the model, it is found that top 3 factors for determining
				the presence of diabetes in patient are Polyuria, Polydipsia & Gender. Using SHAP values, feature
				importance can be carried out for explanation of the model.

				More on what is Polyuria (frequent large amounts of urine due to excessive levels of sugar)
				& Polydipsia (excessive thirst):
				- https://www.jdrf.org/t1d-resources/about/symptoms/frequent-urination/
				- https://www.jdrf.org/t1d-resources/about/symptoms/extreme-thirst/
		 """)


	elif choice == "EDA":
		run_eda_app()
	elif choice == "ML":
		run_ml_app()
	else:
		st.subheader("About")
		st.subheader("Developed by Yuvraj Pandya")
		html = f"<a href='https://www.linkedin.com/in/yuvraj-pandya'><img src='https://github.com/yuvrajpandya/ScikitLearn/raw/master/linkedin-icon.png'/></a>"
		st.markdown(html, unsafe_allow_html=True)

if __name__ == '__main__':
	main()