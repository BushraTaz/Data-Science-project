#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib
from rasa import Action

class ActionPredictMatchOutcome(Action):
    def name(self) -> str:
        return "action_predict_match_outcome"

    def __init__(self):
        # Load the model when the action is initialized
        self.model = joblib.load("models/ml_models/match_outcome_model.joblib")

    def run(self, dispatcher, tracker, domain):
        # Retrieve any necessary data from the conversation context
        home_team = tracker.get_slot('home_team')
        away_team = tracker.get_slot('away_team')
        
        # You may need to prepare the input based on how your model expects it
        # Assuming the model expects a feature array or DataFrame
        input_data = [[home_team, away_team]]  # Adjust this based on your model's input structure
        
        # Make a prediction using the model
        prediction = self.model.predict(input_data)

        # Respond back with the predicted outcome
        dispatcher.utter_message(text=f"The predicted outcome is: {prediction[0]}")  # Assuming single output
        return []

