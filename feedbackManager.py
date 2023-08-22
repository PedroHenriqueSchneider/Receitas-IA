class RecipeFeedbackManager:
    def __init__(self):
        self.feedback_data = {}  # Dictionary to store feedback data
    
    def register_feedback(self, recipe_id, feedback):
        if recipe_id not in self.feedback_data:
            self.feedback_data[recipe_id] = {"likes": 0, "dislikes": 0}
        
        if feedback.lower() == "like":
            self.feedback_data[recipe_id]["likes"] += 1
        elif feedback.lower() == "dislike":
            self.feedback_data[recipe_id]["dislikes"] += 1
    
    def get_recipe_feedback(self, recipe_id):
        return self.feedback_data.get(recipe_id, {"likes": 0, "dislikes": 0})
    
    def get_all_feedback(self):
        return self.feedback_data
