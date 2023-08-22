import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from feedbackManager import RecipeFeedbackManager

# pensar no aprendizado
# perguntar o feedback da pessoa
# ordenar as próximas listas sobre o feedback
# colocar um botão em cada receita de like e deslike 
# fazer interface básica 
# talvez mandar a imagem da comida que ele retorna 
# usar recurso da biblioteca de quais ingredientes estão faltando e o tipo do ingrediente
# api retorna flags de vegano, vegetariano, usar.

# You might need to download the NLTK data if you haven't already
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Feedback manager
feedback_manager = RecipeFeedbackManager()

# Chave da API do Spoonacular
#API_KEY = "f3c3a0bd153e40c18e2e8b1a4848b3d2"
#API_KEY = "a405707f782c4928a1df9956d0c3db81"
API_KEY = "acc9d45ce96b4f1a9cf8295f94837c39"
# Simulação de dados de treinamento (substitua isso pelos seus próprios dados reais)
training_data = [
    {"ingredients": "tomato onion garlic", "preference": "vegetarian"},
    {"ingredients": "chicken broccoli rice", "preference": "non-vegetarian"},
]


def parse_ingredients(message):
    tokens = word_tokenize(message)
    tagged = pos_tag(tokens)


    ingredients = []


    for word, tag in tagged:
        if tag == 'NN' or tag == 'NNS':  # NN: singular noun, NNS: plural noun
            ingredients.append(word)


    return ' '.join(ingredients)


def search_recipes_by_ingredients(ingredients):
    base_url = "https://api.spoonacular.com/recipes/findByIngredients"


    params = {
        "apiKey": API_KEY,
        "ingredients": ingredients,
        "number": 5  # Number of recipes to retrieve
    }


    response = requests.get(base_url, params=params)
    recipes = response.json()


    return recipes


def train_preference_model(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([item["ingredients"] for item in data])
    y = [item["preference"] for item in data]


    model = RandomForestClassifier()
    model.fit(X, y)


    return model, vectorizer


def predict_user_preferences(model, vectorizer, ingredients):
    ingredients_vec = vectorizer.transform([ingredients])
    return model.predict(ingredients_vec)[0]

def get_preparation_steps(recipe_id):
    base_url = f"https://api.spoonacular.com/recipes/{recipe_id}/analyzedInstructions"
    
    params = {
        "apiKey": API_KEY
    }
    
    response = requests.get(base_url, params=params)
    instructions = response.json()
    
    if instructions and 'steps' in instructions[0]:
        return instructions[0]['steps']
    else:
        return None

def main():
    print("Hello! I'm your recipe chatbot.")
    
    preference_model, vectorizer = train_preference_model(training_data)

    
    while True:
        message = input("Please enter a list of ingredients: ")
        
        if message.lower() == "exit":
            print("Goodbye!")
            break
        
        ingredients = parse_ingredients(message)
        if not ingredients:
            print("No valid ingredients detected.")
            continue
        
        user_preferences = predict_user_preferences(preference_model, vectorizer, ingredients)
        print(f"Predicted user preference: {user_preferences}")
        
        recipes = search_recipes_by_ingredients(ingredients)
        if not recipes:
            print("No recipes found for the provided ingredients.")
        else:
            print("Here are some recipes you can try:")
            for recipe in recipes:
                preparation_steps = get_preparation_steps(recipe["id"])
                if preparation_steps:
                    print("- " + recipe["title"])
                    print("Preparation Steps:")
                    for step in preparation_steps:
                        print(step["step"])   
                
                print("- " + recipe["title"])     
                feedback = input("Did you like this recipe? (yes/no): ")
                if feedback.lower() == "yes":
                    feedback_manager.register_feedback(recipe["id"], "like")
                elif feedback.lower() == "no":
                    feedback_manager.register_feedback(recipe["id"], "dislike")
                        
                else:
                    print(f"No available preparation steps for {recipe['title']}.")
                print("\n")
            feedbacks = feedback_manager.get_all_feedback()
            print(feedbacks)
if __name__ == "__main__":
    main()