import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import requests

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

# Spoonacular API credentials
API_KEY = "f3c3a0bd153e40c18e2e8b1a4848b3d2"

def parse_ingredients(message):
    tokens = word_tokenize(message)
    tagged = pos_tag(tokens)
    
    ingredients = []
    
    for word, tag in tagged:
        if tag == 'NN' or tag == 'NNS':  # NN: singular noun, NNS: plural noun
            ingredients.append(word)
    
    return ingredients

def search_recipes_by_ingredients(ingredients):
    base_url = "https://api.spoonacular.com/recipes/findByIngredients"
    
    params = {
        "apiKey": API_KEY,
        "ingredients": ",".join(ingredients),
        "number": 5  # Number of recipes to retrieve
    }
    
    response = requests.get(base_url, params=params)
    recipes = response.json()
    
    return recipes

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
    
    while True:
        message = input("Please enter a list of ingredients: ")
        
        if message.lower() == "exit":
            print("Goodbye!")
            break
        
        ingredients = parse_ingredients(message)
        if not ingredients:
            print("No valid ingredients detected.")
            continue
        
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
                else:
                    print(f"No available preparation steps for {recipe['title']}.")
        
if __name__ == "__main__":
    main()