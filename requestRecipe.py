import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Substitua "YOUR_SPOONACULAR_API_KEY" pela sua chave de API da Spoonacular
API_KEY = "f3c3a0bd153e40c18e2e8b1a4848b3d2"

# Defina as stopwords e o stemmer da NLTK
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def process_ingredients(ingredients):
    processed_ingredients = []
    for ingredient in ingredients:
        tokens = word_tokenize(ingredient.lower())
        filtered_tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
        processed_ingredients.append(" ".join(filtered_tokens))
    return processed_ingredients

def search_recipes_by_ingredients(ingredients, number=10):
    processed_ingredients = process_ingredients(ingredients)
    url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        "ingredients": ",".join(processed_ingredients),
        "number": number,
        "apiKey": API_KEY
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Erro na requisição:", response.status_code)
        return []

def display_recipe_details(recipe):
    recipe_name = recipe['title']
    recipe_id = recipe['id']
    recipe_url = f"https://spoonacular.com/recipes/{recipe_id}"
    
    print(f"Nome da Receita: {recipe_name}")
    print(f"URL da Receita: {recipe_url}")
    
    if 'instructions' in recipe:
        print("Modo de Preparo:")
        print(recipe['instructions'])
    else:
        print("Modo de Preparo não encontrado.")
    
    print("=" * 50)

# Entrada dos ingredientes
user_input = input("Digite os ingredientes separados por vírgula: ")
ingredients_list = user_input.split(",")

# Buscar receitas com base nos ingredientes fornecidos
recipes = search_recipes_by_ingredients(ingredients_list)

if recipes:
    for recipe in recipes:
        display_recipe_details(recipe)
else:
    print("Nenhuma receita encontrada com os ingredientes fornecidos.")
