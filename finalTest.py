from google.cloud import translate_v2 as translate
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from feedbackManager import RecipeFeedbackManager
from nltk.tag import pos_tag
import random
import wit
import re
import nltk
import requests

# You might need to download the NLTK data if you haven't already
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Feedback manager
feedback_manager = RecipeFeedbackManager()

# Chave da API do Spoonacular
#API_KEY = "f3c3a0bd153e40c18e2e8b1a4848b3d2"
#API_KEY = "a405707f782c4928a1df9956d0c3db81"
API_KEY = "acc9d45ce96b4f1a9cf8295f94837c39"

# Configure o caminho para o arquivo de credenciais JSON
caminho_das_credenciais = 'credenciais.json'

# Configure o seu token de acesso
access_token = "N75BVXPKMNI2PG2DIBIBROQAH6NWGZKL"

# Inicialize o cliente Wit
client = wit.Wit(access_token)

# Defina o dicionário de respostas com um nome diferente da função
dicionario = [
    (r'oi|olá|e aí', ["Olá, {nome}!", "Oi, {nome}!"]),
    (r'tudo bem?', ["Estou ótima! Obrigada por perguntar, {nome}!"]),
    (r'receita|ingrediente', ['Claro, vou te ajudar a encontrar {nome}. Me diga quais ingredientes você possui.']),
    (r'humano|humana|robô|robo', ['Eu sou um modelo de assistente virtual baseado em IA. Estou aqui para te ajudar fornecendo informações de receitas para os seus ingredientes. Como um programa de inteligência artificial, não tenho uma identidade pessoal, mas estou aqui para auxiliá-lo da melhor maneira possível. Como posso ajudar você hoje?']),
    (r'funcionamento|funciona', ['Claro! Eu vou responder para você como eu funciono. Eu sou uma assistente virtual baseado em IA que irá te retornar uma receita com base no ingrediente que você me passar. Que tal me passar um ingrediente de exemplo?']),
    (r'consciência|consciencia', ['Eu fui desenvolvida pelo melhor artesão de inteligência artificial do mundo, o professor doutor em ciência da computação, Paulo dos Anjos de Salto. Respondendo a sua pergunta, pode-se dizer quem sim! Eu possuo uma consciência acima da média!']),
]

def traduzir_text_p_ingles(texto):
    client = translate.Client.from_service_account_json(
        caminho_das_credenciais)
    resultado = client.translate(
        texto, source_language='pt', target_language='en')
    return resultado['translatedText']

def traduzir_text_p_portugues(texto):
    client = translate.Client.from_service_account_json(
        caminho_das_credenciais)
    resultado = client.translate(
        texto, source_language='en', target_language='pt')
    return resultado['translatedText']

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

# Função para responder com base nos padrões
def responder_mensagem(mensagem):
    for padrao, respostas in dicionario:
        match = re.match(padrao, mensagem, re.IGNORECASE)
        if match:
            resposta = random.choice(respostas)
            return resposta

    # Se nenhum padrão for encontrado, retorne uma resposta padrão
    return "Desculpe, não entendi. Pode reformular a pergunta?"

# Função para enviar uma mensagem ao Wit.ai e obter a resposta
def enviar_mensagem(texto):
    response = client.message(texto)
    return response

# Exemplo de uso
print("Olá! Eu sou sua assitente virtual de receitas. Como posso te ajudar?")

def main():
    ingredientes = []
    while True:
        mensagem = input(" ")
        if mensagem == 'sair':
            print("Até mais!")
            break
        try:
            response = enviar_mensagem(mensagem)

            # print("Resposta do Wit.ai:", response)

            # Acesse a lista de entidades na resposta
            entidades = response['entities']

            # Verifique se a intenção principal corresponde à ação desejada
            intencao_principal = response['intents'][0]['name']
            if intencao_principal == 'recipe':
                # Itere sobre as entidades
                for entidade in entidades['ingredient:ingredient']:
                    # Verifique a confiança (confidence)
                    if entidade['confidence'] > 0.7:
                        ingrediente = traduzir_text_p_ingles(entidade['value'])
                        ingredientes.append(ingrediente)
                        print(f'Ingrediente: {ingrediente}')
                print("Buscando receitas...")
                recipes = search_recipes_by_ingredients(", ".join(ingredientes))
                if not recipes:
                    print("Nenhuma receita foi achada por meio desses ingredientes :(")
                else:
                    print("Aqui estão algumas receitas que você pode tentar:")
                for recipe in recipes:
                    preparation_steps = get_preparation_steps(recipe["id"])
                    if preparation_steps:
                        print("- " + traduzir_text_p_portugues(recipe["title"]))
                        print("Passos de preparação:")
                        for step in preparation_steps:
                            print( traduzir_text_p_portugues(step["step"]))       
                    else:
                        print(f"Não há modos de preparo disponíveis para { traduzir_text_p_portugues(recipe['title'])}.")
                    print("\n")
            elif intencao_principal == 'conversation':
                if 'object:object' in response['entities']:
                    object = response['entities']['object:object'][0]['value']
                    user_input = object.lower()  # Converte para minúsculas para coincidir com as chaves do dicionário
                else:
                    salutations = response['entities']['salutation:salutation']
                    user_input = salutations[0]['value'].lower()  # Converte para minúsculas para coincidir com as chaves do dicionário

                # Verifique se 'user_input' está presente nas chaves do dicionário 'random_conversation_responses'
                resposta = responder_mensagem(user_input)

                # Verifique se o nome está presente nas entidades
                if 'person_name:person_name' in response['entities']:
                    nome = response['entities']['person_name:person_name'][0]['value']
                    resposta = resposta.replace("{nome}", nome)  # Substitui "{nome}" pela variável nome
                else:
                    resposta = resposta.replace("{nome}", "fique à vontade para me pedir uma receita")  # Substitui "{nome}" pela variável nome
                print(resposta)

            elif intencao_principal == 'questions':
                if 'doubt:doubt' in response['entities']:
                    doubt = response['entities']['doubt:doubt'][0]['value']
                    user_input =  doubt.lower()  # Converte para minúsculas para coincidir com as chaves do dicionário

                resposta = responder_mensagem(user_input)            
                    # Verifique se o nome está presente nas entidades
                if 'person_name:person_name' in response['entities']:
                    nome = response['entities']['person_name:person_name'][0]['value']
                    resposta = resposta.replace("{nome}", nome)  # Substitui "{nome}" pela variável nome   
                print(resposta)
            else:
                print("Desculpe, não entendi. Pode reformular a pergunta.")
        except wit.WitError as e:
            print("Erro do Wit.ai:", e)

if __name__ == "__main__":
    main()