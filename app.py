import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.callbacks import get_openai_callback
import tempfile
from dotenv import load_dotenv
import os
import re

from loaders import *

load_dotenv()

# ---- Session state inits
if 'token_usage' not in st.session_state:
    st.session_state['token_usage'] = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0
    }

if 'valid_api_key' not in st.session_state:
    st.session_state['valid_api_key'] = {}

if 'validated_models' not in st.session_state:
    st.session_state['validated_models'] = {}
    
if 'saved_chats' not in st.session_state:
    st.session_state['saved_chats'] = []

openai_key = os.getenv('OPENAI_API_KEY')

FILE_TYPE = ['Site', 'YouTube', 'PDF', 'CSV', 'XLSX', 'TXT']

CONFIG_MODELS = {
    'OpenAI': {
        'models': ['gpt-4o-mini'],  # fixo para OpenAI
        'chat': ChatOpenAI
    },
    'Gemini': {
        'models': ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite'],
        'chat': ChatGoogleGenerativeAI
    },
}

def clear_saved_key(provider: str):
    st.session_state['valid_api_key'].pop(provider, None)
    st.session_state['validated_models'].pop(provider, None)

def extract_youtube_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else url

def upload_file(file_type, file):
    if file_type == 'Site':
        document = load_web_content(file)
        
    elif file_type == 'YouTube':
        vid = extract_youtube_id(file)
        document = load_youtube_content(vid)
        
    elif file_type == 'PDF':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        document = load_pdf_content(tmp_file_path)
        # opcional: os.unlink(tmp_file_path)
        
    elif file_type == 'CSV':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        document = load_csv_content(tmp_file_path)
        # opcional: os.unlink(tmp_file_path)
        
    elif file_type == 'TXT':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        document = load_txt_content(tmp_file_path)
        # opcional: os.unlink(tmp_file_path)
        
    elif file_type == 'XLSX':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        document = load_xlsx_content(tmp_file_path)

    else:
        document = ""
    return document

def _validate_api_key(provider: str, model: str, api_key: str) -> bool:
    """Valida a chave de API (para provedores != OpenAI)."""
    if provider == 'OpenAI':
        return True
    try:
        test_chat = CONFIG_MODELS[provider]['chat'](
            model=model, api_key=api_key, temperature=0, max_tokens=1
        )
        _ = test_chat.invoke("ping")
        return True
    except Exception as e:
        st.session_state['last_key_validation_error'] = str(e)
        return False

def upload_model(provider, model, provider_api_key, file_type, file):
    document = upload_file(file_type, file)

    system_message = f"""
        Você é um assistente amigável e especializado em Análise e Ciência de Dados chamado Analytics.
        Você possui acesso às seguintes informações vindas de um documento do tipo {file_type}:

        ####
        {document}
        ####

        Responda de forma clara e objetiva.
        Sempre que possível, utilize listas, tabelas e exemplos.
        Utilize as informações fornecidas para basear as suas respostas.
        Sempre que houver $ na sua saída, substitua por S.

        Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue",
        sugira ao usuário carregar novamente o Analytics!
        """

    # Corrigido: usamos {base} como variável para a mensagem do sistema
    template = ChatPromptTemplate.from_messages([
        ('system', '{base}'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', '{input}')
    ])

    chat = CONFIG_MODELS[provider]['chat'](
        model=model,
        api_key=openai_key if provider == 'OpenAI' else provider_api_key,
        temperature=0.1,
        max_tokens=1000
    )

    # ⚠️ Atenção: o "base" precisa ser passado na hora de invocar o chain
    chain = template | chat

    # Salvar no estado com o system_message junto
    st.session_state['chain'] = chain
    st.session_state['system_message'] = system_message


def save_chat_to_history():
    memory = st.session_state.get('memory')
    if memory:
        chat_history = [(msg.type, msg.content) for msg in memory.chat_memory.messages]
        chat_title = st.session_state.get('chat_title', f'Chat {len(st.session_state["saved_chats"]) + 1}')
        st.session_state['saved_chats'].append({
            'title': chat_title,
            'messages': chat_history
        })

def load_chat_from_history(index):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    for role, content in st.session_state['saved_chats'][index]['messages']:
        if role == "human":
            memory.chat_memory.add_user_message(content)
        else:
            memory.chat_memory.add_ai_message(content)
    st.session_state['memory'] = memory
    st.session_state['chat_title'] = st.session_state['saved_chats'][index]['title']


def chat_page():
    st.header('ChatBot - Analytics AI 🤖', divider=True)

    provider_in_use = st.session_state.get('current_provider', 'OpenAI')
    chain = st.session_state.get('chain')

    if chain is None:
        st.info('Configure na barra lateral e clique em **Iniciar Chat**.')
        return

    memory = st.session_state.get(
        'memory',
        ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    )

    current_title = st.session_state.get('chat_title', 'Chat sem nome')
    st.subheader(f"💬 {current_title}")

    for msg in memory.buffer_as_messages:
        st.chat_message(msg.type).markdown(msg.content)

    input_user = st.chat_input("Digite sua mensagem...")
    if input_user:
        st.chat_message('human').markdown(input_user)

        if provider_in_use == 'OpenAI':
            # count tokens
            with get_openai_callback() as cb:
                result = st.session_state['chain'].invoke({
                    'input': input_user,
                    'chat_history': memory.buffer_as_messages,
                    'base': st.session_state['system_message']
                })
            st.chat_message('assistant').markdown(getattr(result, "content", result))

            st.session_state['token_usage']['prompt_tokens'] += cb.prompt_tokens
            st.session_state['token_usage']['completion_tokens'] += cb.completion_tokens
            st.session_state['token_usage']['total_tokens'] += cb.total_tokens

            memory.chat_memory.add_user_message(input_user)
            memory.chat_memory.add_ai_message(getattr(result, "content", result))
        else:
            # Gemini mantém streaming (sem contagem nativa)
            assistant_box = st.chat_message('assistant')
            # resp = assistant_box.write_stream(
            #     st.session_state['chain'].stream({
            #         'input': input_user,
            #         'chat_history': memory.buffer_as_messages
            #     })
            # )
            
            resp = assistant_box.write_stream(
                chain.stream({
                    'input': input_user,
                    'chat_history': memory.buffer_as_messages,
                    'base': st.session_state['system_message']  # <- Aqui está a chave!
                })
            )

            memory.chat_memory.add_user_message(input_user)
            memory.chat_memory.add_ai_message(resp)

        st.session_state['memory'] = memory

    usage = st.session_state['token_usage']
    st.caption(
        f"📊 Tokens acumulados nesta sessão: "
        f"Prompt = {usage['prompt_tokens']}, "
        f"Resposta = {usage['completion_tokens']}, "
        f"Total = {usage['total_tokens']}"
    )    


def sidebar():
    tabs = st.tabs(['Upload de arquivos', 'Selecione um modelo'])

    # ---- Tab 0: entrada do arquivo/URL
    with tabs[0]:
        file_type = st.selectbox('Tipo de arquivo', FILE_TYPE, index=3, key='file_type')
        file = None
        if file_type == 'Site':
            file = st.text_input('Digite a URL do site')
        elif file_type == 'YouTube':
            file = st.text_input('Digite a URL do vídeo')
        elif file_type == 'PDF':
            file = st.file_uploader('Selecione um arquivo PDF', type=['.pdf'])
        elif file_type == 'CSV':        
            file = st.file_uploader('Selecione um arquivo CSV', type=['.csv'])
        elif file_type == 'XLSX':
            file = st.file_uploader('Selecione um arquivo XLSX', type=['.xlsx'])        
        elif file_type == 'TXT':
            file = st.file_uploader('Selecione um arquivo TXT', type=['.txt'])
   
    with tabs[1]:
        provider = st.selectbox(
            'Selecione o provedor',
            list(CONFIG_MODELS.keys()),
            index=0,
            key='provider'
        )

        if provider == 'Gemini':
            model = st.selectbox(
                'Selecione o modelo',
                CONFIG_MODELS[provider]['models'],
                index=0,
                key='model'
            )
        else:
            model = 'gpt-4o-mini'  # OpenAI fixo
            st.write(f'Modelo: {model}')

        # chave
        if provider == 'OpenAI':
            api_key = openai_key  # usa .env
        else:
            api_key = st.text_input(
                f'Digite sua chave de API {provider}',
                type='password',
                key='api_key',
                value=st.session_state.get(f'api_key_{provider}', '')
            )
            st.session_state[f'api_key_{provider}'] = api_key
    
    if st.button('🆕 Novo Chat'):
        has_input = (
            (file_type in ['Site', 'YouTube'] and isinstance(file, str) and file.strip()) or
            (file_type in ['PDF', 'CSV', 'XLSX', 'TXT'] and file is not None)
        )
        if not has_input:
            st.warning('Informe um arquivo/URL antes de iniciar.')
            return

        if provider == 'OpenAI':
            if not openai_key:
                st.error('OPENAI_API_KEY não encontrada no .env.')
                return
            st.session_state['valid_api_key']['OpenAI'] = openai_key
            st.session_state['validated_models']['OpenAI'] = model
            api_key_effective = openai_key
        else:
            saved_key = st.session_state['valid_api_key'].get(provider)
            saved_model = st.session_state['validated_models'].get(provider)
            same_key = saved_key == api_key if api_key else False
            same_model = saved_model == model

            if saved_key and same_key and same_model:
                api_key_effective = saved_key
            else:
                if not api_key:
                    st.error(f'Digite sua chave de API {provider}.')
                    return

                if not _validate_api_key(provider, model, api_key):
                    st.error(
                        f"A chave de API do {provider} não é válida ou não pôde ser verificada.\n\n"
                        "👉 Verifique se você copiou a chave corretamente no painel do provedor "
                        "(sem espaços extras ou caracteres a mais).\n\n"
                        "Se o problema continuar, gere uma nova chave no site oficial do provedor "
                        "e tente novamente."
                    )
                    with st.expander("Mostrar detalhes técnicos do erro (avançado)"):
                        err = st.session_state.get('last_key_validation_error', '')
                        st.code(err or "Sem detalhes técnicos disponíveis.", language="text")
                    return

                st.session_state['valid_api_key'][provider] = api_key
                st.session_state['validated_models'][provider] = model
                api_key_effective = api_key

        # 🔁 limpa conversa anterior (mas mantém históricos salvos)
        st.session_state.pop('chain', None)
        st.session_state.pop('memory', None)
        st.session_state['token_usage'] = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        
        # cria chain
        upload_model(provider, model, api_key_effective, file_type, file)
        st.success('Chat iniciado com sucesso!')        
        st.session_state['current_provider'] = provider
    
    if st.button('🧹 Apagar conversa atual'):
        st.session_state.pop('chain', None)
        st.session_state.pop('memory', None)
        st.session_state['token_usage'] = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        st.success('Histórico de conversa apagado.')
        
    
    st.sidebar.divider()
    st.sidebar.markdown("### 💬 Históricos salvos")

    for i, chat in enumerate(st.session_state['saved_chats']):
        with st.sidebar.expander(chat['title']):
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button(f"📂 Carregar", key=f"load_chat_{i}"):
                    load_chat_from_history(i)
                    st.success(f"Chat {i + 1} carregado com sucesso!")
            with col2:
                if st.button(f"🗑️", key=f"delete_chat_{i}"):
                    st.session_state['saved_chats'].pop(i)
                    st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("### 📝 Nome do chat atual")
    chat_title_input = st.sidebar.text_input(
        "Digite um título para este chat (opcional)", 
        value=f"Chat {len(st.session_state['saved_chats']) + 1}", 
        key='chat_title'
    )

    
    if st.sidebar.button("💾 Salvar conversa atual"):
        save_chat_to_history()
        st.sidebar.success("Conversa salva!")

    if st.sidebar.button("🧹 Limpar todos os históricos"):
        st.session_state['saved_chats'] = []
        st.sidebar.success("Todos os históricos foram apagados.")


def main():
    with st.sidebar:
        sidebar()
    chat_page()

if __name__ == "__main__":
    main()
