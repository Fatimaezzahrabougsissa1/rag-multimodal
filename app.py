# app.py
import streamlit as st
import os
from rag import answer

# Configuration de la page Streamlit
st.set_page_config(
    page_title="MAINTiDOC - Assistant Multimodal et Agentique",
    page_icon="🛠️",
    layout="wide"
)

# En-titre principal
st.title("🛠️ MAINTiDOC : Assistant Documentaire Multimodal")
st.markdown("""
*Assistant intelligent destiné aux techniciens de maintenance, capable d'exploiter des documents techniques, 
des schémas et des photographies avec des références vérifiables[cite: 12, 19].*
""")

# Barre latérale (Sidebar)
with st.sidebar:
    st.header("🎛️ Paramètres & Configuration")
    st.info("Modèle LLM : **qwen2.5:3b** (via Ollama)")
    st.info("Base vectorielle : **PostgreSQL + PGVector**")
    
    st.divider()
    
    # Paramètre dynamique pour le nombre de résultats (Top K)
    top_k = st.slider("Nombre de passages textuels (Top K)", min_value=2, max_value=10, value=5)
    
    st.divider()
    if st.button("🗑️ Effacer l'historique de discussion", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# Initialisation de l'historique de conversation dans la session Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Si des images étaient associées à la réponse, on les réaffiche
        if "images" in message and message["images"]:
            cols = st.columns(len(message["images"]))
            for idx, (p_num, i_path) in enumerate(message["images"]):
                with cols[idx]:
                    if os.path.exists(i_path):
                        st.image(i_path, caption=f"Page {p_num}", use_container_width=True)

# Zone de saisie utilisateur (Chat input moderne de Streamlit)
if query := st.chat_input("Posez votre question technique sur l'équipement..."):
    # Ajouter le message de l'utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Génération de la réponse par le pipeline RAG
    with st.chat_message("assistant"):
        with st.spinner("Analyse du document et recherche multimodale en cours..."):
            try:
                # Appel de la fonction answer (vous pouvez aussi adapter rag.py pour prendre k en paramètre)
                response_text, pages, images = answer(query)

                # Construction de la réponse formatée
                full_response = f"{response_text}\n\n**📖 Pages sources consultées :** {', '.join([str(p) for p in pages]) if pages else 'Aucune'}"
                
                st.markdown(full_response)

                # Affichage des images pertinentes (CLIP) s'il y en a
                if images:
                    st.markdown("**🖼️ Schémas ou figures pertinents :**")
                    cols = st.columns(min(len(images), 3)) # Max 3 colonnes pour l'affichage
                    for idx, (page_num, img_path) in enumerate(images):
                        col_idx = idx % 3
                        with cols[col_idx]:
                            if os.path.exists(img_path):
                                st.image(img_path, caption=f"Page {page_num} - {os.path.basename(img_path)}", use_container_width=True)
                            else:
                                st.warning(f"Image introuvable : {img_path}")

                # Sauvegarde dans l'historique de session
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "images": images
                })

            except Exception as e:
                error_msg = f"Une erreur est survenue lors de l'exécution du RAG : {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})