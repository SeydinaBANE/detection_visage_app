import cv2 as cv
import streamlit as st
import numpy as np
import os
import io

# Détection - Chargement des classificateurs Haar
def load_cascades(base_path):
    paths = {
        "face": os.path.join(base_path, "haarcascade_frontalface_default.xml"),
        "eyes": os.path.join(base_path, "haarcascade_eye.xml"),
        "smile": os.path.join(base_path, "haarcascade_smile.xml"),
    }

    cascades = {}
    for key, path in paths.items():
        if not os.path.isfile(path):
            st.error(f"Fichier manquant : {path}")
            st.stop()
        cascade = cv.CascadeClassifier(path)
        if cascade.empty():
            st.error(f"Erreur au chargement du classificateur : {path}")
            st.stop()
        cascades[key] = cascade

    return cascades

# App Streamlit
def main():
    st.set_page_config(page_title="Détection Visage + Yeux + Sourire", layout="centered")
    st.title("😄 Détection de Visage, Yeux et Sourire")

    base_path = os.path.dirname(__file__)
    cascades = load_cascades(base_path)

    selected_color = st.color_picker("🎨 Couleur du rectangle visage", "#00FF00")
    min_neighbors = st.slider("minNeighbors (visage)", 1, 10, 5)
    scale_factor = st.slider("scaleFactor (visage)", 1.1, 2.0, 1.3, 0.1)

    uploaded_img = st.camera_input("📷 Prenez une photo avec votre webcam")

    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        frame = cv.imdecode(file_bytes, 1)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        color = tuple(int(selected_color[i:i + 2], 16) for i in (1, 3, 5))[::-1]
        faces = cascades["face"].detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        if len(faces) == 0:
            st.info("Aucun visage détecté.")
        else:
            st.success(f"{len(faces)} visage(s) détecté(s).")

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = cascades["eyes"].detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            smiles = cascades["smile"].detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
            for (sx, sy, sw, sh) in smiles:
                cv.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)

        st.image(cv.cvtColor(frame, cv.COLOR_BGR2RGB), channels="RGB", caption="Image annotée")

        _, buffer = cv.imencode('.jpg', frame)
        st.download_button(
            label="📥 Télécharger l'image annotée",
            data=io.BytesIO(buffer),
            file_name="image_detectee.jpg",
            mime="image/jpeg"
        )

if __name__ == "__main__":
    main()
