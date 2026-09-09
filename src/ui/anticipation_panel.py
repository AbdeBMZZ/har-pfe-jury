"""Strict inference interface, shared with the standalone anticipation app."""
def render():
    import io,json
    from pathlib import Path
    import numpy as np
    import streamlit as st
    from scripts.anticipation_pipeline import load_bundle,selected_prob
    from src.models.causal_forecaster import window_features
    st.header('Anticipation d’une fenêtre future')
    st.write('Le modèle utilise uniquement le contexte déjà observé. Les classes prédites sont celles de HAPT ; ce module ne valide pas une alerte de chute.')
    bundles=sorted(Path('checkpoints').glob('anticipation_v2*/metadata.json'))
    if not bundles:
        st.info('Aucun modèle entraîné disponible. Voir LIVRABLE.md.');return
    chosen=st.selectbox('Modèle entraîné',bundles,format_func=lambda p:p.parent.name)
    folder=chosen.parent;meta=json.loads(chosen.read_text());t=meta['temporal']
    st.caption(f"Observation : {100*t['obs_ratio']:.0f} % · Contexte : {t['seq_len']} fenêtres · Modèle choisi sur validation : {meta['selected']}")
    expected=(t['seq_len'],max(1,int(meta['window']*t['obs_ratio'])),6)
    st.write(f'Importer un fichier NumPy de préfixes préparés, de forme {expected}, ou un lot de cette forme. Les mesures doivent suivre le filtrage causal et les unités du protocole.')
    file=st.file_uploader('Contexte observé (.npy)',type=['npy'])
    if file is not None and st.button('Prédire l’activité future'):
        try:
            x=np.load(io.BytesIO(file.getvalue()),allow_pickle=False)
            if x.ndim==3:x=x[None]
            if x.ndim!=4 or tuple(x.shape[1:])!=expected:raise ValueError(f'Forme attendue : (lot, {expected})')
            meta,ck,model,refs=load_bundle(folder,'cpu',inference_only=True);p=selected_prob(window_features(x),ck,model,refs,'cpu',meta['selected'])
            names=['Marche','Montée escaliers','Descente escaliers','Assis','Debout','Allongé','Debout vers assis','Assis vers debout','Assis vers allongé','Allongé vers assis','Debout vers allongé','Allongé vers debout']
            st.dataframe([{'Exemple':i+1,'Activité future':names[int(row.argmax())],'Score non calibré':float(row.max())} for i,row in enumerate(p)])
            st.caption('Le score du modèle ne constitue pas une garantie de justesse. L’horizon porte sur une fenêtre future, pas sur le début d’un événement.')
        except (ValueError,RuntimeError,OSError) as e:st.error(str(e))
