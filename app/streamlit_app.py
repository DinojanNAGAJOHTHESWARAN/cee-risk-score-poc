from pathlib import Path
import json, joblib
import pandas as pd
import streamlit as st
from src.config import ROOT, load_config
from src.data.loader import PRE_CONTROL_COLUMNS, load_dataset, normalize_nulls, validate_schema
from src.data.validation import ingestion_report, sanitize_for_scoring, data_quality_report
from src.models.hybrid_score import score_dataframe, explain_one

st.set_page_config(page_title="CEE Risk Score",page_icon="🛡️",layout="wide")

@st.cache_resource
def get_bundle():
    p=ROOT/"models"/"risk_bundle.joblib"; return joblib.load(p) if p.exists() else None
@st.cache_data(show_spinner=False)
def load_default(path): return load_dataset(path)
def read_upload(f): return normalize_nulls(pd.read_csv(f,low_memory=False) if f.name.lower().endswith(".csv") else pd.read_excel(f))
def queue(scored,n=100):
    cols=[c for c in ["_row_id","DateCreationSoumissionDossier","ReferenceFicheCEE","PartenairePseudonymise","Region","ScoreRisque","NiveauRisque","RaisonsMetier"] if c in scored]
    d=scored[cols].head(n).copy(); d["Dossier"]=d["_row_id"].astype(int).add(1).map(lambda x:f"CEE-{x:05d}"); return d.drop(columns="_row_id")[["Dossier"]+[c for c in d.columns if c not in ["Dossier","_row_id"]]]

def dashboard(scored):
    st.title("🛡️ Tableau de bord du risque CEE"); st.caption("Prioriser les contrôles à partir d'un score explicable — aucune décision automatique de rejet.")
    if st.session_state.get("imported_scored") is not None:
        source=st.segmented_control("Vue",["Portefeuille historique","Dernier lot importé"],default="Portefeuille historique")
        if source=="Dernier lot importé": scored=st.session_state["imported_scored"]; st.info("Le lot importé est affiché séparément et n'est pas ajouté à l'entraînement.")
    c1,c2,c3,c4=st.columns(4); c1.metric("Dossiers scorés",len(scored)); c2.metric("Priorité élevée",int((scored.NiveauRisque=="Élevé").sum())); c3.metric("Priorité modérée",int((scored.NiveauRisque=="Modéré").sum())); c4.metric("Score médian",f"{scored.ScoreRisque.median():.1f}/100")
    a,b=st.columns(2)
    with a: st.subheader("Répartition des priorités"); st.bar_chart(scored.NiveauRisque.value_counts().reindex(["Faible","Modéré","Élevé"],fill_value=0))
    with b: st.subheader("Lecture opérationnelle"); st.info("Un score élevé signifie que plusieurs signaux convergent vers un besoin de vérification renforcée. Il ne signifie pas que la fraude est établie."); st.write("**Score hybride : modèle supervisé + règles métier + détection d'anomalies.**")
    st.subheader("File de contrôle prioritaire"); f1,f2,f3=st.columns(3); levels=f1.multiselect("Niveau",["Élevé","Modéré","Faible"],default=["Élevé","Modéré"]); partner=f2.text_input("Filtrer partenaire"); fiche=f3.text_input("Filtrer fiche CEE"); v=scored[scored.NiveauRisque.isin(levels)]
    if partner:v=v[v.PartenairePseudonymise.astype(str).str.contains(partner,case=False,na=False)]
    if fiche:v=v[v.ReferenceFicheCEE.astype(str).str.contains(fiche,case=False,na=False)]
    st.dataframe(queue(v),width="stretch",hide_index=True,height=430)

def ingestion(bundle):
    st.title("📥 Importer & scorer"); st.caption("Appliquer le modèle déjà entraîné à un nouveau lot, sans réentraînement."); f=st.file_uploader("Déposer un fichier Excel ou CSV",type=["xlsx","csv"])
    if f is None: st.info("Pour tester le dépôt, utilise data/examples/lot_test_synthetique.csv"); return
    try:d=read_upload(f)
    except Exception as e: st.error(f"Lecture impossible : {e}"); return
    r=ingestion_report(d); st.subheader("1. Contrôle qualité du fichier"); c=st.columns(4); c[0].metric("Dossiers",r["rows"]); c[1].metric("Colonnes",r["columns"]); c[2].metric("Données clés incomplètes",r["rows_with_missing_key"]); c[3].metric("Volumes invalides",r["invalid_volume_rows"])
    if r["missing_required"]: st.error("Colonnes indispensables manquantes : "+", ".join(r["missing_required"])); return
    st.success("Schéma compatible avec le moteur de scoring.")
    if r["post_control_columns"]: st.warning("Variables post-contrôle détectées et exclues : "+", ".join(r["post_control_columns"]))
    else: st.success("Aucune variable post-contrôle détectée.")
    with st.expander("Aperçu des données importées"): st.dataframe(d.head(20),width="stretch",hide_index=True)
    with st.expander("Qualité des colonnes"): st.dataframe(data_quality_report(d),width="stretch",hide_index=True)
    st.subheader("2. Scoring du lot")
    if st.button("Analyser les dossiers",type="primary",width="stretch"):
        clean=sanitize_for_scoring(d); scored=score_dataframe(bundle,clean); st.session_state["imported_raw"]=clean.reset_index(drop=True); st.session_state["imported_scored"]=scored; st.session_state["imported_name"]=f.name
    scored=st.session_state.get("imported_scored")
    if scored is not None:
        c=st.columns(4); c[0].metric("Priorité élevée",int((scored.NiveauRisque=="Élevé").sum())); c[1].metric("Priorité modérée",int((scored.NiveauRisque=="Modéré").sum())); c[2].metric("Priorité faible",int((scored.NiveauRisque=="Faible").sum())); c[3].metric("Score médian",f"{scored.ScoreRisque.median():.1f}/100"); st.subheader("3. File priorisée du lot importé"); st.dataframe(queue(scored,len(scored)),width="stretch",hide_index=True)

def dossier(bundle,df,scored):
    st.title("🔎 Analyser un dossier"); sources=["Portefeuille historique"]+(["Lot importé"] if st.session_state.get("imported_scored") is not None else []); source=st.radio("Source",sources,horizontal=True)
    if source=="Lot importé": scored=st.session_state["imported_scored"]; df=st.session_state["imported_raw"]
    top=scored.head(100); labels={f"CEE-{int(r._row_id)+1:05d} · {r.ReferenceFicheCEE} · {r.PartenairePseudonymise} · {r.Region}":int(r._row_id) for _,r in top.iterrows()}; label=st.selectbox("Choisir parmi les dossiers les plus prioritaires",list(labels)); row=df.iloc[[labels[label]]]; e=explain_one(bundle,row)
    l,r=st.columns([.8,1.4]); l.metric("Score de priorisation",f"{e['score']:.1f}/100"); l.write(f"**Priorité {e['level'].lower()}**"); r.subheader("Composition du score"); c=r.columns(3); c[0].metric("Supervisé",f"{e['supervised']:.1f}/100"); c[1].metric("Règles métier",f"{e['business']:.1f}/100"); c[2].metric("Anomalie",f"{e['anomaly']:.1f}/100")
    st.subheader("Pourquoi ce dossier remonte ?"); tabs=st.tabs(["🔴 Facteurs de risque","🟠 Anomalies","🔵 Incertitudes"])
    for tab,key in zip(tabs,["risk_signals","anomaly_signals","uncertainty_signals"]):
        with tab:
            sig=e[key]
            if not sig: st.caption("Aucun signal dans cette catégorie.")
            for s in sig: st.write(f"**{s['title']}** — {s['detail']}")
    with st.expander("🧠 Contributions du modèle supervisé"):
        st.caption("Associations historiques : elles ne prouvent ni causalité ni fraude.")
        for f in e["model_factors"]: st.write(f"**{f['label']}** — {f['direction']} le score ({f['contribution']:+.3f}).")
    with st.expander("📄 Données disponibles avant contrôle"):
        cols=[c for c in PRE_CONTROL_COLUMNS if c in row.columns]; display=row.iloc[0][cols].rename_axis("Variable").reset_index(name="Valeur"); display["Valeur"]=display.Valeur.apply(lambda x:"—" if pd.isna(x) else str(x)); st.dataframe(display,width="stretch",hide_index=True)
    st.info("Le score est une aide à la priorisation. La décision finale appartient au contrôleur.")

def performance(bundle):
    st.title("📊 Performance & gouvernance"); st.caption("Métriques du modèle chargé : elles sont fixes lors du scoring et ne changent qu'après réentraînement/évaluation."); m=bundle.get("metrics",{}); h=m.get("hybrid",{}); c=st.columns(4); c[0].metric("ROC-AUC",f"{h.get('roc_auc',0):.3f}"); c[1].metric("Recall au seuil",f"{h.get('recall',0):.1%}"); c[2].metric("Précision au seuil",f"{h.get('precision',0):.1%}"); c[3].metric("PR-AUC",f"{h.get('average_precision',0):.3f}")
    st.subheader("Performance liée à la capacité de contrôle"); rows=[{"Part du portefeuille contrôlée":k,"Dossiers défavorables capturés":f"{v.get('recall',0):.1%}","Précision dans la file":f"{v.get('precision',0):.1%}"} for k,v in m.get("top_k",{}).items()]; st.dataframe(pd.DataFrame(rows),width="stretch",hide_index=True)
    st.subheader("Comment lire les métriques"); st.markdown("- **ROC-AUC** : qualité générale du classement.\n- **Recall** : part des dossiers défavorables détectés.\n- **Précision** : part réellement défavorable parmi les dossiers signalés.\n- **Recall@Top-k** : part capturée lorsque la capacité de contrôle est limitée.")
    st.subheader("Gouvernance du POC"); a,b=st.columns(2); a.success("✅ Variables post-contrôle exclues des prédicteurs."); a.success("✅ Aucun rejet automatique : décision humaine conservée."); b.warning("⚠️ La cible est un résultat défavorable, pas une fraude juridiquement confirmée."); b.warning("⚠️ Les historiques contrôlés peuvent comporter un biais de sélection.")

bundle=get_bundle(); cfg=load_config(); st.sidebar.title("🛡️ CEE Risk Score"); page=st.sidebar.radio("Navigation",["Tableau de bord","Importer & scorer","Analyser un dossier","Performance & gouvernance"])
if bundle is None: st.error("Modèle local absent. Entraînez-le avec `python -m src.pipeline.train`."); st.stop()
try: df=load_default(str(ROOT/cfg["data"]["default_path"])); validate_schema(df); scored=score_dataframe(bundle,df)
except Exception as e: st.error(f"Portefeuille historique indisponible : {e}"); st.stop()
if page=="Tableau de bord": dashboard(scored)
elif page=="Importer & scorer": ingestion(bundle)
elif page=="Analyser un dossier": dossier(bundle,df,scored)
else: performance(bundle)
