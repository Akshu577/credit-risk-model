"""Streamlit app: move the approval threshold and see what each kind of mistake costs the bank."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.train import (COST_FALSE_APPROVE, COST_FALSE_REJECT, REPORTS, SEED, best_threshold,
                       build_models, business_cost, load_data)

st.set_page_config(page_title="Credit Risk Threshold Explorer", page_icon="📉")
st.title("Credit Risk: Where Should the Bank Draw the Line?")
st.caption("German Credit dataset (1,000 real applicants, UCI). Approving a borrower who defaults costs 5x more "
           "than turning away a good one. Source: github.com/Akshu577/credit-risk-model")


@st.cache_resource(show_spinner="Downloading the data and training both models (about a minute)...")
def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    out = {}
    for name, model in build_models(X).items():
        model.fit(X_train, y_train)
        # Same as src/train.py: the tuned threshold comes from out-of-fold training predictions
        oof = np.zeros(len(y_train))
        for tr, va in cv.split(X_train, y_train):
            m = build_models(X)[name].fit(X_train.iloc[tr], y_train.iloc[tr])
            oof[va] = m.predict_proba(X_train.iloc[va])[:, 1]
        out[name] = (model.predict_proba(X_test)[:, 1], best_threshold(y_train.values, oof))
    return out, y_test.values


results, y_test = train()
name = st.radio("Model", list(results), horizontal=True)
proba, tuned = results[name]

threshold = st.slider("Reject applicants whose predicted risk is above", 0.05, 0.95, 0.50, 0.01)
st.caption(f"The project's tuned threshold for this model is {tuned:.2f}; the usual default is 0.50.")

pred = proba >= threshold
bad_approved = int(((~pred) & (y_test == 1)).sum())
good_rejected = int((pred & (y_test == 0)).sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Approved", f"{(~pred).mean():.0%}")
c2.metric("Bad loans approved", bad_approved)
c3.metric("Good customers rejected", good_rejected)
c4.metric("Cost", business_cost(y_test, proba, threshold),
          delta=int(business_cost(y_test, proba, threshold) - business_cost(y_test, proba, tuned)),
          delta_color="inverse", help="Change compared with the tuned threshold")
st.write(f"Cost = {COST_FALSE_APPROVE} × bad loans approved + {COST_FALSE_REJECT} × good customers rejected, "
         f"on the 250 test applicants.")

grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
curve = pd.DataFrame({m: [business_cost(y_test, p, t) for t in grid] for m, (p, _) in results.items()}, index=grid)
curve.index.name = "threshold"
st.subheader("Cost at every threshold")
st.line_chart(curve, x_label="threshold", y_label="cost")

st.subheader("What drives the risk score most")
imp = pd.read_csv(REPORTS / "feature_importance.csv").head(8)
st.bar_chart(imp, x="feature", y="importance", horizontal=True, sort="-importance")
