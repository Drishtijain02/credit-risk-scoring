
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="📊",
    layout="wide",
)

# Locate the project root locally and on Streamlit Cloud.
current_dir = Path(__file__).resolve().parent
possible_roots = [current_dir, *current_dir.parents]

PROJECT_ROOT = next(
    (
        path for path in possible_roots
        if (
            path / "02_Model_Training" / "adya_outputs"
            / "Hist_Gradient_Boosting_pipeline.joblib"
        ).exists()
    ),
    current_dir.parent,
)

MODEL_PATH = (
    PROJECT_ROOT
    / "02_Model_Training"
    / "adya_outputs"
    / "Hist_Gradient_Boosting_pipeline.joblib"
)

# Exact model feature order from the UCI credit-card dataset.
DEFAULTS = {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 2,
    "AGE": 25,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 5000,
    "BILL_AMT2": 5000,
    "BILL_AMT3": 5000,
    "BILL_AMT4": 5000,
    "BILL_AMT5": 5000,
    "BILL_AMT6": 5000,
    "PAY_AMT1": 1000,
    "PAY_AMT2": 1000,
    "PAY_AMT3": 1000,
    "PAY_AMT4": 1000,
    "PAY_AMT5": 1000,
    "PAY_AMT6": 1000,
}

FEATURE_NAMES = list(DEFAULTS.keys())

category_options = {
    "SEX": {
        1: "Male",
        2: "Female",
    },
    "EDUCATION": {
        0: "Unknown / other code (0)",
        1: "Graduate school",
        2: "University",
        3: "High school",
        4: "Others",
        5: "Unknown / other code (5)",
        6: "Unknown / other code (6)",
    },
    "MARRIAGE": {
        0: "Unknown / other code (0)",
        1: "Married",
        2: "Single",
        3: "Others",
    },
}

repayment_status_options = {
    -2: "-2: Special repayment status",
    -1: "-1: Paid in full",
    0: "0: Revolving credit / no recorded delay",
    **{
        value: f"{value}: Payment delayed by {value} month(s)"
        for value in range(1, 9)
    },
}


@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

# Confirm the saved pipeline expects these exact input columns.
expected_features = getattr(model, "feature_names_in_", None)
if expected_features is not None:
    if list(expected_features) != FEATURE_NAMES:
        st.error(
            "The app's feature names or order do not match the saved model. "
            "Please verify them against X_train.csv before deployment."
        )
        st.stop()

st.title("📊 Credit Risk Scoring")
st.caption(
    "Probability-Based Credit Risk Assessment using Machine Learning"
)

st.info(
    "This academic demonstration estimates credit-card default "
    "probability. It is not a real lending decision system."
)

st.subheader("Customer Financial Profile")
st.write(
    "Enter a hypothetical customer profile. The starting values are "
    "illustrative and are not a real customer's data."
)

with st.expander("Understand the financial fields"):
    st.markdown(
        """
        - **LIMIT_BAL:** Credit limit assigned to the customer.
        - **SEX, EDUCATION, MARRIAGE:** Dataset-coded categories.
        - **AGE:** Customer's age in years.
        - **PAY_0, PAY_2–PAY_6:** Repayment-status codes.
        - **BILL_AMT1–BILL_AMT6:** Bill statement amounts.
        - **PAY_AMT1–PAY_AMT6:** Amounts paid in different months.

        The month suffixes follow the original dataset's naming convention.
        """
    )

values = {}

with st.form("customer_form"):
    st.markdown("### Customer details")

    for feature in FEATURE_NAMES:
        default = DEFAULTS[feature]

        if feature in category_options:
            selected = st.selectbox(
                feature.replace("_", " ").title(),
                options=list(category_options[feature].keys()),
                index=list(category_options[feature].keys()).index(default),
                format_func=lambda code, opts=category_options[feature]:
                    opts[code],
            )
            values[feature] = selected

        elif feature.startswith("PAY_") and feature[4:].isdigit():
            selected = st.selectbox(
                feature.replace("_", " ").title(),
                options=list(repayment_status_options.keys()),
                index=list(repayment_status_options.keys()).index(default),
                format_func=lambda code: repayment_status_options[code],
            )
            values[feature] = selected

        elif feature == "AGE":
            values[feature] = st.number_input(
                "Age (years)", min_value=18, max_value=100,
                value=default, step=1,
            )

        elif feature == "LIMIT_BAL":
            values[feature] = st.number_input(
                "Credit Limit (LIMIT BAL)", min_value=0,
                value=default, step=1000,
            )

        else:
            values[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=float(default), step=100.0, format="%.2f",
            )

    submitted = st.form_submit_button(
        "Estimate Default Probability",
        type="primary",
        use_container_width=True,
    )

if submitted:
    customer = pd.DataFrame(
        [[values[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )

    try:
        probabilities = model.predict_proba(customer)
        classes = list(model.classes_)
        default_index = classes.index(1)
        default_probability = float(probabilities[0, default_index])

        st.divider()
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric(
            "Estimated Probability of Default",
            f"{default_probability:.2%}",
        )
        col2.metric(
            "Estimated Probability of No Default",
            f"{1 - default_probability:.2%}",
        )

        st.progress(default_probability)

        if default_probability < 0.10:
            st.success("Lower estimated default probability under this model.")
        elif default_probability < 0.25:
            st.warning("Moderate estimated default probability under this model.")
        else:
            st.error("Higher estimated default probability under this model.")

        st.caption(
            "These display bands are illustrative, not validated lending "
            "thresholds. A predicted probability is an estimate, not a guarantee."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
