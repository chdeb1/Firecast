import streamlit as st
import pandas as pd
import pydeck as pdk

# -----------------------------
# Config
# -----------------------------
PAGE_SIZE = 1000

HIGH_RISK_MIN = 0.75   # > 0.75 = High Risk
MID_RISK_MIN = 0.60    # > 0.60 to <= 0.75 = Mid Risk
LOW_RISK_MIN = 0.50    # > 0.50 to <= 0.60 = Low Risk
# <= 0.50 = No Risk

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Lightning Strike Prediction Viewer", layout="wide")

# -----------------------------
# Session state init
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "upload"

if "data" not in st.session_state:
    st.session_state.data = None

if "dropped_count" not in st.session_state:
    st.session_state.dropped_count = 0

if "original_count" not in st.session_state:
    st.session_state.original_count = 0

if "page_index" not in st.session_state:
    st.session_state.page_index = 0


# -----------------------------
# Helpers
# -----------------------------
def load_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Only CSV and XLSX files are supported.")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map the input columns to the names used by the app."""
    df = df.copy()

    if "strike_lat" in df.columns:
        df["latitude"] = df["strike_lat"]

    if "strike_lon" in df.columns:
        df["longitude"] = df["strike_lon"]

    if "strike_date" in df.columns:
        df["strike_time"] = df["strike_date"]

    if "ignition_probability" in df.columns:
        df["probability"] = df["ignition_probability"]

    if "dataset_type" in df.columns:
        df["dataset_type_clean"] = df["dataset_type"].astype(str).str.strip()

        df["ground_truth"] = df["dataset_type_clean"].str.lower().map(
            {
                "ignition": 1,
                "non-ignition": 0,
                "non_ignition": 0,
                "non ignition": 0,
                "nonignition": 0,
            }
        )

        def ground_truth_label(val):
            if pd.isna(val):
                return "Unknown"
            return "Ignition" if int(val) == 1 else "Non-Ignition"

        df["ground_truth_label"] = df["ground_truth"].apply(ground_truth_label)

    return df


def validate_data(df: pd.DataFrame):
    required = ["latitude", "longitude", "probability"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        return False, f"Missing required column(s): {', '.join(missing)}"

    df = df.copy()
    original_count = len(df)

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")

    # Only drop rows that truly cannot be mapped/classified
    df = df.dropna(subset=["latitude", "longitude", "probability"]).copy()

    dropped_count = original_count - len(df)

    if df.empty:
        return False, "No valid rows with strike_lat, strike_lon, and ignition_probability were found."

    df["probability"] = df["probability"].clip(0, 1)

    if "strike_time" not in df.columns:
        df["strike_time"] = ""

    if "ground_truth" not in df.columns:
        df["ground_truth"] = pd.NA
        df["ground_truth_label"] = "Unknown"
    else:
        df["ground_truth"] = pd.to_numeric(df["ground_truth"], errors="coerce")
        if "ground_truth_label" not in df.columns:
            df["ground_truth_label"] = df["ground_truth"].apply(
                lambda x: "Ignition" if x == 1 else ("Non-Ignition" if x == 0 else "Unknown")
            )

    return True, (df, dropped_count, original_count)


def get_risk_band_from_probability(prob: float) -> str:
    """
    High Risk > 0.75
    Mid Risk  0.60 < p <= 0.75
    Low Risk  0.50 < p <= 0.60
    No Risk   p <= 0.50
    """
    if prob > HIGH_RISK_MIN:
        return "High Risk"
    elif prob > MID_RISK_MIN:
        return "Mid Risk"
    elif prob > LOW_RISK_MIN:
        return "Low Risk"
    return "No Risk"


def get_color_for_risk(risk_band: str):
    color_map = {
        "No Risk": [34, 197, 94, 180],    # green
        "Low Risk": [250, 204, 21, 180],  # yellow
        "Mid Risk": [249, 115, 22, 180],  # orange
        "High Risk": [220, 38, 38, 180],  # red
    }
    return color_map.get(risk_band, [120, 120, 120, 180])


def process_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    binary_class = 1 means At Risk
    binary_class = 0 means No Risk
    """
    out = df.copy()

    # At Risk = anything above 0.50
    out["binary_class"] = (out["probability"] > LOW_RISK_MIN).astype(int)

    # Distance from the no-risk cutoff
    out["signed_distance_from_cutoff"] = out["probability"] - LOW_RISK_MIN
    out["distance_from_cutoff"] = out["signed_distance_from_cutoff"].abs()

    out["risk_band"] = out["probability"].apply(get_risk_band_from_probability)
    out["color"] = out["risk_band"].apply(get_color_for_risk)

    out["class_label"] = out["binary_class"].map({0: "No Risk", 1: "At Risk"})

    if "ground_truth" in out.columns:
        valid_truth = out["ground_truth"].isin([0, 1])
        out["correct_prediction"] = pd.NA
        out.loc[valid_truth, "correct_prediction"] = (
            out.loc[valid_truth, "ground_truth"].astype(int) == out.loc[valid_truth, "binary_class"]
        )
    else:
        out["correct_prediction"] = pd.NA

    return out


def get_page_slice(df: pd.DataFrame, page_index: int, page_size: int = PAGE_SIZE) -> pd.DataFrame:
    start_idx = page_index * page_size
    end_idx = start_idx + page_size
    return df.iloc[start_idx:end_idx].copy()


def make_map(df: pd.DataFrame):
    if df.empty:
        st.warning("No points available for this page.")
        return

    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())

    lat_range = float(df["latitude"].max() - df["latitude"].min()) if len(df) > 1 else 0
    lon_range = float(df["longitude"].max() - df["longitude"].min()) if len(df) > 1 else 0
    spread = max(lat_range, lon_range)

    if spread > 15:
        zoom = 4.5
    elif spread > 8:
        zoom = 5.2
    elif spread > 4:
        zoom = 6
    elif spread > 2:
        zoom = 7
    else:
        zoom = 8

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius=12000,
        radius_min_pixels=4,
        radius_max_pixels=20,
        pickable=True,
        stroked=True,
        get_line_color=[0, 0, 0, 120],
        line_width_min_pixels=1,
    )

    tooltip = {
        "html": """
        <b>Risk:</b> {risk_band}<br/>
        <b>Ground Truth:</b> {ground_truth_label}<br/>
        <b>Ignition Probability:</b> {probability}<br/>
        <b>Date:</b> {strike_time}<br/>
        <b>Latitude:</b> {latitude}<br/>
        <b>Longitude:</b> {longitude}
        """,
        "style": {
            "backgroundColor": "rgba(30, 30, 30, 0.9)",
            "color": "white"
        }
    }

    deck = pdk.Deck(
        initial_view_state=view_state,
        layers=[layer],
        tooltip=tooltip,
        map_provider="carto",
        map_style="light",
    )

    st.pydeck_chart(deck, use_container_width=True)


# -----------------------------
# Upload page
# -----------------------------
if st.session_state.page == "upload":
    st.title("Lightning Strike Prediction Viewer")
    st.write("Upload a predictions file and visualize fire-risk levels on an interactive map.")

    st.caption(
        "Risk thresholds: "
        f"No Risk <= {LOW_RISK_MIN:.2f}, "
        f"Low Risk > {LOW_RISK_MIN:.2f} and <= {MID_RISK_MIN:.2f}, "
        f"Mid Risk > {MID_RISK_MIN:.2f} and <= {HIGH_RISK_MIN:.2f}, "
        f"High Risk > {HIGH_RISK_MIN:.2f}"
    )

    uploaded_file = st.file_uploader(
        "Drag and drop your predictions file here",
        type=["csv", "xlsx"]
    )

    st.markdown(
        """
        **Required columns:**
        - `strike_lat`
        - `strike_lon`
        - `ignition_probability`

        **Optional but used if present:**
        - `strike_date`
        - `dataset_type`
        """
    )

    if uploaded_file is not None:
        try:
            df = load_uploaded_file(uploaded_file)
            df = standardize_columns(df)

            valid, result = validate_data(df)
            if not valid:
                st.error(result)
            else:
                clean_df, dropped_count, original_count = result
                processed_df = process_predictions(clean_df).reset_index(drop=True)

                st.session_state.data = processed_df
                st.session_state.dropped_count = dropped_count
                st.session_state.original_count = original_count
                st.session_state.page_index = 0
                st.session_state.page = "map"
                st.rerun()

        except Exception as e:
            st.error(f"Could not read/process file: {e}")


# -----------------------------
# Map page
# -----------------------------
elif st.session_state.page == "map":
    df = st.session_state.data

    if df is None or df.empty:
        st.error("No data loaded.")
        if st.button("Return to Upload"):
            st.session_state.page = "upload"
            st.rerun()
    else:
        total_points = len(df)
        total_pages = (total_points - 1) // PAGE_SIZE + 1

        # Keep page index in bounds
        st.session_state.page_index = max(0, min(st.session_state.page_index, total_pages - 1))

        page_df = get_page_slice(df, st.session_state.page_index, PAGE_SIZE)

        start_row = st.session_state.page_index * PAGE_SIZE + 1
        end_row = min((st.session_state.page_index + 1) * PAGE_SIZE, total_points)

        st.title("Mapped Prediction Results")

        st.info(
            f"Loaded {len(df)} usable rows out of {st.session_state.original_count}. "
            f"Dropped {st.session_state.dropped_count} rows missing strike_lat, strike_lon, or ignition_probability."
        )

        st.caption(
            f"Showing rows {start_row:,} to {end_row:,} of {total_points:,} "
            f"(Page {st.session_state.page_index + 1} of {total_pages})"
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Points", total_points)
        col2.metric("At Risk", int((df["binary_class"] == 1).sum()))
        col3.metric("No Risk", int((df["binary_class"] == 0).sum()))
        col4.metric("High Risk", int((df["risk_band"] == "High Risk").sum()))

        nav1, nav2, nav3 = st.columns([1, 1, 2])

        with nav1:
            if st.button("Previous 1000", disabled=(st.session_state.page_index == 0)):
                st.session_state.page_index -= 1
                st.rerun()

        with nav2:
            if st.button("Next 1000", disabled=(st.session_state.page_index >= total_pages - 1)):
                st.session_state.page_index += 1
                st.rerun()

        left, right = st.columns([3, 1])

        with left:
            make_map(page_df)

        with right:
            st.subheader("Legend")
            for band in ["No Risk", "Low Risk", "Mid Risk", "High Risk"]:
                r, g, b, _ = get_color_for_risk(band)
                st.markdown(
                    f'<span style="color:rgb({r},{g},{b})">&#9679;</span> **{band}**',
                    unsafe_allow_html=True,
                )

            st.subheader("Current Page")
            st.write(f"Points shown: {len(page_df)}")

            st.subheader("Actions")
            if st.button("Upload New File"):
                st.session_state.data = None
                st.session_state.page = "upload"
                st.session_state.dropped_count = 0
                st.session_state.original_count = 0
                st.session_state.page_index = 0
                st.rerun()

        st.subheader("Results Table (Current 1000 Points)")

        display_cols = [
            "latitude",
            "longitude",
            "strike_time",
            "probability",
            "risk_band",
            "ground_truth_label",
            "binary_class",
            "distance_from_cutoff",
        ]

        if "ground_truth" in page_df.columns:
            display_cols.append("ground_truth")
        if "correct_prediction" in page_df.columns:
            display_cols.append("correct_prediction")

        existing_display_cols = [c for c in display_cols if c in page_df.columns]

        st.dataframe(
            page_df[existing_display_cols],
            use_container_width=True
        )

        output_cols = [
            "latitude",
            "longitude",
            "strike_time",
            "probability",
            "risk_band",
            "ground_truth_label",
            "binary_class",
            "distance_from_cutoff",
        ]

        if "ground_truth" in df.columns:
            output_cols.append("ground_truth")
        if "correct_prediction" in df.columns:
            output_cols.append("correct_prediction")

        existing_output_cols = [c for c in output_cols if c in df.columns]

        csv = df[existing_output_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Full Results CSV",
            data=csv,
            file_name="viewed_predictions.csv",
            mime="text/csv"
        )
