import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Perbandingan Fuzzy Logic vs RNN",
    page_icon="📊",
    layout="wide"
)

# Sidebar configuration
st.sidebar.title("⚙️ Konfigurasi")
st.sidebar.markdown("---")

# Fire scale thresholds
st.sidebar.subheader("🔥 Threshold Skala Kebakaran")
small_threshold = st.sidebar.slider(
    "Batas Kecil-Sedang",
    min_value=20,
    max_value=50,
    value=34,
    help="Nilai di bawah ini dikategorikan sebagai 'Kecil'"
)
medium_threshold = st.sidebar.slider(
    "Batas Sedang-Besar",
    min_value=50,
    max_value=80,
    value=67,
    help="Nilai di atas ini dikategorikan sebagai 'Besar'"
)

# Update classify function based on sidebar values
def classify_fire_scale(value):
    if value < small_threshold:
        return 'Kecil'
    elif value < medium_threshold:
        return 'Sedang'
    else:
        return 'Besar'

# Display settings
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Pengaturan Tampilan")
show_debug = st.sidebar.checkbox("Tampilkan Debug Info", value=False)
max_display_rows = st.sidebar.number_input(
    "Max Baris Tampilan",
    min_value=10,
    max_value=1000,
    value=20,
    step=10
)

# About section
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Tentang Aplikasi")
st.sidebar.info(
    """
    Aplikasi ini membandingkan:
    - **Fuzzy Logic**: Sistem berbasis aturan
    - **RNN**: Neural network untuk time series
    
    Untuk prediksi risiko kebakaran berdasarkan:
    - Suhu (°C)
    - Kelembapan (%)
    - Intensitas Api (%)
    """
)

st.title(":thermometer: Prediksi Kebakaran: Fuzzy Logic vs RNN")
st.markdown("**Prediksi berdasarkan Kelembapan, Suhu, dan Intensitas Api**")

# Fungsi bantu
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def classify_fire_scale(value):
    if value < 34:
        return 'Kecil'
    elif value < 67:
        return 'Sedang'
    else:
        return 'Besar'

# Fuzzy Logic Controller
class FuzzyController:
    def __init__(self):
        self.temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
        self.humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
        self.fire_intensity = ctrl.Antecedent(np.arange(0, 101, 1), 'fire_intensity')
        self.output = ctrl.Consequent(np.arange(0, 101, 1), 'output')

        # Membership functions
        self.temperature['low'] = fuzz.trimf(self.temperature.universe, [0, 0, 40])
        self.temperature['medium'] = fuzz.trimf(self.temperature.universe, [20, 50, 80])
        self.temperature['high'] = fuzz.trimf(self.temperature.universe, [60, 100, 100])

        self.humidity['low'] = fuzz.trimf(self.humidity.universe, [0, 0, 40])
        self.humidity['medium'] = fuzz.trimf(self.humidity.universe, [20, 50, 80])
        self.humidity['high'] = fuzz.trimf(self.humidity.universe, [60, 100, 100])

        self.fire_intensity['low'] = fuzz.trimf(self.fire_intensity.universe, [0, 0, 40])
        self.fire_intensity['medium'] = fuzz.trimf(self.fire_intensity.universe, [20, 50, 80])
        self.fire_intensity['high'] = fuzz.trimf(self.fire_intensity.universe, [60, 100, 100])

        self.output['very_low'] = fuzz.trimf(self.output.universe, [0, 0, 25])
        self.output['low'] = fuzz.trimf(self.output.universe, [0, 25, 50])
        self.output['medium'] = fuzz.trimf(self.output.universe, [25, 50, 75])
        self.output['high'] = fuzz.trimf(self.output.universe, [50, 75, 100])
        self.output['very_high'] = fuzz.trimf(self.output.universe, [75, 100, 100])

        self.rules = [
            ctrl.Rule(self.temperature['low'] & self.humidity['low'] & self.fire_intensity['low'], self.output['very_low']),
            ctrl.Rule(self.temperature['low'] & self.humidity['low'] & self.fire_intensity['medium'], self.output['low']),
            ctrl.Rule(self.temperature['low'] & self.humidity['medium'] & self.fire_intensity['low'], self.output['low']),
            ctrl.Rule(self.temperature['medium'] & self.humidity['low'] & self.fire_intensity['low'], self.output['low']),
            ctrl.Rule(self.temperature['medium'] & self.humidity['medium'] & self.fire_intensity['medium'], self.output['medium']),
            ctrl.Rule(self.temperature['high'] & self.humidity['high'] & self.fire_intensity['high'], self.output['very_high']),
            ctrl.Rule(self.temperature['high'] & self.humidity['medium'] & self.fire_intensity['high'], self.output['high']),
            ctrl.Rule(self.temperature['medium'] & self.humidity['high'] & self.fire_intensity['medium'], self.output['high']),
            ctrl.Rule(self.temperature['low'] & self.humidity['high'] & self.fire_intensity['high'], self.output['medium']),
        ]

        self.control_system = ctrl.ControlSystem(self.rules)
        self.controller = ctrl.ControlSystemSimulation(self.control_system)

    def predict(self, temp, hum, fire):
        try:
            self.controller.input['temperature'] = temp
            self.controller.input['humidity'] = hum
            self.controller.input['fire_intensity'] = fire
            self.controller.compute()
            return self.controller.output['output']
        except:
            return 50

# --------------------------------------------
# Section: Manual Input & Prediction
# --------------------------------------------

st.header("🎛️ Prediksi Manual")
st.markdown("Masukkan nilai parameter untuk mendapatkan prediksi kebakaran")

col1, col2, col3 = st.columns(3)

with col1:
    manual_temp = st.slider(
        "Suhu (°C)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.1,
        help="Masukkan nilai suhu dalam Celsius"
    )

with col2:
    manual_humidity = st.slider(
        "Kelembapan (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.1,
        help="Masukkan nilai kelembapan dalam persen"
    )

with col3:
    manual_fire_intensity = st.slider(
        "Intensitas Api (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.1,
        help="Masukkan nilai intensitas api dalam persen"
    )

# Initialize Fuzzy Controller
fuzzy_controller = FuzzyController()

# Predict button
if st.button("🔮 Prediksi Sekarang", type="primary"):
    # Fuzzy prediction
    fuzzy_result = fuzzy_controller.predict(manual_temp, manual_humidity, manual_fire_intensity)
    fuzzy_scale = classify_fire_scale(fuzzy_result)
    
    # Display results
    st.success("✅ Hasil Prediksi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Fuzzy Logic Output",
            value=f"{fuzzy_result:.2f}",
            delta=f"Skala: {fuzzy_scale}"
        )
        
        # Visual indicator
        if fuzzy_result < 34:
            st.info("🟢 Risiko Kebakaran: **RENDAH**")
        elif fuzzy_result < 67:
            st.warning("🟡 Risiko Kebakaran: **SEDANG**")
        else:
            st.error("🔴 Risiko Kebakaran: **TINGGI**")
    
    with col2:
        st.info("ℹ️ **Catatan**: Untuk membandingkan dengan RNN, upload file CSV hasil prediksi RNN di bawah.")
        
    # Create a dataframe for the current prediction
    current_prediction = pd.DataFrame({
        'Temperature_C': [manual_temp],
        'Humidity_Percent': [manual_humidity],
        'Fire_Intensity_Percent': [manual_fire_intensity],
        'Fuzzy_Prediction': [fuzzy_result],
        'Fuzzy_Scale': [fuzzy_scale]
    })
    
    # Option to save prediction
    st.markdown("---")
    st.subheader("💾 Simpan Prediksi")
    
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = pd.DataFrame()
    
    if st.button("Tambah ke Riwayat"):
        st.session_state.prediction_history = pd.concat(
            [st.session_state.prediction_history, current_prediction], 
            ignore_index=True
        )
        st.success("✅ Prediksi ditambahkan ke riwayat!")
    
    # Show prediction history if exists
    if not st.session_state.prediction_history.empty:
        st.subheader("📊 Riwayat Prediksi")
        st.dataframe(st.session_state.prediction_history)
        
        # Download history
        csv_history = convert_df_to_csv(st.session_state.prediction_history)
        st.download_button(
            label="📥 Download Riwayat Prediksi",
            data=csv_history,
            file_name="riwayat_prediksi_fuzzy.csv",
            mime="text/csv"
        )
        
        # Clear history button
        if st.button("🗑️ Hapus Riwayat"):
            st.session_state.prediction_history = pd.DataFrame()
            st.rerun()

st.markdown("---")

# --------------------------------------------
# Section: Batch Manual Prediction
# --------------------------------------------

with st.expander("🔢 Prediksi Batch (Multiple Values)", expanded=False):
    st.markdown("Masukkan beberapa nilai sekaligus untuk prediksi batch")
    
    # Create input dataframe
    st.subheader("Input Data")
    
    # Number of rows to predict
    n_rows = st.number_input("Jumlah data yang ingin diprediksi", min_value=1, max_value=50, value=5)
    
    # Create empty dataframe for input
    batch_data = pd.DataFrame({
        'Temperature_C': [50.0] * n_rows,
        'Humidity_Percent': [50.0] * n_rows,
        'Fire_Intensity_Percent': [50.0] * n_rows
    })
    
    # Edit the dataframe
    edited_data = st.data_editor(
        batch_data,
        num_rows="fixed",
        column_config={
            "Temperature_C": st.column_config.NumberColumn(
                "Suhu (°C)",
                help="Masukkan suhu dalam Celsius (0-100)",
                min_value=0,
                max_value=100,
                step=0.1,
                format="%.1f"
            ),
            "Humidity_Percent": st.column_config.NumberColumn(
                "Kelembapan (%)",
                help="Masukkan kelembapan dalam persen (0-100)",
                min_value=0,
                max_value=100,
                step=0.1,
                format="%.1f"
            ),
            "Fire_Intensity_Percent": st.column_config.NumberColumn(
                "Intensitas Api (%)",
                help="Masukkan intensitas api dalam persen (0-100)",
                min_value=0,
                max_value=100,
                step=0.1,
                format="%.1f"
            )
        }
    )
    
    # Predict batch
    if st.button("🚀 Prediksi Batch", type="secondary"):
        with st.spinner("Memproses prediksi..."):
            # Initialize Fuzzy Controller for batch
            batch_fuzzy = FuzzyController()
            
            # Predict for each row
            predictions = []
            for idx, row in edited_data.iterrows():
                pred = batch_fuzzy.predict(
                    row['Temperature_C'],
                    row['Humidity_Percent'],
                    row['Fire_Intensity_Percent']
                )
                predictions.append(pred)
            
            # Add predictions to dataframe
            result_df = edited_data.copy()
            result_df['Fuzzy_Prediction'] = predictions
            result_df['Fuzzy_Scale'] = result_df['Fuzzy_Prediction'].apply(classify_fire_scale)
            
            # Display results
            st.success("✅ Prediksi Batch Selesai!")
            st.dataframe(
                result_df,
                column_config={
                    "Fuzzy_Prediction": st.column_config.NumberColumn(
                        "Prediksi Output",
                        format="%.2f"
                    ),
                    "Fuzzy_Scale": st.column_config.TextColumn(
                        "Skala Kebakaran"
                    )
                }
            )
            
            # Statistics
            st.subheader("📊 Statistik Hasil")
            col1, col2, col3 = st.columns(3)
            with col1:
                kecil_count = (result_df['Fuzzy_Scale'] == 'Kecil').sum()
                st.metric("🟢 Skala Kecil", kecil_count)
            with col2:
                sedang_count = (result_df['Fuzzy_Scale'] == 'Sedang').sum()
                st.metric("🟡 Skala Sedang", sedang_count)
            with col3:
                besar_count = (result_df['Fuzzy_Scale'] == 'Besar').sum()
                st.metric("🔴 Skala Besar", besar_count)
            
            # Download batch results
            csv_batch = convert_df_to_csv(result_df)
            st.download_button(
                label="📥 Download Hasil Batch",
                data=csv_batch,
                file_name="hasil_prediksi_batch_fuzzy.csv",
                mime="text/csv"
            )

st.markdown("---")

# --------------------------------------------
# Section: Upload CSV & Compare
# --------------------------------------------

st.header("📄 Load Hasil Prediksi dari CSV")

uploaded_file_fuzzy = st.file_uploader("Unggah file hasil prediksi Fuzzy Logic (CSV)", type="csv")
uploaded_file_rnn = st.file_uploader("Unggah file hasil prediksi RNN (CSV)", type="csv")

if uploaded_file_fuzzy and uploaded_file_rnn:
    fuzzy_df = pd.read_csv(uploaded_file_fuzzy)
    rnn_df = pd.read_csv(uploaded_file_rnn)

    # Clean up the column names (strip spaces and convert to lowercase)
    fuzzy_df.columns = fuzzy_df.columns.str.strip().str.lower()
    rnn_df.columns = rnn_df.columns.str.strip().str.lower()

    # Display original column names for debugging
    if show_debug:
        st.write("### Debug Information:")
        st.write(f"**Fuzzy Logic Columns:** {list(fuzzy_df.columns)}")
        st.write(f"**RNN Columns:** {list(rnn_df.columns)}")

    # Check if required columns exist before merging
    required_columns = ['temperature_c', 'humidity_percent', 'fire_intensity_percent']
    fuzzy_has_all = all(col in fuzzy_df.columns for col in required_columns)
    rnn_has_all = all(col in rnn_df.columns for col in required_columns)

    if fuzzy_has_all and rnn_has_all:
        # Merge the dataframes
        merged_df = pd.merge(
            fuzzy_df, 
            rnn_df, 
            on=['temperature_c', 'humidity_percent', 'fire_intensity_percent'], 
            how='inner',
            suffixes=('_fuzzy', '_rnn')
        )

        # Display the columns of the merged dataframe for debugging
        if show_debug:
            st.write(f"**Columns in Merged DataFrame:** {list(merged_df.columns)}")

        # Handle the target_output column
        # After merging, we might have target_output_fuzzy and target_output_rnn
        # We need to check which one exists or if they're the same
        if 'target_output' in merged_df.columns:
            target_col = 'target_output'
        elif 'target_output_fuzzy' in merged_df.columns:
            target_col = 'target_output_fuzzy'
            # If both exist and they're the same, we can drop one
            if 'target_output_rnn' in merged_df.columns:
                merged_df['target_output'] = merged_df[target_col]
                target_col = 'target_output'
        else:
            st.error("❌ Column 'target_output' not found in the merged dataframe!")
            st.stop()

        # Check if prediction columns exist
        if 'fuzzy_prediction' not in merged_df.columns:
            if 'fuzzy_prediction_fuzzy' in merged_df.columns:
                merged_df['fuzzy_prediction'] = merged_df['fuzzy_prediction_fuzzy']
            else:
                st.error("❌ Column 'fuzzy_prediction' not found!")
                st.stop()

        if 'rnn_prediction' not in merged_df.columns:
            if 'rnn_prediction_rnn' in merged_df.columns:
                merged_df['rnn_prediction'] = merged_df['rnn_prediction_rnn']
            else:
                st.error("❌ Column 'rnn_prediction' not found!")
                st.stop()

        # Klasifikasi Skala Kebakaran
        merged_df['target_skala'] = merged_df[target_col].apply(classify_fire_scale)
        merged_df['fuzzy_skala'] = merged_df['fuzzy_prediction'].apply(classify_fire_scale)
        merged_df['rnn_skala'] = merged_df['rnn_prediction'].apply(classify_fire_scale)

        st.subheader(":mag: Perbandingan Skala Kebakaran")
        display_df = merged_df[['temperature_c', 'humidity_percent', 'fire_intensity_percent',
                                'target_skala', 'fuzzy_skala', 'rnn_skala']].head(max_display_rows)
        st.dataframe(display_df)

        # Calculate accuracy
        fuzzy_accuracy = (merged_df['fuzzy_skala'] == merged_df['target_skala']).mean() * 100
        rnn_accuracy = (merged_df['rnn_skala'] == merged_df['target_skala']).mean() * 100

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fuzzy Logic Accuracy", f"{fuzzy_accuracy:.2f}%")
        with col2:
            st.metric("RNN Accuracy", f"{rnn_accuracy:.2f}%")

        # ============================================
        # Section: Confusion Matrix
        # ============================================
        st.subheader("🎯 Confusion Matrix")
        
        # Create confusion matrices
        classes = ['Kecil', 'Sedang', 'Besar']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Fuzzy Logic Confusion Matrix**")
            cm_fuzzy = confusion_matrix(merged_df['target_skala'], merged_df['fuzzy_skala'], labels=classes)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_fuzzy, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Fuzzy Logic Confusion Matrix')
            st.pyplot(fig)
            
        with col2:
            st.markdown("**RNN Confusion Matrix**")
            cm_rnn = confusion_matrix(merged_df['target_skala'], merged_df['rnn_skala'], labels=classes)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_rnn, annot=True, fmt='d', cmap='Greens', 
                       xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('RNN Confusion Matrix')
            st.pyplot(fig)

        # ============================================
        # Section: Classification Report
        # ============================================
        st.subheader("📊 Classification Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Fuzzy Logic Metrics**")
            # Calculate metrics
            precision_f, recall_f, f1_f, _ = precision_recall_fscore_support(
                merged_df['target_skala'], merged_df['fuzzy_skala'], 
                labels=classes, average='weighted'
            )
            
            metrics_df_fuzzy = pd.DataFrame({
                'Metric': ['Precision', 'Recall', 'F1-Score'],
                'Value': [f"{precision_f:.3f}", f"{recall_f:.3f}", f"{f1_f:.3f}"]
            })
            st.dataframe(metrics_df_fuzzy, hide_index=True)
            
        with col2:
            st.markdown("**RNN Metrics**")
            # Calculate metrics
            precision_r, recall_r, f1_r, _ = precision_recall_fscore_support(
                merged_df['target_skala'], merged_df['rnn_skala'], 
                labels=classes, average='weighted'
            )
            
            metrics_df_rnn = pd.DataFrame({
                'Metric': ['Precision', 'Recall', 'F1-Score'],
                'Value': [f"{precision_r:.3f}", f"{recall_r:.3f}", f"{f1_r:.3f}"]
            })
            st.dataframe(metrics_df_rnn, hide_index=True)

        # ============================================
        # Section: Regression Metrics
        # ============================================
        st.subheader("📈 Regression Metrics")
        
        # Calculate regression metrics
        mae_fuzzy = mean_absolute_error(merged_df[target_col], merged_df['fuzzy_prediction'])
        mse_fuzzy = mean_squared_error(merged_df[target_col], merged_df['fuzzy_prediction'])
        rmse_fuzzy = np.sqrt(mse_fuzzy)
        r2_fuzzy = r2_score(merged_df[target_col], merged_df['fuzzy_prediction'])
        
        mae_rnn = mean_absolute_error(merged_df[target_col], merged_df['rnn_prediction'])
        mse_rnn = mean_squared_error(merged_df[target_col], merged_df['rnn_prediction'])
        rmse_rnn = np.sqrt(mse_rnn)
        r2_rnn = r2_score(merged_df[target_col], merged_df['rnn_prediction'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Fuzzy Logic**")
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.metric("MAE", f"{mae_fuzzy:.3f}")
                st.metric("MSE", f"{mse_fuzzy:.3f}")
            with col1_2:
                st.metric("RMSE", f"{rmse_fuzzy:.3f}")
                st.metric("R² Score", f"{r2_fuzzy:.3f}")
                
        with col2:
            st.markdown("**RNN**")
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("MAE", f"{mae_rnn:.3f}")
                st.metric("MSE", f"{mse_rnn:.3f}")
            with col2_2:
                st.metric("RMSE", f"{rmse_rnn:.3f}")
                st.metric("R² Score", f"{r2_rnn:.3f}")

        # ============================================
        # Section: Error Distribution
        # ============================================
        st.subheader("📊 Error Distribution Analysis")
        
        # Calculate errors if not in dataframe
        if 'fuzzy_error' not in merged_df.columns:
            merged_df['fuzzy_error'] = abs(merged_df[target_col] - merged_df['fuzzy_prediction'])
        if 'rnn_error' not in merged_df.columns:
            merged_df['rnn_error'] = abs(merged_df[target_col] - merged_df['rnn_prediction'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of errors
        ax1.hist(merged_df['fuzzy_error'], bins=30, alpha=0.6, label='Fuzzy Error', color='blue')
        ax1.hist(merged_df['rnn_error'], bins=30, alpha=0.6, label='RNN Error', color='green')
        ax1.set_xlabel('Absolute Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Error Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot of errors
        error_data = pd.DataFrame({
            'Fuzzy Logic': merged_df['fuzzy_error'],
            'RNN': merged_df['rnn_error']
        })
        error_data.boxplot(ax=ax2)
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Error Distribution Box Plot')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)

        # ============================================
        # Section: Scatter Plot - Predictions vs Actual
        # ============================================
        st.subheader("🔍 Predictions vs Actual Values")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Fuzzy Logic scatter plot
        ax1.scatter(merged_df[target_col], merged_df['fuzzy_prediction'], alpha=0.5, s=20)
        ax1.plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Output')
        ax1.set_ylabel('Fuzzy Prediction')
        ax1.set_title('Fuzzy Logic: Predicted vs Actual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        
        # RNN scatter plot
        ax2.scatter(merged_df[target_col], merged_df['rnn_prediction'], alpha=0.5, s=20, color='green')
        ax2.plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Output')
        ax2.set_ylabel('RNN Prediction')
        ax2.set_title('RNN: Predicted vs Actual')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 100)
        
        st.pyplot(fig)

        # ============================================
        # Section: Performance by Fire Scale
        # ============================================
        st.subheader("🔥 Performance Analysis by Fire Scale")
        
        # Calculate accuracy for each class
        scale_performance = []
        for scale in classes:
            mask = merged_df['target_skala'] == scale
            if mask.sum() > 0:
                fuzzy_acc = (merged_df.loc[mask, 'fuzzy_skala'] == scale).mean() * 100
                rnn_acc = (merged_df.loc[mask, 'rnn_skala'] == scale).mean() * 100
                count = mask.sum()
                scale_performance.append({
                    'Scale': scale,
                    'Count': count,
                    'Fuzzy Accuracy': fuzzy_acc,
                    'RNN Accuracy': rnn_acc
                })
        
        perf_df = pd.DataFrame(scale_performance)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(perf_df))
            width = 0.35
            
            ax.bar(x - width/2, perf_df['Fuzzy Accuracy'], width, label='Fuzzy Logic', color='skyblue')
            ax.bar(x + width/2, perf_df['RNN Accuracy'], width, label='RNN', color='lightgreen')
            
            ax.set_xlabel('Fire Scale')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Accuracy by Fire Scale Category')
            ax.set_xticks(x)
            ax.set_xticklabels(perf_df['Scale'])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (fuzzy, rnn) in enumerate(zip(perf_df['Fuzzy Accuracy'], perf_df['RNN Accuracy'])):
                ax.text(i - width/2, fuzzy + 1, f'{fuzzy:.1f}%', ha='center', va='bottom')
                ax.text(i + width/2, rnn + 1, f'{rnn:.1f}%', ha='center', va='bottom')
            
            st.pyplot(fig)
            
        with col2:
            st.dataframe(perf_df, hide_index=True)

        st.subheader(":bar_chart: Grafik Prediksi")
        # Create a subset for visualization (first 100 rows for clarity)
        viz_df = merged_df.head(100)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(viz_df.index, viz_df[target_col], label='Target Output', linewidth=2)
        ax.plot(viz_df.index, viz_df['fuzzy_prediction'], label='Fuzzy Prediction', alpha=0.7)
        ax.plot(viz_df.index, viz_df['rnn_prediction'], label='RNN Prediction', alpha=0.7)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Fire Intensity Output')
        ax.set_title('Comparison of Target vs Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader(":inbox_tray: Unduh Hasil Klasifikasi")
        # Prepare download dataframe
        download_df = merged_df[['temperature_c', 'humidity_percent', 'fire_intensity_percent',
                                 target_col, 'fuzzy_prediction', 'rnn_prediction',
                                 'target_skala', 'fuzzy_skala', 'rnn_skala',
                                 'fuzzy_error', 'rnn_error']]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📅 Download Hasil Klasifikasi (CSV)",
                data=convert_df_to_csv(download_df),
                file_name="hasil_klasifikasi_kebakaran.csv",
                mime="text/csv"
            )
        
        with col2:
            # Generate summary report
            summary_report = f"""
LAPORAN PERBANDINGAN MODEL PREDIKSI KEBAKARAN
=============================================

Tanggal: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Data: {len(merged_df)} samples

HASIL KLASIFIKASI:
-----------------
Fuzzy Logic Accuracy: {fuzzy_accuracy:.2f}%
RNN Accuracy: {rnn_accuracy:.2f}%

REGRESSION METRICS:
------------------
Fuzzy Logic:
- MAE: {mae_fuzzy:.3f}
- MSE: {mse_fuzzy:.3f}
- RMSE: {rmse_fuzzy:.3f}
- R² Score: {r2_fuzzy:.3f}

RNN:
- MAE: {mae_rnn:.3f}
- MSE: {mse_rnn:.3f}
- RMSE: {rmse_rnn:.3f}
- R² Score: {r2_rnn:.3f}

CLASSIFICATION METRICS:
----------------------
Fuzzy Logic:
- Precision: {precision_f:.3f}
- Recall: {recall_f:.3f}
- F1-Score: {f1_f:.3f}

RNN:
- Precision: {precision_r:.3f}
- Recall: {recall_r:.3f}
- F1-Score: {f1_r:.3f}

KESIMPULAN:
-----------
Model terbaik berdasarkan accuracy: {'Fuzzy Logic' if fuzzy_accuracy > rnn_accuracy else 'RNN'}
Model terbaik berdasarkan MAE: {'Fuzzy Logic' if mae_fuzzy < mae_rnn else 'RNN'}
Model terbaik berdasarkan R² Score: {'Fuzzy Logic' if r2_fuzzy > r2_rnn else 'RNN'}
            """
            
            st.download_button(
                label="📄 Download Laporan Summary (TXT)",
                data=summary_report,
                file_name="laporan_perbandingan_model.txt",
                mime="text/plain"
            )
        
        with col3:
            # Create performance comparison dataframe
            comparison_df = pd.DataFrame({
                'Metric': ['Accuracy (%)', 'MAE', 'MSE', 'RMSE', 'R² Score', 'Precision', 'Recall', 'F1-Score'],
                'Fuzzy Logic': [
                    f"{fuzzy_accuracy:.2f}",
                    f"{mae_fuzzy:.3f}",
                    f"{mse_fuzzy:.3f}",
                    f"{rmse_fuzzy:.3f}",
                    f"{r2_fuzzy:.3f}",
                    f"{precision_f:.3f}",
                    f"{recall_f:.3f}",
                    f"{f1_f:.3f}"
                ],
                'RNN': [
                    f"{rnn_accuracy:.2f}",
                    f"{mae_rnn:.3f}",
                    f"{mse_rnn:.3f}",
                    f"{rmse_rnn:.3f}",
                    f"{r2_rnn:.3f}",
                    f"{precision_r:.3f}",
                    f"{recall_r:.3f}",
                    f"{f1_r:.3f}"
                ]
            })
            
            st.download_button(
                label="📊 Download Tabel Perbandingan (CSV)",
                data=convert_df_to_csv(comparison_df),
                file_name="tabel_perbandingan_metrics.csv",
                mime="text/csv"
            )
    else:
        st.error("❌ Required columns not found in one or both CSV files!")
        st.write("Required columns: temperature_c, humidity_percent, fire_intensity_percent")
        if not fuzzy_has_all:
            st.write(f"Missing in Fuzzy CSV: {[col for col in required_columns if col not in fuzzy_df.columns]}")
        if not rnn_has_all:
            st.write(f"Missing in RNN CSV: {[col for col in required_columns if col not in rnn_df.columns]}")
else:
    st.info("📁 Silakan unggah kedua file CSV untuk melihat hasil perbandingan.")
    st.markdown("""
    ### File yang dibutuhkan:
    1. **Fuzzy Logic Results** - File CSV dengan kolom:
       - Temperature_C
       - Humidity_Percent
       - Fire_Intensity_Percent
       - Target_Output
       - Fuzzy_Prediction
       - Fuzzy_Error (optional)
    
    2. **RNN Results** - File CSV dengan kolom:
       - Temperature_C
       - Humidity_Percent
       - Fire_Intensity_Percent
       - Target_Output
       - RNN_Prediction
       - RNN_Error (optional)
    """)