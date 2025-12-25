# Coastline Prediction System v3.0

## Sea Level Rise Impact Analysis using Machine Learning Ensemble Models

[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b+-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

### 📋 Project Description

This project develops a MATLAB-based system to calculate and visualize coastline changes due to rising sea levels. Using historical land area data and digital elevation models, the system employs ensemble machine learning techniques to predict future coastlines under various IPCC climate scenarios.

### 🎯 Motivation

Climate change poses a significant risk to coastal areas as rising global temperatures produce rising sea levels. According to NOAA, average global sea level rose 2.6 inches between 1993 and 2014. This tool helps communities assess the potential impact of rising sea levels with applications for:
- Insurance risk assessment
- Zoning and urban planning
- Construction permitting
- Potential relocation planning

### 🚀 Features

- **Ensemble ML Models**: Linear, Quadratic, Cubic, Robust Linear, and Exponential regression
- **Cross-Validation**: K-fold validation for model performance assessment
- **Uncertainty Quantification**: Bootstrap confidence intervals (1000 iterations)
- **IPCC Scenarios**: SSP1-2.6, SSP2-4.5, SSP5-8.5 sea level rise projections
- **Inundation Simulation**: Physics-based flooding model using DEM
- **Comprehensive Visualization**: 
  - Model comparison dashboard
  - Spatial flood maps
  - 3D terrain visualization
  - Rate of change analysis

### 📁 Project Structure

```
CoastLine Project/
├── Final_Project_Model.m          # Main MATLAB script
├── Dataset/
│   ├── Coastline_TimeSeries_41Years.csv   # Historical land area data
│   ├── Elevation_Matrix.tif               # Digital Elevation Model (DEM)
│   └── Visual_Base.tif                    # Satellite/Base image
└── README.md                      # This file
```

### 📊 Required Data Files

| File | Description | Format |
|------|-------------|--------|
| `Coastline_TimeSeries_41Years.csv` | Historical land area measurements | CSV with `year` and `land_area_m2` columns |
| `Elevation_Matrix.tif` | Digital Elevation Model | GeoTIFF (meters) |
| `Visual_Base.tif` | Satellite/aerial image | GeoTIFF (RGB) |

### 🔧 Requirements

- MATLAB R2020b or later
- Required Toolboxes:
  - Statistics and Machine Learning Toolbox
  - Image Processing Toolbox
  - Mapping Toolbox (optional, for advanced GIS features)

### 💻 Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/coastline-prediction.git
   ```

2. Open MATLAB and navigate to the project folder:
   ```matlab
   cd 'path/to/CoastLine Project'
   ```

3. Run the main script:
   ```matlab
   Final_Project_Model
   ```

### 📈 Output

The system generates:
1. **Console Report**: Detailed predictions with confidence intervals
2. **Figure 1**: Model Analysis Dashboard (6 panels)
3. **Figure 2**: Spatial Inundation Analysis (3 maps)
4. **Figure 3**: 3D Terrain Visualization

### 🌍 IPCC Scenarios

| Scenario | Description | Sea Level Rise by 2050 |
|----------|-------------|------------------------|
| SSP1-2.6 | Sustainability | +0.3 m |
| SSP2-4.5 | Middle of the Road | +0.5 m |
| SSP5-8.5 | Fossil-fueled Development | +1.0 m |

### 📚 Data Sources

- **Elevation Data**: [USGS National Map](https://apps.nationalmap.gov/downloader/) or [Google Earth Engine](https://earthengine.google.com/)
- **Coastline Data**: [NOAA Digital Coast](https://coast.noaa.gov/digitalcoast/)
- **Sea Level Projections**: [IPCC AR6](https://www.ipcc.ch/report/ar6/wg1/)

### 🧮 Model Performance Metrics

The ensemble model combines 5 regression models weighted by inverse RMSE:
- **R² Score**: Coefficient of determination (0-1)
- **RMSE**: Root Mean Square Error (km²)
- **CV-RMSE**: Cross-validated RMSE

### 📖 References

1. IPCC, 2021: Climate Change 2021: The Physical Science Basis
2. NOAA Sea Level Rise Technical Report, 2022
3. MathWorks Mapping Toolbox Documentation

### 👤 Author

**Surya Gunjapalli**

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Developed as part of MATLAB academic project for climate change impact assessment.*
