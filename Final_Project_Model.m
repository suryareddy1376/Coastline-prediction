function Final_Project_Model
    % =========================================================
    % COASTLINE PREDICTION SYSTEM v3.0 (PRODUCTION)
    % Sea Level Rise Impact Analysis using ML Ensemble Models
    % =========================================================
    % Author: Surya Gunjapalli
    % Project: Climate Change Coastline Prediction
    % =========================================================
    % Required Input Files (in Dataset/ folder):
    %   1. Dataset/Coastline_TimeSeries_41Years.csv - Historical land area data
    %   2. Dataset/Elevation_Matrix.tif             - Digital Elevation Model
    %   3. Dataset/Visual_Base.tif                  - Satellite/Base image
    % =========================================================
    
    clc;
    fprintf('╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║     COASTLINE PREDICTION SYSTEM v3.0 (PRODUCTION)        ║\n');
    fprintf('║     Sea Level Rise Impact Analysis                       ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n\n');
    
    %% 1. CONFIGURATION
    config = struct();
    config.target_year = 2100;
    config.pixel_resolution = 30;           % meters per pixel
    config.confidence_level = 0.95;         % 95% confidence interval
    config.n_bootstrap = 1000;              % Bootstrap iterations
    config.ipcc_scenarios = {'SSP1-2.6', 'SSP2-4.5', 'SSP5-8.5'};
    config.slr_rates = [0.3, 0.5, 1.0];     % Sea level rise (meters) by 2050
    
    %% 2. DATA LOADING AND VALIDATION
    fprintf('[1/6] Loading and Validating Input Data...\n');
    fprintf('─────────────────────────────────────────────\n');
    
    [data_loaded, years, area, dem, sat_img] = loadRealData();
    
    if ~data_loaded
        fprintf('\n⛔ CRITICAL: Cannot proceed without required data files.\n');
        fprintf('   Please ensure the following files exist in the Dataset/ folder:\n');
        fprintf('   • Dataset/Coastline_TimeSeries_41Years.csv\n');
        fprintf('   • Dataset/Elevation_Matrix.tif\n');
        fprintf('   • Dataset/Visual_Base.tif\n\n');
        return;
    end
    
    %% 3. DATA PREPROCESSING
    fprintf('\n[2/6] Preprocessing Data...\n');
    fprintf('─────────────────────────────────────────────\n');
    [years, area] = preprocessData(years, area);
    
    fprintf('   ✓ Data range: %d - %d (%d data points)\n', min(years), max(years), length(years));
    fprintf('   ✓ Area range: %.2f - %.2f km²\n', min(area), max(area));
    fprintf('   ✓ Current area (latest): %.2f km²\n', area(end));
    
    %% 4. MODEL TRAINING
    fprintf('\n[3/6] Training Ensemble Models...\n');
    fprintf('─────────────────────────────────────────────\n');
    models = trainModels(years, area);
    displayModelMetrics(models);
    
    %% 5. PREDICTION WITH UNCERTAINTY
    fprintf('\n[4/6] Generating Predictions...\n');
    fprintf('─────────────────────────────────────────────\n');
    [predictions, uncertainty] = generatePredictions(models, years, area, config);
    
    %% 6. INUNDATION SIMULATION
    fprintf('\n[5/6] Simulating Inundation...\n');
    fprintf('─────────────────────────────────────────────\n');
    [flood_masks, sea_levels] = simulateFlooding(dem, predictions, config);
    
    %% 7. VISUALIZATION
    fprintf('\n[6/6] Creating Visualizations...\n');
    fprintf('─────────────────────────────────────────────\n');
    createVisualizations(years, area, models, predictions, uncertainty, ...
                        flood_masks, sat_img, dem, config);
    
    %% 8. FINAL REPORT
    printFinalReport(predictions, uncertainty, sea_levels, config);
    
    fprintf('\n╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║              ANALYSIS COMPLETED SUCCESSFULLY              ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n');
end

%% ═══════════════════════════════════════════════════════════════════════
%  DATA LOADING FUNCTIONS
%% ═══════════════════════════════════════════════════════════════════════

function [success, years, area, dem, sat_img] = loadRealData()
    % Load real data files - NO synthetic data generation
    
    success = false;
    years = []; area = []; dem = []; sat_img = [];
    
    % --- Load Time Series CSV ---
    csv_file = findCSVFile();
    if isempty(csv_file)
        fprintf('   ✗ ERROR: No CSV file found with coastline time series data.\n');
        fprintf('     Expected: CSV file with columns for "year" and "area" (or similar)\n');
        return;
    end
    
    fprintf('   ✓ Found CSV: %s\n', csv_file);
    
    try
        data = readtable(csv_file, 'VariableNamingRule', 'preserve');
        [years, area] = extractTimeSeriesData(data);
        
        if isempty(years) || isempty(area)
            fprintf('   ✗ ERROR: Could not extract year/area columns from CSV.\n');
            fprintf('     Available columns: %s\n', strjoin(data.Properties.VariableNames, ', '));
            return;
        end
        fprintf('   ✓ Loaded %d records from CSV\n', length(years));
    catch ME
        fprintf('   ✗ ERROR reading CSV: %s\n', ME.message);
        return;
    end
    
    % --- Load DEM (Elevation) ---
    dem_file = findFile({'Elevation_Matrix.tif', 'elevation.tif', 'DEM.tif', '*.tif'}, 'elevation');
    if isempty(dem_file)
        fprintf('   ✗ ERROR: No elevation/DEM file found.\n');
        return;
    end
    
    try
        dem = double(imread(dem_file));
        dem = preprocessDEM(dem);
        fprintf('   ✓ Loaded DEM: %s (%dx%d pixels)\n', dem_file, size(dem,1), size(dem,2));
    catch ME
        fprintf('   ✗ ERROR reading DEM: %s\n', ME.message);
        return;
    end
    
    % --- Load Satellite/Base Image ---
    img_file = findFile({'Visual_Base.tif', 'satellite.tif', 'base.tif', 'image.tif', '*.tif', '*.png', '*.jpg'}, 'visual');
    if isempty(img_file)
        fprintf('   ⚠ WARNING: No satellite image found. Using DEM for visualization.\n');
        sat_img = createImageFromDEM(dem);
    else
        try
            sat_img = imread(img_file);
            fprintf('   ✓ Loaded Image: %s\n', img_file);
        catch
            fprintf('   ⚠ WARNING: Could not read image. Using DEM visualization.\n');
            sat_img = createImageFromDEM(dem);
        end
    end
    
    success = true;
end

function csv_file = findCSVFile()
    % Find CSV file with coastline data in Dataset folder
    csv_file = '';
    
    % Priority list of expected names (in Dataset folder)
    expected_names = {'Dataset/Coastline_TimeSeries_41Years.csv', ...
                      'Dataset/Coastline_TimeSeries.csv', ...
                      'Dataset/coastline_data.csv', ...
                      'Coastline_TimeSeries_41Years.csv', ...
                      'Coastline_TimeSeries.csv'};
    
    for i = 1:length(expected_names)
        if isfile(expected_names{i})
            csv_file = expected_names{i};
            return;
        end
    end
    
    % Search in Dataset folder first
    if isfolder('Dataset')
        csv_files = dir('Dataset/*.csv');
        if ~isempty(csv_files)
            csv_file = fullfile('Dataset', csv_files(1).name);
            return;
        end
    end
    
    % Fallback: Search in current directory
    csv_files = dir('*.csv');
    if ~isempty(csv_files)
        csv_file = csv_files(1).name;
    end
end

function file_path = findFile(patterns, keyword)
    % Find file matching patterns or containing keyword
    % Searches in Dataset folder first, then current directory
    file_path = '';
    
    % Add Dataset/ prefix to patterns
    dataset_patterns = cellfun(@(x) fullfile('Dataset', x), patterns, 'UniformOutput', false);
    all_patterns = [dataset_patterns, patterns];
    
    % Check exact matches first
    for i = 1:length(all_patterns)
        if ~contains(all_patterns{i}, '*') && isfile(all_patterns{i})
            file_path = all_patterns{i};
            return;
        end
    end
    
    % Search in Dataset folder with wildcards
    if isfolder('Dataset')
        for i = 1:length(patterns)
            if contains(patterns{i}, '*')
                files = dir(fullfile('Dataset', patterns{i}));
                for j = 1:length(files)
                    if contains(lower(files(j).name), keyword)
                        file_path = fullfile('Dataset', files(j).name);
                        return;
                    end
                end
            end
        end
        
        % Return first matching TIF in Dataset folder
        tif_files = dir('Dataset/*.tif');
        for j = 1:length(tif_files)
            if contains(lower(tif_files(j).name), keyword)
                file_path = fullfile('Dataset', tif_files(j).name);
                return;
            end
        end
    end
    
    % Fallback: Search in current directory
    for i = 1:length(patterns)
        if contains(patterns{i}, '*')
            files = dir(patterns{i});
            for j = 1:length(files)
                if contains(lower(files(j).name), keyword)
                    file_path = files(j).name;
                    return;
                end
            end
        end
    end
end

function [years, area] = extractTimeSeriesData(data)
    % Extract year and area columns from table
    years = [];
    area = [];
    
    colNames = data.Properties.VariableNames;
    
    % Find year column
    yearCol = findMatchingColumn(colNames, {'year', 'Year', 'YEAR', 'date', 'time'});
    if isempty(yearCol)
        return;
    end
    
    % Find area column
    areaCol = findMatchingColumn(colNames, {'land_area_m2', 'area', 'land_area', ...
              'Area', 'AREA', 'landarea', 'surface', 'km2', 'sqm'});
    if isempty(areaCol)
        return;
    end
    
    years = data.(yearCol);
    area = data.(areaCol);
    
    % Convert to km² if needed (values > 1e6 are likely in m²)
    if mean(area, 'omitnan') > 1e6
        area = area / 1e6;
    end
    
    % Ensure column vectors
    years = double(years(:));
    area = double(area(:));
end

function colName = findMatchingColumn(colNames, patterns)
    % Find column matching any pattern
    colName = '';
    
    % Exact match
    for i = 1:length(patterns)
        idx = find(strcmpi(colNames, patterns{i}), 1);
        if ~isempty(idx)
            colName = colNames{idx};
            return;
        end
    end
    
    % Partial match
    for i = 1:length(patterns)
        idx = find(contains(lower(colNames), lower(patterns{i})), 1);
        if ~isempty(idx)
            colName = colNames{idx};
            return;
        end
    end
end

function dem = preprocessDEM(dem)
    % Clean and preprocess DEM
    dem(dem < -500) = NaN;
    dem(dem > 5000) = NaN;
    
    % Fill small gaps
    if sum(isnan(dem(:))) < numel(dem) * 0.1
        dem = fillmissing(dem, 'linear');
    end
    
    dem(isnan(dem)) = 0;
    dem(dem < 0) = 0;
    
    % Light smoothing
    if exist('imgaussfilt', 'file')
        dem = imgaussfilt(dem, 0.5);
    end
end

function img = createImageFromDEM(dem)
    % Create RGB image from DEM for visualization
    dem_norm = (dem - min(dem(:))) / (max(dem(:)) - min(dem(:)) + eps);
    img = uint8(255 * cat(3, dem_norm * 0.6, dem_norm * 0.8 + 0.2, dem_norm * 0.4));
end

%% ═══════════════════════════════════════════════════════════════════════
%  DATA PREPROCESSING
%% ═══════════════════════════════════════════════════════════════════════

function [years, area] = preprocessData(years, area)
    % Clean and preprocess time series data
    
    % Remove NaN values
    valid = ~isnan(years) & ~isnan(area) & area > 0;
    years = years(valid);
    area = area(valid);
    
    % Sort by year
    [years, idx] = sort(years);
    area = area(idx);
    
    % Remove duplicates (keep mean)
    [unique_years, ~, ic] = unique(years);
    unique_area = accumarray(ic, area, [], @mean);
    years = unique_years;
    area = unique_area;
    
    % Remove outliers (IQR method)
    Q1 = prctile(area, 25);
    Q3 = prctile(area, 75);
    IQR = Q3 - Q1;
    valid = area >= (Q1 - 2*IQR) & area <= (Q3 + 2*IQR);
    
    if sum(~valid) > 0
        fprintf('   ⚠ Removed %d outliers\n', sum(~valid));
        years = years(valid);
        area = area(valid);
    end
end

%% ═══════════════════════════════════════════════════════════════════════
%  MODEL TRAINING
%% ═══════════════════════════════════════════════════════════════════════

function models = trainModels(years, area)
    % Train multiple regression models
    
    n = length(area);
    k = min(5, max(3, floor(n/4)));
    
    models = struct();
    models.data.years = years;
    models.data.area = area;
    models.data.n = n;
    
    % Normalization parameters
    models.norm.year_mean = mean(years);
    models.norm.year_std = std(years);
    years_norm = (years - models.norm.year_mean) / models.norm.year_std;
    
    % Model 1: Linear
    [p, S, mu] = polyfit(years, area, 1);
    models.linear = createPolyModel(p, S, mu, 'Linear', 1, years, area, k);
    
    % Model 2: Quadratic
    [p, S, mu] = polyfit(years, area, 2);
    models.quadratic = createPolyModel(p, S, mu, 'Quadratic', 2, years, area, k);
    
    % Model 3: Cubic
    if n >= 10
        [p, S, mu] = polyfit(years, area, 3);
        models.cubic = createPolyModel(p, S, mu, 'Cubic', 3, years, area, k);
    else
        models.cubic = models.quadratic;
        models.cubic.name = 'Cubic (fallback)';
    end
    
    % Model 4: Robust Linear
    try
        b = robustfit(years_norm, area);
        models.robust.coeffs = b;
        models.robust.name = 'Robust Linear';
        models.robust.predict = @(y) b(1) + b(2) * (y - models.norm.year_mean) / models.norm.year_std;
        pred = models.robust.predict(years);
        models.robust.r2 = computeR2(area, pred);
        models.robust.rmse = sqrt(mean((area - pred).^2));
    catch
        models.robust = models.linear;
        models.robust.name = 'Robust (fallback)';
    end
    
    % Model 5: Exponential
    try
        models.exponential = fitExponential(years, area);
    catch
        models.exponential = models.quadratic;
        models.exponential.name = 'Exponential (fallback)';
    end
    
    % Calculate ensemble weights
    model_list = {'linear', 'quadratic', 'cubic', 'robust', 'exponential'};
    rmse_vals = zeros(length(model_list), 1);
    
    for i = 1:length(model_list)
        rmse_vals(i) = models.(model_list{i}).rmse;
    end
    
    % Inverse RMSE weighting
    weights = 1 ./ (rmse_vals + 0.01);
    weights = weights / sum(weights);
    
    for i = 1:length(model_list)
        models.(model_list{i}).weight = weights(i);
    end
    
    models.weights = weights;
    models.model_list = model_list;
end

function model = createPolyModel(p, S, mu, name, degree, years, area, k)
    % Create polynomial model structure
    model.coeffs = p;
    model.S = S;
    model.mu = mu;
    model.name = name;
    model.degree = degree;
    model.predict = @(y) polyval(p, y, S, mu);
    
    pred = model.predict(years);
    model.r2 = computeR2(area, pred);
    model.rmse = sqrt(mean((area - pred).^2));
    
    % Cross-validation RMSE
    try
        cv = cvpartition(length(area), 'KFold', k);
        cv_errors = zeros(k, 1);
        for i = 1:k
            train_idx = training(cv, i);
            test_idx = test(cv, i);
            [p_cv, S_cv, mu_cv] = polyfit(years(train_idx), area(train_idx), degree);
            pred_cv = polyval(p_cv, years(test_idx), S_cv, mu_cv);
            cv_errors(i) = sqrt(mean((area(test_idx) - pred_cv).^2));
        end
        model.cv_rmse = mean(cv_errors);
    catch
        model.cv_rmse = model.rmse;
    end
end

function model = fitExponential(years, area)
    % Fit exponential decay model
    year0 = min(years);
    years_shift = years - year0;
    
    % Initial parameters
    a0 = area(1) - area(end);
    b0 = -0.01;
    c0 = area(end);
    
    % Optimize
    opts = optimset('Display', 'off', 'MaxIter', 2000);
    objFun = @(p) sum((area - (p(1) * exp(p(2) * years_shift) + p(3))).^2);
    params = fminsearch(objFun, [a0, b0, c0], opts);
    
    model.params = params;
    model.year0 = year0;
    model.name = 'Exponential';
    model.predict = @(y) params(1) * exp(params(2) * (y - year0)) + params(3);
    
    pred = model.predict(years);
    model.r2 = computeR2(area, pred);
    model.rmse = sqrt(mean((area - pred).^2));
end

function r2 = computeR2(actual, predicted)
    ss_res = sum((actual - predicted).^2);
    ss_tot = sum((actual - mean(actual)).^2);
    r2 = max(0, min(1, 1 - ss_res / (ss_tot + eps)));
end

function displayModelMetrics(models)
    % Display model comparison table
    fprintf('\n   ┌──────────────────────┬──────────┬──────────┬──────────┐\n');
    fprintf('   │ Model                │    R²    │   RMSE   │  Weight  │\n');
    fprintf('   ├──────────────────────┼──────────┼──────────┼──────────┤\n');
    
    for i = 1:length(models.model_list)
        m = models.(models.model_list{i});
        fprintf('   │ %-20s │  %6.4f  │  %6.4f  │  %5.1f%%  │\n', ...
                m.name, m.r2, m.rmse, m.weight * 100);
    end
    fprintf('   └──────────────────────┴──────────┴──────────┴──────────┘\n');
end

%% ═══════════════════════════════════════════════════════════════════════
%  PREDICTION
%% ═══════════════════════════════════════════════════════════════════════

function [predictions, uncertainty] = generatePredictions(models, years, area, config)
    % Generate ensemble predictions with uncertainty
    
    target = config.target_year;
    current_area = area(end);
    
    % Individual predictions
    preds = zeros(length(models.model_list), 1);
    for i = 1:length(models.model_list)
        preds(i) = models.(models.model_list{i}).predict(target);
    end
    
    % Bound predictions to reasonable range
    min_area = current_area * 0.2;
    max_area = current_area * 1.2;
    preds_bounded = max(min_area, min(max_area, preds));
    
    % Weighted ensemble
    ensemble_pred = sum(preds_bounded .* models.weights);
    
    predictions.target_year = target;
    predictions.current_area = current_area;
    predictions.ensemble = ensemble_pred;
    predictions.individual = preds;
    predictions.bounded = preds_bounded;
    
    % IPCC Scenarios
    predictions.scenarios = struct();
    for i = 1:length(config.ipcc_scenarios)
        scenario = config.ipcc_scenarios{i};
        slr = config.slr_rates(i);
        
        % Sea level impact factor (empirical: ~3% loss per 0.1m rise)
        factor = max(0.4, 1 - slr * 0.03);
        predictions.scenarios.(matlab.lang.makeValidName(scenario)) = ensemble_pred * factor;
    end
    
    % Bootstrap uncertainty
    n_boot = config.n_bootstrap;
    boot_preds = zeros(n_boot, 1);
    n = length(years);
    
    rng(42);
    for b = 1:n_boot
        idx = randi(n, n, 1);
        [p, S, mu] = polyfit(years(idx), area(idx), 2);
        pred = polyval(p, target, S, mu);
        boot_preds(b) = max(min_area, min(max_area, pred));
    end
    
    alpha = (1 - config.confidence_level) / 2;
    uncertainty.ci_lower = prctile(boot_preds, alpha * 100);
    uncertainty.ci_upper = prctile(boot_preds, (1 - alpha) * 100);
    uncertainty.std = std(boot_preds);
    uncertainty.model_spread = std(preds_bounded);
    
    fprintf('   Ensemble Prediction for %d: %.2f km²\n', target, ensemble_pred);
    fprintf('   95%% CI: [%.2f, %.2f] km²\n', uncertainty.ci_lower, uncertainty.ci_upper);
    fprintf('   Change from current: %.1f%%\n', (ensemble_pred - current_area) / current_area * 100);
end

%% ═══════════════════════════════════════════════════════════════════════
%  INUNDATION SIMULATION
%% ═══════════════════════════════════════════════════════════════════════

function [flood_masks, sea_levels] = simulateFlooding(dem, predictions, config)
    % Simulate flooding based on predictions
    
    pixel_area_km2 = (config.pixel_resolution^2) / 1e6;
    current_dem_area = sum(dem(:) > 0) * pixel_area_km2;
    
    fprintf('   DEM land area: %.2f km²\n', current_dem_area);
    
    flood_masks = struct();
    sea_levels = struct();
    
    % Ensemble flooding
    [flood_masks.ensemble, sea_levels.ensemble] = ...
        calculateFloodMask(dem, predictions.ensemble, current_dem_area, pixel_area_km2);
    
    % Scenario flooding
    scenarios = fieldnames(predictions.scenarios);
    for i = 1:length(scenarios)
        target_area = predictions.scenarios.(scenarios{i});
        [flood_masks.(scenarios{i}), sea_levels.(scenarios{i})] = ...
            calculateFloodMask(dem, target_area, current_dem_area, pixel_area_km2);
        
        flooded = sum(flood_masks.(scenarios{i})(:)) * pixel_area_km2;
        fprintf('   %s: +%.2fm SLR, %.2f km² flooded\n', ...
                scenarios{i}, sea_levels.(scenarios{i}), flooded);
    end
end

function [flood_mask, sea_level] = calculateFloodMask(dem, target_area, current_area, pixel_area)
    % Calculate flood mask for target area
    
    flood_area = current_area - target_area;
    
    if flood_area <= 0
        flood_mask = false(size(dem));
        sea_level = 0;
        return;
    end
    
    flood_pixels = round(flood_area / pixel_area);
    valid_elev = sort(dem(dem > 0));
    
    if flood_pixels >= length(valid_elev)
        sea_level = max(valid_elev);
        flood_mask = dem > 0;
    elseif flood_pixels <= 0
        sea_level = 0;
        flood_mask = false(size(dem));
    else
        sea_level = valid_elev(flood_pixels);
        flood_mask = (dem > 0) & (dem <= sea_level);
    end
end

%% ═══════════════════════════════════════════════════════════════════════
%  VISUALIZATION
%% ═══════════════════════════════════════════════════════════════════════

function createVisualizations(years, area, models, predictions, uncertainty, ...
                              flood_masks, sat_img, dem, config)
    
    %% Figure 1: Analysis Dashboard
    fig1 = figure('Name', 'Coastline Analysis Dashboard', ...
                  'Position', [50 50 1400 800], 'Color', 'w');
    
    % 1.1 Historical Data + Model Fits
    subplot(2,3,1);
    hold on;
    scatter(years, area, 60, 'k', 'filled', 'DisplayName', 'Historical Data');
    
    t = linspace(min(years), config.target_year, 200);
    colors = lines(length(models.model_list));
    
    for i = 1:length(models.model_list)
        m = models.(models.model_list{i});
        y_pred = m.predict(t);
        y_pred = max(0, min(max(area)*1.5, y_pred));
        plot(t, y_pred, 'LineWidth', 1.5, 'Color', colors(i,:), ...
             'DisplayName', sprintf('%s (R²=%.2f)', m.name, m.r2));
    end
    
    xline(config.target_year, '--r', 'LineWidth', 2, 'HandleVisibility', 'off');
    xlabel('Year'); ylabel('Land Area (km²)');
    title('Model Fits Comparison');
    legend('Location', 'best', 'FontSize', 7);
    grid on; xlim([min(years)-2 config.target_year+5]);
    hold off;
    
    % 1.2 Model Performance
    subplot(2,3,2);
    r2_vals = cellfun(@(x) models.(x).r2, models.model_list);
    weights = cellfun(@(x) models.(x).weight, models.model_list) * 100;
    names = cellfun(@(x) models.(x).name, models.model_list, 'UniformOutput', false);
    
    x = 1:length(names);
    yyaxis left;
    bar(x-0.2, r2_vals, 0.35, 'FaceColor', [0.2 0.6 0.9]);
    ylabel('R² Score'); ylim([0 1.1]);
    
    yyaxis right;
    bar(x+0.2, weights, 0.35, 'FaceColor', [0.9 0.4 0.2]);
    ylabel('Weight (%)');
    
    set(gca, 'XTick', x, 'XTickLabel', names, 'XTickLabelRotation', 45);
    title('Model Performance'); grid on;
    
    % 1.3 Prediction with Uncertainty
    subplot(2,3,3);
    hold on;
    scatter(years, area, 50, 'k', 'filled');
    
    t_future = linspace(min(years), config.target_year, 100);
    trend = models.quadratic.predict(t_future);
    trend = max(0, min(max(area)*1.5, trend));
    plot(t_future, trend, 'b-', 'LineWidth', 2);
    
    errorbar(config.target_year, predictions.ensemble, ...
             predictions.ensemble - uncertainty.ci_lower, ...
             uncertainty.ci_upper - predictions.ensemble, ...
             'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r', 'LineWidth', 2);
    
    xlabel('Year'); ylabel('Land Area (km²)');
    title(sprintf('%d Prediction with 95%% CI', config.target_year));
    grid on; xlim([min(years)-2 config.target_year+5]);
    hold off;
    
    % 1.4 Rate of Change
    subplot(2,3,4);
    if length(years) >= 2
        rate = diff(area) ./ diff(years);
        mid_years = years(1:end-1) + diff(years)/2;
        
        yyaxis left;
        plot(mid_years, rate, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4);
        ylabel('Rate (km²/year)');
        
        yyaxis right;
        pct = (area - area(1)) / area(1) * 100;
        plot(years, pct, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 4);
        ylabel('Cumulative Change (%)');
        
        xlabel('Year');
        title('Rate of Change Analysis');
        legend({'Annual Rate', 'Cumulative %'}, 'Location', 'best');
        grid on;
    end
    
    % 1.5 Residuals
    subplot(2,3,5);
    pred = models.quadratic.predict(years);
    residuals = area - pred;
    scatter(years, residuals, 50, 'b', 'filled');
    hold on;
    yline(0, 'r--', 'LineWidth', 2);
    xlabel('Year'); ylabel('Residual (km²)');
    title('Residual Analysis');
    grid on;
    hold off;
    
    % 1.6 Scenario Comparison
    subplot(2,3,6);
    scenarios = config.ipcc_scenarios;
    vals = [predictions.ensemble];
    for i = 1:length(scenarios)
        vals = [vals; predictions.scenarios.(matlab.lang.makeValidName(scenarios{i}))];
    end
    labels = [{'Ensemble'}; scenarios(:)];
    
    colors_bar = [0.3 0.3 0.8; 0.2 0.7 0.2; 0.9 0.6 0.1; 0.8 0.2 0.2];
    b = bar(vals, 'FaceColor', 'flat');
    b.CData = colors_bar(1:length(vals), :);
    
    set(gca, 'XTickLabel', labels, 'XTickLabelRotation', 45);
    ylabel('Predicted Area (km²)');
    title(sprintf('Scenario Comparison (%d)', config.target_year));
    
    for i = 1:length(vals)
        text(i, vals(i)+max(vals)*0.02, sprintf('%.1f', vals(i)), ...
             'HorizontalAlignment', 'center', 'FontSize', 9);
    end
    grid on;
    
    sgtitle('Coastline Erosion Analysis Dashboard', 'FontSize', 14, 'FontWeight', 'bold');
    
    %% Figure 2: Spatial Analysis
    fig2 = figure('Name', 'Spatial Inundation Analysis', ...
                  'Position', [100 100 1400 500], 'Color', 'w');
    
    % Current
    subplot(1,3,1);
    imshow(sat_img);
    title('Current State', 'FontSize', 12);
    
    % Ensemble Prediction
    subplot(1,3,2);
    plotFloodMap(sat_img, flood_masks.ensemble);
    title(sprintf('Predicted %d (Ensemble)', config.target_year), 'FontSize', 12);
    
    % Worst Case
    subplot(1,3,3);
    worst = matlab.lang.makeValidName(config.ipcc_scenarios{end});
    plotFloodMap(sat_img, flood_masks.(worst));
    title(sprintf('Worst Case (%s)', config.ipcc_scenarios{end}), 'FontSize', 12);
    
    sgtitle(sprintf('Sea Level Rise Impact Analysis (%d)', config.target_year), ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    %% Figure 3: 3D Terrain
    fig3 = figure('Name', '3D Terrain', 'Position', [150 150 800 600], 'Color', 'w');
    plot3DMap(dem, flood_masks.ensemble, config);
    
    fprintf('   ✓ Created 3 visualization figures\n');
end

function plotFloodMap(sat_img, flood_mask)
    % Plot satellite image with flood overlay
    imshow(sat_img);
    hold on;
    
    if any(flood_mask(:))
        overlay = cat(3, ones(size(flood_mask)), ...
                        zeros(size(flood_mask)), ...
                        zeros(size(flood_mask)));
        h = imshow(overlay);
        set(h, 'AlphaData', double(flood_mask) * 0.5);
        
        [B, ~] = bwboundaries(flood_mask, 'noholes');
        for k = 1:length(B)
            plot(B{k}(:,2), B{k}(:,1), 'r-', 'LineWidth', 1.5);
        end
    else
        text(size(sat_img,2)/2, size(sat_img,1)/2, 'No significant flooding', ...
             'HorizontalAlignment', 'center', 'Color', 'g', 'FontSize', 14, ...
             'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);
    end
    hold off;
end

function plot3DMap(dem, flood_mask, config)
    % 3D terrain visualization
    scale = max(1, floor(max(size(dem)) / 250));
    dem_s = dem(1:scale:end, 1:scale:end);
    flood_s = flood_mask(1:scale:end, 1:scale:end);
    
    [rows, cols] = size(dem_s);
    [X, Y] = meshgrid(1:cols, 1:rows);
    
    % Create color map
    C = zeros(rows, cols, 3);
    land = dem_s > 0 & ~flood_s;
    water = dem_s <= 0;
    
    max_elev = max(dem_s(:)) + eps;
    norm_elev = dem_s / max_elev;
    
    % Land - green gradient
    C(:,:,2) = (0.3 + 0.5 * norm_elev) .* land;
    C(:,:,1) = 0.2 * norm_elev .* land;
    C(:,:,3) = 0.1 * land;
    
    % Water - blue
    C(:,:,3) = C(:,:,3) + 0.7 * water;
    C(:,:,2) = C(:,:,2) + 0.3 * water;
    
    % Flooded - red
    C(:,:,1) = C(:,:,1) + 0.8 * flood_s;
    C(:,:,2) = C(:,:,2) .* ~flood_s + 0.1 * flood_s;
    C(:,:,3) = C(:,:,3) .* ~flood_s + 0.1 * flood_s;
    
    surf(X, Y, dem_s, C, 'EdgeColor', 'none');
    
    cb = colorbar;
    cb.Label.String = 'Elevation (m)';
    
    xlabel('X'); ylabel('Y'); zlabel('Elevation (m)');
    title(sprintf('3D Terrain with Projected Flooding (%d)', config.target_year));
    view(45, 30);
    axis tight;
    lighting gouraud;
    camlight('headlight');
end

%% ═══════════════════════════════════════════════════════════════════════
%  REPORT GENERATION
%% ═══════════════════════════════════════════════════════════════════════

function printFinalReport(predictions, uncertainty, sea_levels, config)
    % Print comprehensive final report
    
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║                   PREDICTION REPORT                      ║\n');
    fprintf('╠══════════════════════════════════════════════════════════╣\n');
    fprintf('║ Target Year: %-43d ║\n', config.target_year);
    fprintf('║ Current Land Area: %-37.2f km² ║\n', predictions.current_area);
    fprintf('╠══════════════════════════════════════════════════════════╣\n');
    fprintf('║ ENSEMBLE PREDICTION                                      ║\n');
    fprintf('║   Predicted Area: %-38.2f km² ║\n', predictions.ensemble);
    fprintf('║   95%% Confidence: [%-6.2f, %-6.2f] km²                   ║\n', ...
            uncertainty.ci_lower, uncertainty.ci_upper);
    
    pct_change = (predictions.ensemble - predictions.current_area) / predictions.current_area * 100;
    fprintf('║   Projected Change: %-36.1f%% ║\n', pct_change);
    
    fprintf('╠══════════════════════════════════════════════════════════╣\n');
    fprintf('║ IPCC SCENARIO ANALYSIS                                   ║\n');
    
    for i = 1:length(config.ipcc_scenarios)
        scenario = config.ipcc_scenarios{i};
        key = matlab.lang.makeValidName(scenario);
        area_pred = predictions.scenarios.(key);
        slr = config.slr_rates(i);
        
        fprintf('║   %-10s: +%.1fm SLR → %.2f km²                    ║\n', ...
                scenario, slr, area_pred);
    end
    
    fprintf('╚══════════════════════════════════════════════════════════╝\n');
end